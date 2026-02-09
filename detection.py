

import os, json, time, math, random, csv, warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageFile

from sklearn.metrics import roc_auc_score, roc_curve

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


# ======================
# CONFIG (EDIT THESE)
# ======================

# Option A: import from your config/defaults.py (recommended)
try:
    from config.defaults import DST_DATASET as _DST_DATASET
except Exception:
    _DST_DATASET = None

# nuScenes OOD dataset root (your modified dataset root)
# Example: /path/to/NuScenesMiniNovel
# If you use config/defaults.py, DST_DATASET will be used automatically.
NUSCENES_OOD_ROOT = Path(_DST_DATASET) if _DST_DATASET else Path("WRITE_NUSCENES_OOD_ROOT_PATH_HERE")

# nuScenes json directory name inside dataset root
JSONDIR_NAME = "v1.0-mini"

# Limit number of images for speed (same role as your script)
MAX_IMAGES = 500

# Matching / filtering thresholds (same roles as your script)
IOU_MATCH = 0.5
SCORE_THRESH = 0.05

# CSV output directory (placeholder; do not use personal paths)
CSV_DIR = Path("WRITE_RESULTS_OUTPUT_DIR_HERE")  # e.g., Path("results/")
CSV_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = CSV_DIR / "ood_report.csv"


# Cameras (same as your script)
ALL_CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}")


# ======================
# DATASET LOADING
# ======================

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def load_frame_index(dataroot: Path):
    """
    Same idea as your notebook:
    Build a list of frames:
      (img_path, sample_data_token, camera_channel, id_boxes[N,4], ood_boxes[M,4])
    """
    jsondir = dataroot / JSONDIR_NAME

    sd_rows = {d["token"]: d for d in load_json(jsondir / "sample_data.json")}
    samples = {s["token"]: s for s in load_json(jsondir / "sample.json")}
    scenes  = load_json(jsondir / "scene.json")
    calib_by  = {c["token"]: c for c in load_json(jsondir / "calibrated_sensor.json")}
    sensor_by = {s["token"]: s for s in load_json(jsondir / "sensor.json")}

    id_path  = jsondir / "detection_id.json"
    ood_path = jsondir / "detection_novel.json"
    gt_id  = load_json(id_path)["results"] if id_path.exists() else {}
    gt_ood = load_json(ood_path)["results"] if ood_path.exists() else {}

    def channel_of_sd_row(sd_row):
        calib  = calib_by[sd_row["calibrated_sensor_token"]]
        sensor = sensor_by[calib["sensor_token"]]
        return sensor["channel"]

    # sample_token -> {CAM_* -> sample_data_token}
    sample_to_ch2sd = {}
    for sd in sd_rows.values():
        ch = channel_of_sd_row(sd)
        if not ch.startswith("CAM_"):
            continue
        st = sd["sample_token"]
        sample_to_ch2sd.setdefault(st, {})[ch] = sd["token"]

    frames = []
    for sc in scenes:
        s_tok = sc["first_sample_token"]
        while s_tok:
            sample = samples[s_tok]
            ch2sd  = sample_to_ch2sd.get(s_tok, {})

            for ch in ALL_CAMS:
                sd_tok = ch2sd.get(ch)
                if not sd_tok:
                    continue

                fn = sd_rows[sd_tok]["filename"]
                img_path = dataroot / fn
                if not img_path.exists():
                    continue

                id_boxes  = [b["bbox_2d"] for b in gt_id.get(sd_tok, [])]
                ood_boxes = [b["bbox_2d"] for b in gt_ood.get(sd_tok, [])]

                frames.append((
                    str(img_path),                               # 0
                    sd_tok,                                      # 1
                    ch,                                          # 2
                    np.array(id_boxes,  dtype=np.float32),       # 3
                    np.array(ood_boxes, dtype=np.float32),       # 4
                ))

            s_tok = sample["next"]

    return frames


def pick_subset(frames, k=None, seed=13):
    if (k is None) or (k >= len(frames)):
        return frames
    rng = random.Random(seed)
    idx = list(range(len(frames)))
    rng.shuffle(idx)
    return [frames[i] for i in idx[:k]]


# ======================
# MODELS (same families)
# ======================

to_tensor = T.ToTensor()

def load_frcnn_r50():
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    return m.to(device).eval()

def load_frcnn_mbv3():
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    return m.to(device).eval()

def load_retinanet_r50():
    m = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    return m.to(device).eval()

def load_ssdlite_mbv3():
    m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    return m.to(device).eval()

def load_ssd300_vgg16():
    m = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
    return m.to(device).eval()

def infer_torchvision_detector(model, pil_img):
    x = to_tensor(pil_img).to(device)
    with torch.no_grad():
        out = model([x])[0]
    boxes  = out["boxes"].detach().float().cpu().numpy().astype(np.float32)
    scores = out["scores"].detach().float().cpu().numpy().astype(np.float32)
    return boxes, scores, "conf"


# YOLO (optional, same approach as your notebook)
try:
    from ultralytics import YOLO
    _has_yolo = True
except Exception:
    _has_yolo = False

def load_yolov8n():
    if not _has_yolo:
        return None
    return YOLO("yolov8n.pt").to(device)

def load_yolov8s():
    if not _has_yolo:
        return None
    return YOLO("yolov8s.pt").to(device)

def infer_yolov8(model, pil_img):
    im = np.array(pil_img.convert("RGB"))
    ydev = 0 if device.type == "cuda" else "cpu"
    r = model.predict(source=im, verbose=False, conf=0.001, device=ydev)[0]
    if r is None or r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), "conf"

    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)

    # If ultralytics returns probs, treat MSP as score (same intent as notebook)
    if hasattr(r, "probs") and r.probs is not None:
        probs = r.probs.data.cpu().numpy()
        scores = probs.max(axis=1).astype(np.float32)
        return xyxy, scores, "MSP"
    else:
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        return xyxy, scores, "conf"


# ======================
# METRICS (same as notebook)
# ======================

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Na, Nb = a.shape[0], b.shape[0]
    if Na == 0 or Nb == 0:
        return np.zeros((Na, Nb), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)

    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def ap50_single_class(all_scores, all_tp, total_gt_pos):
    """
    Same AP computation style as notebook (all-points interpolation).
    Returns: (AP, P@bestF1, R@bestF1)
    """
    if len(all_scores) == 0:
        return 0.0, 0.0, 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order].astype(np.float32)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / max(1, total_gt_pos)
    precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))

    # all-points interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

    # pick best F1 operating point for P/R reporting
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_i = int(np.argmax(f1)) if f1.size else 0
    return ap, float(precision[best_i]) if precision.size else 0.0, float(recall[best_i]) if recall.size else 0.0


def compute_auroc_fpr95(scores, labels):
    """
    Compute AUROC and FPR@95 using sklearn roc_curve.
    Returns (auroc, fpr95) or (nan, nan) if not computable.
    """
    if len(scores) < 2:
        return float("nan"), float("nan")
    if len(set(labels)) < 2:
        return float("nan"), float("nan")

    au = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    # FPR@95: smallest FPR where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if idx.size else float("nan")
    return au, fpr95


# ======================
# EVAL (same logic)
# ======================

def evaluate_dataset(frames, model_name, model, infer_fn, compute_ood_roc=True):
    """
    Mirrors your notebook logic:
    - AP on ID GT only
    - OOD-FP = overlap any OOD GT
    - AUROC per-GT using score=1-conf, missed ID->0, missed OOD->1
    - per-camera breakdown
    """
    all_scores, all_tp = [], []
    total_id_gt = 0

    det_ood_hits = 0
    total_dets = 0

    roc_scores_glob, roc_labels_glob = [], []

    per_cam = {
        cam: {
            "scores": [], "tp": [], "id_gt": 0,
            "OOD_FP_hits": 0, "detections": 0,
            "roc_scores": [], "roc_labels": [], "N": 0
        }
        for cam in ALL_CAMS
    }

    have_any_ood = any(len(fr[4]) > 0 for fr in frames)
    t0 = time.time()

    for (img_path, sd_tok, cam, id_boxes, ood_boxes) in frames:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        boxes, conf_like, score_type = infer_fn(model, img)

        if conf_like is None or len(conf_like) == 0:
            keep = np.zeros((0,), dtype=bool)
        else:
            keep = conf_like >= SCORE_THRESH

        boxes = boxes[keep] if boxes.size else boxes
        conf_like = conf_like[keep] if (conf_like is not None and keep.size) else conf_like

        # --- ID AP accounting (global) ---
        total_id_gt += int(id_boxes.shape[0])
        used = np.zeros((id_boxes.shape[0],), dtype=bool)

        if boxes.shape[0] > 0 and id_boxes.shape[0] > 0:
            IoU = iou_xyxy(boxes, id_boxes)
            order = np.argsort(-conf_like)
            for di in order:
                j = int(np.argmax(IoU[di]))
                if IoU[di, j] >= IOU_MATCH and not used[j]:
                    all_scores.append(float(conf_like[di])); all_tp.append(1); used[j] = True
                else:
                    all_scores.append(float(conf_like[di])); all_tp.append(0)
        else:
            for s in (conf_like if conf_like is not None else []):
                all_scores.append(float(s)); all_tp.append(0)

        # --- ID AP accounting (per-camera) ---
        cam_used = np.zeros((id_boxes.shape[0],), dtype=bool)
        if boxes.shape[0] > 0 and id_boxes.shape[0] > 0:
            IoU = iou_xyxy(boxes, id_boxes)
            order = np.argsort(-conf_like)
            for di in order:
                j = int(np.argmax(IoU[di]))
                if IoU[di, j] >= IOU_MATCH and not cam_used[j]:
                    per_cam[cam]["scores"].append(float(conf_like[di]))
                    per_cam[cam]["tp"].append(1)
                    cam_used[j] = True
                else:
                    per_cam[cam]["scores"].append(float(conf_like[di]))
                    per_cam[cam]["tp"].append(0)
        else:
            for s in (conf_like if conf_like is not None else []):
                per_cam[cam]["scores"].append(float(s))
                per_cam[cam]["tp"].append(0)

        per_cam[cam]["id_gt"] += int(id_boxes.shape[0])

        # --- OOD-FP: det overlaps any OOD GT ---
        total_dets += int(boxes.shape[0])
        per_cam[cam]["detections"] += int(boxes.shape[0])

        if boxes.shape[0] > 0 and ood_boxes.shape[0] > 0:
            IoU_ood = iou_xyxy(boxes, ood_boxes)
            hits = (IoU_ood.max(axis=1) >= IOU_MATCH).sum()
            det_ood_hits += int(hits)
            per_cam[cam]["OOD_FP_hits"] += int(hits)

        # --- AUROC/FPR@95 per-GT ---
        if compute_ood_roc and have_any_ood:
            if boxes.shape[0] > 0:
                IoU_id  = iou_xyxy(id_boxes,  boxes) if id_boxes.shape[0]  > 0 else np.zeros((0, boxes.shape[0]), dtype=np.float32)
                IoU_ood = iou_xyxy(ood_boxes, boxes) if ood_boxes.shape[0] > 0 else np.zeros((0, boxes.shape[0]), dtype=np.float32)
            else:
                IoU_id  = np.zeros((id_boxes.shape[0],  0), dtype=np.float32)
                IoU_ood = np.zeros((ood_boxes.shape[0], 0), dtype=np.float32)

            det_ood_score = (1.0 - conf_like) if (conf_like is not None and len(conf_like) > 0) else np.array([], dtype=np.float32)

            # ID GT -> label 0
            for gi in range(id_boxes.shape[0]):
                if IoU_id.shape[1] > 0 and IoU_id[gi].max() >= IOU_MATCH:
                    di = int(IoU_id[gi].argmax())
                    s = float(det_ood_score[di]) if det_ood_score.size > 0 else 0.0
                    roc_scores_glob.append(s); roc_labels_glob.append(0)
                    per_cam[cam]["roc_scores"].append(s); per_cam[cam]["roc_labels"].append(0)
                else:
                    roc_scores_glob.append(0.0); roc_labels_glob.append(0)
                    per_cam[cam]["roc_scores"].append(0.0); per_cam[cam]["roc_labels"].append(0)

            # OOD GT -> label 1
            for go in range(ood_boxes.shape[0]):
                if IoU_ood.shape[1] > 0 and IoU_ood[go].max() >= IOU_MATCH:
                    di = int(IoU_ood[go].argmax())
                    s = float(det_ood_score[di]) if det_ood_score.size > 0 else 1.0
                    roc_scores_glob.append(s); roc_labels_glob.append(1)
                    per_cam[cam]["roc_scores"].append(s); per_cam[cam]["roc_labels"].append(1)
                else:
                    roc_scores_glob.append(1.0); roc_labels_glob.append(1)
                    per_cam[cam]["roc_scores"].append(1.0); per_cam[cam]["roc_labels"].append(1)

        per_cam[cam]["N"] += 1

    # overall AP/P/R
    ap, p_at, r_at = ap50_single_class(all_scores, all_tp, total_id_gt)

    # OOD-FP rate
    ood_fp_rate = (det_ood_hits / max(1, total_dets)) if have_any_ood else float("nan")

    # overall ROC
    auroc, fpr95 = compute_auroc_fpr95(roc_scores_glob, roc_labels_glob) if (compute_ood_roc and have_any_ood) else (float("nan"), float("nan"))

    res = {
        "model": model_name,
        "AP50": ap,
        "P@0.5": p_at,
        "R@0.5": r_at,
        "OOD_FP_rate": ood_fp_rate,
        "AUROC": auroc,
        "FPR@95": fpr95,
        "N_imgs": len(frames),
        "time_s": float(time.time() - t0),
        "per_camera": {},
    }

    # per-camera stats
    for cam in ALL_CAMS:
        c = per_cam[cam]
        cap, cp, cr = ap50_single_class(c["scores"], c["tp"], c["id_gt"])
        ood_fp = (c["OOD_FP_hits"] / max(1, c["detections"])) if have_any_ood else float("nan")
        cau, cfpr = compute_auroc_fpr95(c["roc_scores"], c["roc_labels"]) if (compute_ood_roc and have_any_ood) else (float("nan"), float("nan"))
        res["per_camera"][cam] = {
            "AP50": cap, "P@0.5": cp, "R@0.5": cr,
            "OOD_FP": ood_fp,
            "AUROC": cau,
            "FPR@95": cfpr,
            "N": c["N"],
        }

    return res


def camera_average(res):
    """
    Camera-average for easy comparison (same intent as notebook).
    """
    vals = {"AP50": [], "P@0.5": [], "R@0.5": [], "OOD_FP": [], "AUROC": [], "FPR@95": []}
    for cam in ALL_CAMS:
        c = res["per_camera"][cam]
        vals["AP50"].append(c["AP50"])
        vals["P@0.5"].append(c["P@0.5"])
        vals["R@0.5"].append(c["R@0.5"])
        if not (isinstance(c["OOD_FP"], float) and math.isnan(c["OOD_FP"])):
            vals["OOD_FP"].append(c["OOD_FP"])
        if not (isinstance(c["AUROC"], float) and math.isnan(c["AUROC"])):
            vals["AUROC"].append(c["AUROC"])
        if not (isinstance(c["FPR@95"], float) and math.isnan(c["FPR@95"])):
            vals["FPR@95"].append(c["FPR@95"])

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "AP50": mean_or_nan(vals["AP50"]),
        "P@0.5": mean_or_nan(vals["P@0.5"]),
        "R@0.5": mean_or_nan(vals["R@0.5"]),
        "OOD_FP": mean_or_nan(vals["OOD_FP"]),
        "AUROC": mean_or_nan(vals["AUROC"]),
        "FPR@95": mean_or_nan(vals["FPR@95"]),
    }


# ======================
# MAIN RUN (same spirit)
# ======================

def main():
    assert NUSCENES_OOD_ROOT.exists(), (
        f"NUSCENES_OOD_ROOT does not exist: {NUSCENES_OOD_ROOT}\n"
        "Edit NUSCENES_OOD_ROOT in detection.py or set DST_DATASET in config/defaults.py."
    )

    frames = load_frame_index(NUSCENES_OOD_ROOT)
    frames = pick_subset(frames, k=MAX_IMAGES, seed=13)
    print(f"Loaded {len(frames)} frames (max={MAX_IMAGES}).")

    # Model registry (same models as your notebook block)
    model_specs = [
        ("FasterRCNN_R50",  load_frcnn_r50,     infer_torchvision_detector),
        ("FasterRCNN_MBV3", load_frcnn_mbv3,    infer_torchvision_detector),
        ("RetinaNet_R50",   load_retinanet_r50, infer_torchvision_detector),
        ("SSDLite_MBV3",    load_ssdlite_mbv3,  infer_torchvision_detector),
        ("SSD300_VGG16",    load_ssd300_vgg16,  infer_torchvision_detector),
    ]

    if _has_yolo:
        model_specs += [
            ("YOLOv8n", load_yolov8n, infer_yolov8),
            ("YOLOv8s", load_yolov8s, infer_yolov8),
        ]
    else:
        print("Ultralytics not available → skipping YOLO models.")

    rows_for_csv = []

    for name, loader, infer_fn in model_specs:
        print(f"\n=== Evaluating: {name} ===")
        try:
            model = loader()
            if model is None:
                print(f"Skipped {name} (not available).")
                continue

            res = evaluate_dataset(frames, name, model, infer_fn, compute_ood_roc=True)
            cam_avg = camera_average(res)

            # Print summary
            print(f"Overall: AP50={res['AP50']:.4f}, P={res['P@0.5']:.4f}, R={res['R@0.5']:.4f}, "
                  f"OOD-FP={res['OOD_FP_rate'] if not math.isnan(res['OOD_FP_rate']) else '—'}, "
                  f"AUROC={res['AUROC'] if not math.isnan(res['AUROC']) else '—'}, "
                  f"FPR@95={res['FPR@95'] if not math.isnan(res['FPR@95']) else '—'}, "
                  f"N={res['N_imgs']}, Time(s)={res['time_s']:.2f}")

            # Build CSV row (same fields style)
            row = {
                "model": name,
                "overall_AP50": res["AP50"],
                "overall_P@0.5": res["P@0.5"],
                "overall_R@0.5": res["R@0.5"],
                "overall_OOD_FP": res["OOD_FP_rate"] if not math.isnan(res["OOD_FP_rate"]) else "",
                "overall_AUROC": res["AUROC"] if not math.isnan(res["AUROC"]) else "",
                "overall_FPR@95": res["FPR@95"] if not math.isnan(res["FPR@95"]) else "",
                "N": res["N_imgs"],
                "time_s": res["time_s"],
                "camera_avg_AP50": cam_avg["AP50"],
                "camera_avg_P@0.5": cam_avg["P@0.5"],
                "camera_avg_R@0.5": cam_avg["R@0.5"],
                "camera_avg_OOD_FP": cam_avg["OOD_FP"] if not math.isnan(cam_avg["OOD_FP"]) else "",
                "camera_avg_AUROC": cam_avg["AUROC"] if not math.isnan(cam_avg["AUROC"]) else "",
                "camera_avg_FPR@95": cam_avg["FPR@95"] if not math.isnan(cam_avg["FPR@95"]) else "",
            }

            for cam in ALL_CAMS:
                c = res["per_camera"][cam]
                row[f"{cam}_AP50"] = c["AP50"]
                row[f"{cam}_P@0.5"] = c["P@0.5"]
                row[f"{cam}_R@0.5"] = c["R@0.5"]
                row[f"{cam}_OOD_FP"] = c["OOD_FP"] if not (isinstance(c["OOD_FP"], float) and math.isnan(c["OOD_FP"])) else ""
                row[f"{cam}_AUROC"] = c["AUROC"] if not (isinstance(c["AUROC"], float) and math.isnan(c["AUROC"])) else ""
                row[f"{cam}_FPR@95"] = c["FPR@95"] if not (isinstance(c["FPR@95"], float) and math.isnan(c["FPR@95"])) else ""
                row[f"{cam}_N"] = c["N"]

            rows_for_csv.append(row)

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

        # free VRAM between models
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save CSV
    if rows_for_csv:
        fieldnames = list(rows_for_csv[0].keys())
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_for_csv:
                writer.writerow(r)
        print(f"\nSaved CSV → {CSV_OUT}")

    print("\nNotes:")
    print("• AP@0.5/P/R use only ID GT (single-class, greedy 1–1 matches at IoU≥0.5).")
    print("• OOD-FP is the fraction of detections that overlap any OOD GT (IoU≥0.5).")
    print("• AUROC/FPR@95 use per-GT scores: matched → 1−confidence (MSP for YOLO if available),")
    print("  missed ID → 0.0, missed OOD → 1.0. Reported overall and per camera, plus camera-average.")


if __name__ == "__main__":
    main()
