

from __future__ import annotations

import os
import json
import uuid
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import timm
from tqdm import tqdm

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

from torch_geometric.data import Data as GeoData
from torch_geometric.nn import knn_graph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ==========================
# CONFIG (EDIT THESE)
# ==========================

try:
    from config.defaults import DST_DATASET as _DST
except Exception:
    _DST = None

DATAROOT = Path(_DST) if _DST else Path("WRITE_NUSCENES_OOD_ROOT_PATH_HERE")
JSONDIR  = DATAROOT / "v1.0-mini"

ALL_CAMS = [
    "CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
    "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"
]

# Graph parameters (same intent as your script)
KNN_K = 6
INCLUDE_ID_FROM_OOD_FRAMES = True  # same as your build_graphs default
TRAIN_SPLIT = "train"
VAL_SPLIT   = "val"  # in your script val includes OOD nodes

# Embedding parameters (same as your script)
CACHE_DIR  = DATAROOT / ".cache" / "emb_v1"
BATCH_SIZE = 64  # tune for GPU

# VGAE training
EPOCHS = 15
LR = 1e-3

# Output file for later results/viz
OUTDIR = Path("WRITE_CONTEXT_GRAPH_OUTPUT_DIR_HERE")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTDIR / "context_graph_scores.json"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "Device:", device)


# ==========================
# JSON + TABLE LOADING
# ==========================

def load_json(path: Path):
    return json.loads(path.read_text())

def _assert_exists(p: Path, msg: str):
    if not p.exists():
        raise FileNotFoundError(f"{msg}: {p}")

def load_tables():
    _assert_exists(JSONDIR, "JSONDIR missing")
    _assert_exists(JSONDIR / "sample_data.json", "sample_data.json missing")
    _assert_exists(JSONDIR / "sample.json", "sample.json missing")
    _assert_exists(JSONDIR / "scene.json", "scene.json missing")
    _assert_exists(JSONDIR / "sensor.json", "sensor.json missing")
    _assert_exists(JSONDIR / "calibrated_sensor.json", "calibrated_sensor.json missing")
    _assert_exists(JSONDIR / "detection_id.json", "detection_id.json missing")
    _assert_exists(JSONDIR / "detection_novel.json", "detection_novel.json missing")

    sd_rows   = {d["token"]: d for d in load_json(JSONDIR / "sample_data.json")}
    samples   = {s["token"]: s for s in load_json(JSONDIR / "sample.json")}
    scenes    = load_json(JSONDIR / "scene.json")
    sensor_by = {s["token"]: s for s in load_json(JSONDIR / "sensor.json")}
    calib_by  = {c["token"]: c for c in load_json(JSONDIR / "calibrated_sensor.json")}

    gt_id  = load_json(JSONDIR / "detection_id.json")["results"]
    gt_ood = load_json(JSONDIR / "detection_novel.json")["results"]

    return sd_rows, samples, scenes, sensor_by, calib_by, gt_id, gt_ood


sd_rows, samples, scenes, sensor_by, calib_by, gt_id, gt_ood = load_tables()


def channel_of_sd_row(sd_row):
    calib = calib_by[sd_row["calibrated_sensor_token"]]
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


# ==========================
# EMBEDDING BACKBONE (same as your file)
# ==========================

CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    backbone = timm.create_model("dinov2_small", pretrained=True, num_classes=0).to(device).eval()
except Exception:
    backbone = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(device).eval()

FEAT_DIM = backbone.num_features

preproc = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])


def _sanitize_box(box, W, H, min_size=2):
    # clamp + fix ordering + remove NaNs; returns integer box or None
    x0, y0, x1, y1 = [float(v) for v in box]
    if not np.isfinite([x0,y0,x1,y1]).all():
        return None
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    x0 = max(0, min(x0, W-1))
    x1 = max(0, min(x1, W-1))
    y0 = max(0, min(y0, H-1))
    y1 = max(0, min(y1, H-1))
    if (x1-x0) < min_size or (y1-y0) < min_size:
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def _box_key(sd_token: str, sb):
    # stable key for caching
    h = hashlib.md5((sd_token + "_" + "_".join(map(str, sb))).encode("utf-8")).hexdigest()
    return h

def _cache_path(key: str):
    return CACHE_DIR / f"{key}.npy"


def embed_crops_batched(pil_img: Image.Image, boxes, sd_token: str):
    """Return (N, FEAT_DIM) embeddings; invalid boxes -> zero vectors. Uses GPU batching + cache."""
    W, H = pil_img.size
    embs = [None] * len(boxes)
    to_run, run_meta = [], []  # tensors, (i, cache_key)

    for i, box in enumerate(boxes):
        sb = _sanitize_box(box, W, H)
        if sb is None:
            embs[i] = np.zeros((FEAT_DIM,), dtype=np.float32)
            continue
        key = _box_key(sd_token, sb)
        cp = _cache_path(key)
        if cp.exists():
            embs[i] = np.load(cp)
        else:
            crop = pil_img.crop(sb)
            to_run.append(preproc(crop))
            run_meta.append((i, key))

    for start in range(0, len(to_run), BATCH_SIZE):
        batch = torch.stack(to_run[start:start+BATCH_SIZE], dim=0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            Z = backbone(batch)
        Z = Z.float().cpu().numpy()
        for (i, key), z in zip(run_meta[start:start+BATCH_SIZE], Z):
            np.save(_cache_path(key), z)
            embs[i] = z

    for i in range(len(embs)):
        if embs[i] is None:
            embs[i] = np.zeros((FEAT_DIM,), dtype=np.float32)

    return np.stack(embs, axis=0)


# ==========================
# BOX HELPERS (ID/OOD)
# ==========================

def get_boxes_labels_classes_for_sd(sd_token: str):
    """
    From detection_id.json and detection_novel.json:
      - ID label: 0
      - OOD label: 1
    Class names:
      - ID: detection_name
      - OOD: detection_name (often "novel" in your generator)
    """
    id_recs  = gt_id.get(sd_token, [])
    ood_recs = gt_ood.get(sd_token, [])

    boxes, labels, classes = [], [], []

    for r in id_recs:
        boxes.append(r["bbox_2d"])
        labels.append(0)
        classes.append(r.get("detection_name", "id"))

    for r in ood_recs:
        boxes.append(r["bbox_2d"])
        labels.append(1)
        classes.append(r.get("detection_name", "novel"))

    return boxes, labels, classes


# ==========================
# GRAPH BUILDING (same logic as your build_graphs)
# ==========================

def build_graphs(split="train", knn_k=6, include_id_from_ood_frames=True):
    graphs = []

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
                img_path = DATAROOT / fn
                if not img_path.exists():
                    continue

                img = Image.open(img_path).convert("RGB")
                W, H = img.width, img.height

                # gather boxes + labels
                id_boxes  = [b["bbox_2d"] for b in gt_id.get(sd_tok, [])]
                ood_boxes = [b["bbox_2d"] for b in gt_ood.get(sd_tok, [])]
                has_ood   = len(ood_boxes) > 0

                boxes, labels = [], []
                if include_id_from_ood_frames or not has_ood:
                    boxes += id_boxes
                    labels += [0]*len(id_boxes)

                if split != "train":
                    boxes += ood_boxes
                    labels += [1]*len(ood_boxes)

                if not boxes:
                    continue

                # embeddings (batched + cached)
                emb = embed_crops_batched(img, boxes, sd_tok)

                # geometry + pos
                geo, pos = [], []
                for x0,y0,x1,y1 in boxes:
                    cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
                    w, h   = max(1.0, x1-x0), max(1.0, y1-y0)
                    asp, area = w/h, (w*h)/(W*H)
                    geo.append([cx/W, cy/H, w/W, h/H, asp, area])
                    pos.append([cx/W, cy/H])

                X = torch.from_numpy(np.concatenate([emb, np.array(geo, dtype=np.float32)], axis=1))
                P = torch.from_numpy(np.array(pos, dtype=np.float32))
                Y = torch.from_numpy(np.array(labels, dtype=np.int64))

                k = min(knn_k, max(1, len(P)-1))
                ei = knn_graph(P, k=k)

                g = GeoData(x=X, pos=P, y=Y, edge_index=ei)
                g.meta = {"sd_token": sd_tok, "channel": ch, "scene": sc.get("name","")}
                graphs.append(g)

            s_tok = sample["next"]

    return graphs


def is_valid_graph(g: GeoData, min_nodes=2):
    return (g is not None) and (g.x is not None) and (g.x.size(0) >= min_nodes)


def normalize_graphs(graphs):
    # feature normalization per-graph (stable + simple)
    out = []
    for g in graphs:
        if not is_valid_graph(g):
            continue
        x = g.x.float()
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        g.x = (x - mu) / sd
        out.append(g)
    return out


# ==========================
# VGAE (same architecture as your file)
# ==========================

class GEncoder(nn.Module):
    def __init__(self, in_dim, hid=256, z=128):
        super().__init__()
        self.g1 = GCNConv(in_dim, hid)
        self.g2 = GCNConv(hid, z)

    def forward(self, x, ei):
        h = F.relu(self.g1(x, ei))
        z = self.g2(h, ei)
        return z


class VGAE(nn.Module):
    def __init__(self, in_dim, hid=256, z=128):
        super().__init__()
        self.mu = GEncoder(in_dim, hid, z)
        self.lv = GEncoder(in_dim, hid, z)
        self.dec = nn.Sequential(
            nn.Linear(z, hid),
            nn.ReLU(),
            nn.Linear(hid, in_dim),
        )

    def forward(self, x, ei):
        mu = self.mu(x, ei)
        logv = self.lv(x, ei)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mu + eps * std
        xrec = self.dec(z)
        return z, xrec, mu, logv


def vgae_loss(x, xrec, mu, logv, ei, z):
    xrec = torch.nan_to_num(xrec)
    mu = torch.nan_to_num(mu)
    logv = torch.clamp(torch.nan_to_num(logv), min=-10.0, max=10.0)

    feat = F.mse_loss(xrec, x)

    # edge BCE
    if ei.numel() > 0:
        pos = ei
        num_pos = pos.size(1)
        if num_pos > 0:
            neg = negative_sampling(pos, num_nodes=z.size(0), num_neg_samples=num_pos)
            pos_log = (z[pos[0]] * z[pos[1]]).sum(dim=1)
            neg_log = (z[neg[0]] * z[neg[1]]).sum(dim=1)
            pos_tgt = torch.ones_like(pos_log)
            neg_tgt = torch.zeros_like(neg_log)
            logits = torch.cat([pos_log, neg_log], dim=0)
            target = torch.cat([pos_tgt, neg_tgt], dim=0)
            edge = F.binary_cross_entropy_with_logits(torch.nan_to_num(logits), target)
        else:
            edge = torch.tensor(0.0, device=x.device)
    else:
        edge = torch.tensor(0.0, device=x.device)

    kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - torch.exp(logv))
    total = feat + 0.1 * edge + 0.01 * kl
    return total


def train_vgae(train_graphs, epochs=15, lr=1e-3):
    assert len(train_graphs) > 0, "No training graphs."

    in_dim = train_graphs[0].x.size(1)
    model = VGAE(in_dim=in_dim, hid=256, z=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for g in train_graphs:
            x = g.x.to(device)
            ei = g.edge_index.to(device)
            z, xrec, mu, logv = model(x, ei)
            loss = vgae_loss(x, xrec, mu, logv, ei, z)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        print(f"[VGAE] epoch {ep:02d}/{epochs}  loss={np.mean(losses):.4f}")

    model.eval()
    return model


def vgae_node_scores(model_vgae: VGAE, g: GeoData):
    """
    SAME scoring logic from your attached file:
      err = sum((x - xrec)^2)
      mean_sim = mean(sigmoid(dot(z_u, z_v))) over outgoing edges
      score = err + (1 - mean_sim)
    """
    x = g.x.to(device)
    ei = g.edge_index.to(device)

    z, xrec, _, _ = model_vgae(x, ei)
    err = ((x - xrec) ** 2).sum(dim=1)

    src, dst = ei
    sim = torch.sigmoid((z[src] * z[dst]).sum(dim=1))

    deg = torch.zeros(z.size(0), device=z.device).scatter_add_(
        0, src, torch.ones_like(src, dtype=torch.float32)
    )
    agg = torch.zeros(z.size(0), device=z.device).scatter_add_(0, src, sim)
    mean_sim = torch.where(deg > 0, agg / deg, torch.zeros_like(deg))

    return (err + (1.0 - mean_sim)).detach().cpu().numpy()


# ==========================
# CONTEXT MAHALANOBIS (same idea as your file)
# ==========================

def neighbor_mean_features(g: GeoData):
    """
    Returns (X, C, y):
      X: node features
      C: neighbor mean features
      y: labels
    """
    X = g.x.float().cpu()
    y = g.y.long().cpu().numpy()

    ei = g.edge_index
    if ei.numel() == 0:
        return X.numpy(), X.numpy(), y

    src, dst = ei[0].cpu(), ei[1].cpu()
    N, D = X.shape

    ones = torch.ones_like(src, dtype=torch.float32)
    deg  = torch.zeros(N, dtype=torch.float32).scatter_add_(0, src, ones)
    sumN = torch.zeros(N, D, dtype=torch.float32).index_add_(0, src, X[dst])

    C = torch.where(deg.view(-1,1) > 0,
                    sumN / deg.clamp_min(1.0).view(-1,1),
                    X)
    return X.numpy(), C.numpy(), y


def fpr_at_tpr(y, s, target=0.95):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmin(np.abs(tpr - target)))
    return float(fpr[i]), float(tpr[i]), float(thr[i])


def fit_ctx_mahalanobis(train_graphs):
    """
    Fit LedoitWolf covariance on TRAIN ID nodes only using [X ; neighbor_mean(X)].
    """
    X_tr, C_tr = [], []
    for g in train_graphs:
        x, c, y = neighbor_mean_features(g)
        m = (y == 0)
        if m.any():
            X_tr.append(x[m])
            C_tr.append(c[m])

    Z_tr = np.concatenate([np.concatenate(X_tr,0), np.concatenate(C_tr,0)], axis=1)

    mu = Z_tr.mean(0, keepdims=True)
    sd = Z_tr.std(0, keepdims=True).clip(1e-3)
    Zs = (Z_tr - mu) / sd

    cov = LedoitWolf().fit(Zs)
    mu_loc = cov.location_
    prec = cov.precision_

    return mu, sd, mu_loc, prec


def maha2(Z, mu, sd, mu_loc, prec):
    Zs = (Z - mu) / sd
    d = Zs - mu_loc
    return np.einsum("nd,dd,nd->n", d, prec, d)


# ==========================
# MAIN
# ==========================

def main():
    print("Building graphs …")
    train_graphs = build_graphs(split="train", knn_k=KNN_K, include_id_from_ood_frames=INCLUDE_ID_FROM_OOD_FRAMES)
    val_graphs   = build_graphs(split="val",   knn_k=KNN_K, include_id_from_ood_frames=INCLUDE_ID_FROM_OOD_FRAMES)

    train_graphs = normalize_graphs(train_graphs)
    val_graphs   = normalize_graphs(val_graphs)

    print(f"After filtering: {len(train_graphs)} train | {len(val_graphs)} val")
    if len(train_graphs) == 0 or len(val_graphs) == 0:
        raise RuntimeError("No graphs built. Check dataset paths and detection_*.json.")

    # === VGAE TRAIN ===
    model_vgae = train_vgae(train_graphs, epochs=EPOCHS, lr=LR)

    # === VGAE SCORE ===
    vgae_scores_all, vgae_labels_all = [], []
    per_cam_v_s, per_cam_v_y = defaultdict(list), defaultdict(list)

    node_records = []  # save for later viz

    for g in tqdm(val_graphs, desc="VGAE scoring"):
        s = vgae_node_scores(model_vgae, g)
        y = g.y.cpu().numpy()
        vgae_scores_all.append(s)
        vgae_labels_all.append(y)

        cam = g.meta["channel"]
        per_cam_v_s[cam].append(s)
        per_cam_v_y[cam].append(y)

        # store per-node info for later plotting/visualization
        sd = g.meta["sd_token"]
        boxes, labs, clss = get_boxes_labels_classes_for_sd(sd)
        # safety: lengths may mismatch in rare cases
        n = min(len(boxes), len(s))
        for i in range(n):
            node_records.append({
                "sd_token": sd,
                "channel": cam,
                "scene": g.meta.get("scene",""),
                "bbox_2d": [float(v) for v in boxes[i]],
                "label": int(labs[i]),
                "class": str(clss[i]),
                "vgae_score": float(s[i]),
            })

    vgae_scores_all = np.concatenate(vgae_scores_all)
    vgae_labels_all = np.concatenate(vgae_labels_all)

    if len(set(vgae_labels_all.tolist())) >= 2:
        auc = roc_auc_score(vgae_labels_all, vgae_scores_all)
        fpr95, tpr95, _ = fpr_at_tpr(vgae_labels_all, vgae_scores_all, 0.95)
        print(f"[VGAE] AUROC={auc:.4f}  FPR@95={fpr95:.4f} (TPR≈{tpr95:.3f})")
    else:
        print("[VGAE] Not enough positives/negatives for AUROC.")

    print("\n[VGAE] per-camera:")
    for cam in sorted(per_cam_v_s.keys()):
        s = np.concatenate(per_cam_v_s[cam])
        y = np.concatenate(per_cam_v_y[cam])
        if len(set(y.tolist())) < 2:
            print(f"{cam:16s} — not enough pos/neg  n={len(y)}")
            continue
        auc = roc_auc_score(y, s)
        f95, _, _ = fpr_at_tpr(y, s, 0.95)
        print(f"{cam:16s} AUROC={auc:.4f}  FPR@95={f95:.4f}  n={len(y)}")

    # === CONTEXT MAHALANOBIS ===
    print("\nFitting Context Mahalanobis …")
    mu, sd, mu_loc, prec = fit_ctx_mahalanobis(train_graphs)

    scores_all, labels_all = [], []
    per_cam_scores, per_cam_labels = defaultdict(list), defaultdict(list)

    for g in tqdm(val_graphs, desc="Ctx-Mahalanobis"):
        x, c, y = neighbor_mean_features(g)
        Z = np.concatenate([x, c], axis=1)
        s = maha2(Z, mu, sd, mu_loc, prec)
        scores_all.append(s); labels_all.append(y)

        cam = g.meta["channel"]
        per_cam_scores[cam].append(s)
        per_cam_labels[cam].append(y)

        # attach ctx score to saved nodes
        # (match by sd_token order; keep best effort)
        # We add later by sd_token join in results_viz script if needed.

    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)

    if len(set(labels_all.tolist())) >= 2:
        auroc = roc_auc_score(labels_all, scores_all)
        fpr95, tpr95, _ = fpr_at_tpr(labels_all, scores_all, 0.95)
        print(f"[Ctx-Mahalanobis] AUROC={auroc:.4f}  FPR@95={fpr95:.4f} (TPR≈{tpr95:.3f})")
    else:
        print("[Ctx-Mahalanobis] Not enough positives/negatives for AUROC.")

    print("\n[Ctx-Mahalanobis] per-camera:")
    for cam in sorted(per_cam_scores.keys()):
        s = np.concatenate(per_cam_scores[cam]); y = np.concatenate(per_cam_labels[cam])
        if len(set(y.tolist())) < 2:
            print(f"{cam:16s} — not enough pos/neg  n={len(y)}")
            continue
        auc = roc_auc_score(y, s)
        f95, _, _ = fpr_at_tpr(y, s, 0.95)
        print(f"{cam:16s} AUROC={auc:.4f}  FPR@95={f95:.4f}  n={len(y)}")

    # Save outputs for results/visualization script
    out = {
        "meta": {
            "dataroot": str(DATAROOT.resolve()) if DATAROOT.exists() else str(DATAROOT),
            "knn_k": KNN_K,
            "feat_dim": FEAT_DIM,
            "epochs": EPOCHS,
        },
        "node_records": node_records,
        "summary": {
            "vgae": {
                "auroc": float(roc_auc_score(vgae_labels_all, vgae_scores_all)) if len(set(vgae_labels_all.tolist())) >= 2 else None
            },
            "ctx_mahalanobis": {
                "auroc": float(roc_auc_score(labels_all, scores_all)) if len(set(labels_all.tolist())) >= 2 else None
            }
        }
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved node scores → {OUT_JSON}")


if __name__ == "__main__":
    main()
