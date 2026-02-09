

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt



# Where detection.py wrote the CSV
DETECTION_CSV = Path("WRITE_PATH_TO_ood_report.csv_HERE")  # e.g., Path("results/ood_report.csv")

# Where context_graph.py wrote scores JSON
CONTEXT_JSON = Path("WRITE_PATH_TO_context_graph_scores.json_HERE")  # e.g., Path("out/context_graph_scores.json")

# nuScenes root (your DST dataset root)
# Used only for loading images for overlay examples
NUSCENES_ROOT = Path("WRITE_NUSCENES_OOD_ROOT_PATH_HERE")  # e.g., Path("/path/to/NuScenesMiniNovel")

JSONDIR_NAME = "v1.0-mini"  # inside nuScenes root

# Output directory
OUTDIR = Path("WRITE_RESULTS_VIZ_OUTPUT_DIR_HERE")  # e.g., Path("results_viz/")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Top-K examples to save per method
TOPK = 30

# Whether to save plots as PNG
SAVE_PLOTS = True


def load_json(p: Path):
    return json.loads(p.read_text())

def safe_float(x):
    try:
        if x == "" or x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def try_load_font(size=18):
    # optional, falls back gracefully
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return None

FONT = try_load_font(18)

def draw_boxes_on_image(img: Image.Image, boxes, labels=None, texts=None, width=3):
    """
    boxes: list of [x0,y0,x1,y1]
    labels: optional list 0/1
    texts: optional list of strings
    """
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = map(float, b)
        # label-based outline thickness only; do not force specific colors if not needed
        # We'll use default outline color differences lightly:
        outline = "white"
        if labels is not None:
            outline = "red" if int(labels[i]) == 1 else "lime"
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)

        if texts is not None:
            t = texts[i]
            # background box for readability
            tx, ty = x0 + 2, max(0, y0 - 22)
            if FONT:
                tw, th = draw.textbbox((tx, ty), t, font=FONT)[2:]
            else:
                tw, th = draw.textbbox((tx, ty), t)[2:]
            draw.rectangle([tx, ty, tx + (tw - tx) + 6, ty + (th - ty) + 4], fill=(0, 0, 0, 160))
            draw.text((tx + 3, ty + 2), t, fill="white", font=FONT)
    return img

def path_from_sample_data(sd_token: str, sample_data_rows: dict) -> Path | None:
    row = sample_data_rows.get(sd_token)
    if not row:
        return None
    return NUSCENES_ROOT / row["filename"]

def load_sample_data_index():
    """
    Build sd_token -> row mapping from sample_data.json.
    """
    p = NUSCENES_ROOT / JSONDIR_NAME / "sample_data.json"
    if not p.exists():
        return {}
    rows = load_json(p)
    return {r["token"]: r for r in rows}




def detection_summary_and_plots(csv_path: Path):
    if not csv_path.exists():
        print(f"[WARN] Detection CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # normalize numeric columns of interest
    for col in ["overall_AP50","overall_AUROC","overall_FPR@95","overall_OOD_FP",
                "camera_avg_AP50","camera_avg_AUROC","camera_avg_FPR@95","camera_avg_OOD_FP"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    print("\n=== Detection Summary (Overall) ===")
    show_cols = [c for c in ["model","overall_AP50","overall_OOD_FP","overall_AUROC","overall_FPR@95","N","time_s"] if c in df.columns]
    if show_cols:
        print(df[show_cols].sort_values(by="overall_AUROC", ascending=False).to_string(index=False))

    # Per-camera AUROC plot for each model
    cams = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
    per_cam_auroc_cols = [f"{c}_AUROC" for c in cams if f"{c}_AUROC" in df.columns]

    if per_cam_auroc_cols:
        # melt to long format
        dfl = df.melt(id_vars=["model"], value_vars=per_cam_auroc_cols, var_name="camera", value_name="AUROC")
        dfl["camera"] = dfl["camera"].str.replace("_AUROC","", regex=False)
        dfl["AUROC"] = dfl["AUROC"].apply(safe_float)

        plt.figure()
        # simple box-like line per model (mean per camera)
        mean_by = dfl.groupby("model")["AUROC"].mean().sort_values(ascending=False)
        plt.plot(range(len(mean_by)), mean_by.values, marker="o")
        plt.xticks(range(len(mean_by)), mean_by.index, rotation=45, ha="right")
        plt.ylabel("Mean per-camera AUROC")
        plt.title("Detection: Mean Per-Camera AUROC by Model")
        plt.tight_layout()
        if SAVE_PLOTS:
            out = OUTDIR / "det_mean_cam_auroc_by_model.png"
            plt.savefig(out, dpi=200)
            print(f"Saved plot: {out}")
        plt.close()

    # Overall AUROC bar
    if "overall_AUROC" in df.columns:
        plt.figure()
        d2 = df[["model","overall_AUROC"]].dropna().sort_values("overall_AUROC", ascending=False)
        plt.plot(range(len(d2)), d2["overall_AUROC"].values, marker="o")
        plt.xticks(range(len(d2)), d2["model"].tolist(), rotation=45, ha="right")
        plt.ylabel("AUROC")
        plt.title("Detection: Overall AUROC by Model")
        plt.tight_layout()
        if SAVE_PLOTS:
            out = OUTDIR / "det_overall_auroc_by_model.png"
            plt.savefig(out, dpi=200)
            print(f"Saved plot: {out}")
        plt.close()

    # Overall FPR@95 bar
    if "overall_FPR@95" in df.columns:
        plt.figure()
        d3 = df[["model","overall_FPR@95"]].dropna().sort_values("overall_FPR@95", ascending=True)
        plt.plot(range(len(d3)), d3["overall_FPR@95"].values, marker="o")
        plt.xticks(range(len(d3)), d3["model"].tolist(), rotation=45, ha="right")
        plt.ylabel("FPR@95 (lower is better)")
        plt.title("Detection: Overall FPR@95 by Model")
        plt.tight_layout()
        if SAVE_PLOTS:
            out = OUTDIR / "det_overall_fpr95_by_model.png"
            plt.savefig(out, dpi=200)
            print(f"Saved plot: {out}")
        plt.close()

    return df



def context_summary_and_examples(ctx_json: Path):
    if not ctx_json.exists():
        print(f"[WARN] Context JSON not found: {ctx_json}")
        return None

    data = load_json(ctx_json)
    nodes = data.get("node_records", [])
    if not nodes:
        print("[WARN] No node_records found in context JSON.")
        return None

    df = pd.DataFrame(nodes)
    # df columns: sd_token, channel, scene, bbox_2d, label, class, vgae_score
    df["label"] = df["label"].astype(int)
    df["vgae_score"] = df["vgae_score"].astype(float)

    print("\n=== Context Graph Summary (VGAE Score) ===")
    print("Counts:", df["label"].value_counts().to_dict())
    print("Score stats (OOD vs ID):")
    print(df.groupby("label")["vgae_score"].describe().to_string())

    # Per-camera AUROC
    print("\nPer-camera AUROC (VGAE):")
    for cam, sub in df.groupby("channel"):
        y = sub["label"].values
        s = sub["vgae_score"].values
        if len(np.unique(y)) < 2:
            print(f"{cam:16s} AUROC=—  (not enough pos/neg)")
            continue
        auc = float(__import__("sklearn.metrics").metrics.roc_auc_score(y, s))
        print(f"{cam:16s} AUROC={auc:.4f}  n={len(sub)}")

    # Plot score histograms
    plt.figure()
    id_s  = df[df["label"]==0]["vgae_score"].values
    ood_s = df[df["label"]==1]["vgae_score"].values
    plt.hist(id_s,  bins=50, alpha=0.6, label="ID")
    plt.hist(ood_s, bins=50, alpha=0.6, label="OOD")
    plt.legend()
    plt.title("VGAE Node Score Distribution")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    if SAVE_PLOTS:
        out = OUTDIR / "ctx_vgae_score_hist.png"
        plt.savefig(out, dpi=200)
        print(f"Saved plot: {out}")
    plt.close()

    # Example overlays (Top OOD highest score, Top ID lowest score)
    sd_index = load_sample_data_index()

    def save_examples(subdf, tag: str):
        out_dir = OUTDIR / f"examples_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for rank, row in enumerate(subdf.itertuples(index=False), start=1):
            sd = row.sd_token
            img_path = path_from_sample_data(sd, sd_index)
            if img_path is None or not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            b = row.bbox_2d
            text = f"{row.channel} | y={row.label} | score={row.vgae_score:.3f}"
            img = draw_boxes_on_image(img, [b], labels=[row.label], texts=[text], width=4)

            fname = f"{rank:03d}_{row.channel}_{sd[:8]}_y{row.label}_s{row.vgae_score:.3f}.png"
            img.save(out_dir / fname)

    # Top-K OOD (highest score)
    top_ood = df[df["label"]==1].sort_values("vgae_score", ascending=False).head(TOPK)
    # Top-K ID (lowest score)
    top_id  = df[df["label"]==0].sort_values("vgae_score", ascending=True).head(TOPK)

    print(f"\nSaving top-{TOPK} OOD examples and top-{TOPK} ID examples (overlay bbox) …")
    save_examples(top_ood, "ood_top")
    save_examples(top_id,  "id_bottom")

    print(f"Saved example images in: {OUTDIR}")

    return df



def export_6cam_grid_for_sample(sample_token: str, out_path: Path):
    """
    Optional helper:
    Given a sample_token, export a 2x3 grid image of the 6 cameras.
    Requires sample.json + sample_data.json mapping.
    """
    sample_path = NUSCENES_ROOT / JSONDIR_NAME / "sample.json"
    if not sample_path.exists():
        print("[WARN] sample.json missing for 6-cam export.")
        return

    samples = {s["token"]: s for s in load_json(sample_path)}
    sample_data = load_sample_data_index()
    if sample_token not in samples:
        print("[WARN] sample_token not found.")
        return

    s = samples[sample_token]
    sd_tokens = []
    for ch in ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]:
        if ch in s["data"]:
            sd_tokens.append((ch, s["data"][ch]))

    imgs = []
    for ch, sd in sd_tokens:
        p = path_from_sample_data(sd, sample_data)
        if p and p.exists():
            imgs.append((ch, Image.open(p).convert("RGB")))
        else:
            imgs.append((ch, Image.new("RGB",(1600,900))))

    # resize all to same
    W, H = 800, 450
    imgs = [(ch, im.resize((W,H))) for ch, im in imgs]

    grid = Image.new("RGB", (3*W, 2*H))
    order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]
    pos = {order[i]: i for i in range(len(order))}

    for ch, im in imgs:
        i = pos.get(ch, 0)
        r, c = i // 3, i % 3
        grid.paste(im, (c*W, r*H))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print("Saved 6-cam grid:", out_path)


def main():
    print("=== results_viz.py ===")
    print("OUTDIR:", OUTDIR)

    # Detection summary + plots
    if DETECTION_CSV.exists():
        _ = detection_summary_and_plots(DETECTION_CSV)
    else:
        print(f"[WARN] DETECTION_CSV not found: {DETECTION_CSV}")

    # Context graph summary + examples
    if CONTEXT_JSON.exists():
        _ = context_summary_and_examples(CONTEXT_JSON)
    else:
        print(f"[WARN] CONTEXT_JSON not found: {CONTEXT_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
