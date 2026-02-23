

from __future__ import annotations

from pathlib import Path
import random, uuid, json, shutil
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points


try:
    from config.defaults import SRC_DATASET, DST_DATASET, ASSETS_DIR
except Exception:
    # If you don't want config import, set these manually:
    SRC_DATASET = "WRITE_SOURCE_NUSCENES_DATASET_PATH_HERE"
    DST_DATASET = "WRITE_DESTINATION_DATASET_PATH_HERE"
    ASSETS_DIR  = "WRITE_ASSETS_DIRECTORY_PATH_HERE"

ALL_CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

N_PASTES = 50

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


augment = A.Compose(
    [
        A.RandomScale(scale_limit=(0.3, 0.6), p=1.0),
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(p=0.45),
        A.HorizontalFlip(p=0.25),
    ],
    additional_targets={"mask": "mask"},
)


def paste_object(img_bgr: np.ndarray, obj_rgba: np.ndarray):
    """
    SAME paste_object logic as your attached script:
    - ensure RGBA
    - apply augment(image=obj_rgb, mask=alpha)
    - resize if too large
    - paste into LOWER HALF
    - alpha blending
    """
    h, w = img_bgr.shape[:2]

    # ensure 4-channel RGBA
    if obj_rgba.ndim == 2:
        obj_rgba = np.dstack(
            [obj_rgba, obj_rgba, obj_rgba, 255 * np.ones_like(obj_rgba)]
        )
    if obj_rgba.shape[2] == 3:
        alpha = 255 * np.ones(obj_rgba.shape[:2], obj_rgba.dtype)
        obj_rgba = np.dstack([obj_rgba, alpha])
    if obj_rgba.shape[2] != 4:
        return img_bgr, None

    # Split color + alpha
    obj_rgb = obj_rgba[:, :, :3]  # RGB from PIL-style assets
    alpha = obj_rgba[:, :, 3]

    aug = augment(image=obj_rgb, mask=alpha)
    obj_rgb_aug, alpha_mask = aug["image"], aug["mask"]

    obj_bgr = cv2.cvtColor(obj_rgb_aug, cv2.COLOR_RGB2BGR)
    oh, ow = obj_bgr.shape[:2]

    # keep inside lower half, resize if needed
    max_w, max_h = w - 10, int(h * 0.5) - 10
    if ow >= max_w or oh >= max_h:
        scale = 0.9 * min(max_w / max(ow, 1), max_h / max(oh, 1))
        if scale <= 0:
            return img_bgr, None
        obj_bgr = cv2.resize(obj_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        alpha_mask = cv2.resize(alpha_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        oh, ow = obj_bgr.shape[:2]

    # place somewhere in lower half
    x0 = random.randint(0, max(0, w - ow))
    y0 = random.randint(int(h * 0.5), max(int(h * 0.5), h - oh))

    roi = img_bgr[y0 : y0 + oh, x0 : x0 + ow]
    alpha_f = (alpha_mask[:, :, None] / 255.0).astype(np.float32)

    img_bgr[y0 : y0 + oh, x0 : x0 + ow] = (1 - alpha_f) * roi + alpha_f * obj_bgr

    return img_bgr, (x0, y0, x0 + ow, y0 + oh)


def _load_rgba_asset(path: Path) -> np.ndarray:
    """
    Loads asset as RGBA (matching your original behavior).
    Expect PNG with alpha.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load asset: {path}")
    # OpenCV loads as BGRA usually; convert to RGBA-like layout for your paste code expecting RGB
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        return cv2.merge([r, g, b, a])  # RGBA
    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = 255 * np.ones_like(b)
        return cv2.merge([r, g, b, a])
    if img.ndim == 2:
        a = 255 * np.ones_like(img)
        return np.dstack([img, img, img, a]).astype(np.uint8)
    raise ValueError(f"Unexpected asset shape: {img.shape} for {path}")


def build_scene_to_frames(nusc: NuScenes):
    """
    Builds:
    scene_to_frames[scene_token][sample_token][channel] = (img_abs_path, sample_data_token)
    This matches your original intent.
    """
    # Map sample_data token -> record (we need filename + sample_token + channel)
    sd_by_token = {sd["token"]: sd for sd in nusc.sample_data}

    # scene_token -> sample_token -> ch -> (abs_path, sd_token)
    scene_to_frames = {}
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_to_frames[scene_token] = {}
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            scene_to_frames[scene_token][sample_token] = {}
            for ch in ALL_CAM_CHANNELS:
                if ch not in sample["data"]:
                    continue
                sd_token = sample["data"][ch]
                sd = sd_by_token[sd_token]
                img_abs = str((Path(nusc.dataroot) / sd["filename"]).resolve())
                scene_to_frames[scene_token][sample_token][ch] = (img_abs, sd_token)
            sample_token = sample["next"]
    return scene_to_frames


def inject_ood_and_save_jsons(
    src_root: str,
    dst_root: str,
    assets_dir: str,
    *,
    n_pastes: int = N_PASTES,
    version: str = "v1.0-mini",
):
    SRC = Path(src_root)
    DST = Path(dst_root)
    ASSETS = Path(assets_dir)

    assert SRC.exists(), f"SRC not found: {SRC}"
    assert ASSETS.exists() and any(ASSETS.glob("*.png")), "ASSETS folder empty"
    if DST.exists():
        raise FileExistsError(f"{DST} already exists — remove it or choose a new path")

    # 1) Clone dataset
    print("Copying nuScenes tree …")
    shutil.copytree(SRC, DST)
    print("Copied to:", DST)

    # 2) Load assets
    asset_paths = sorted(ASSETS.glob("*.png"))
    print(f"{len(asset_paths)} assets found in {ASSETS}")

    # 3) Index scenes/frames
    print("Indexing scenes across 6 cameras …")
    nusc_dst = NuScenes(version=version, dataroot=str(DST), verbose=False)
    scene_to_frames = build_scene_to_frames(nusc_dst)

    # 4) Randomly paste objects and record bboxes
    new_boxes = []  # list of dict entries for detection_novel.json
    scene_tokens = list(scene_to_frames.keys())
    if not scene_tokens:
        raise RuntimeError("No scenes found in DST dataset.")

    print(f"Pasting {n_pastes} OOD objects …")
    for _ in tqdm(range(n_pastes), desc="OOD pastes"):
        scene_tok = random.choice(scene_tokens)
        sample_tok = random.choice(list(scene_to_frames[scene_tok].keys()))
        ch2info = scene_to_frames[scene_tok][sample_tok]

        # choose random camera among available
        ch = random.choice([c for c in ALL_CAM_CHANNELS if c in ch2info])
        img_path, sd_token = ch2info[ch]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        asset_path = random.choice(asset_paths)
        obj_rgba = _load_rgba_asset(asset_path)

        img_out, bbox = paste_object(img_bgr, obj_rgba)
        if bbox is None:
            continue

        # save modified image back
        cv2.imwrite(img_path, img_out)

        x0, y0, x1, y1 = bbox
        new_boxes.append(
            {
                "sample_data_token": sd_token,
                "bbox_2d": [float(x0), float(y0), float(x1), float(y1)],
                "token": str(uuid.uuid4()),
                "detection_name": "novel",
                "detection_score": 1.0,
                "attribute_name": "",
            }
        )

    # 5) Write detection_novel.json
    det_src = DST / version / "detection.json"
    if det_src.exists():
        det_data = json.loads(det_src.read_text())
    else:
        det_data = {"results": {}, "meta": {"version": version}}

    for rec in new_boxes:
        det_data["results"].setdefault(rec["sample_data_token"], []).append(rec)

    out_novel = DST / version / "detection_novel.json"
    out_novel.parent.mkdir(parents=True, exist_ok=True)
    out_novel.write_text(json.dumps(det_data))
    print("✔ OOD Detection JSON →", out_novel)

    # 6) Project GT 3D boxes -> 2D and write detection_id.json
    print("Projecting GT 3D boxes to 2D for all 6 cameras …")
    nusc_src = NuScenes(version=version, dataroot=str(SRC), verbose=False)
    id_results = {}

    for sd in tqdm(nusc_src.sample_data, desc="ID boxes"):
        if sd["channel"] not in ALL_CAM_CHANNELS:
            continue

        sd_token = sd["token"]
        cs = nusc_src.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = nusc_src.get("ego_pose", sd["ego_pose_token"])
        intr = np.array(cs["camera_intrinsic"], dtype=np.float32)

        # image size
        img_path = SRC / sd["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # get 3D boxes in camera frame
        _, boxes, _ = nusc_src.get_sample_data(sd_token, box_vis_level=None)

        entries = []
        for box in boxes:
            # Transform from global -> ego -> sensor is already handled by get_sample_data
            # Now project corners to 2D using camera intrinsic
            corners_3d = box.corners()  # shape (3,8)
            corners_2d = view_points(corners_3d, intr, normalize=True)  # (3,8)
            xs, ys = corners_2d[0, :], corners_2d[1, :]

            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

            # clip & validate
            x0 = max(0.0, min(x0, W - 1.0))
            x1 = max(0.0, min(x1, W - 1.0))
            y0 = max(0.0, min(y0, H - 1.0))
            y1 = max(0.0, min(y1, H - 1.0))
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                continue

            entries.append(
                {
                    "translation": list(map(float, box.center)),
                    "size": list(map(float, box.wlh)),
                    "rotation": list(map(float, box.orientation.elements)),
                    "velocity": [float(box.velocity[0]), float(box.velocity[1]), 0.0]
                    if box.velocity is not None
                    else None,
                    "detection_name": box.name,
                    "detection_score": 1.0,
                    "attribute_name": "",
                    "bbox_2d": [x0, y0, x1, y1],
                    "token": str(uuid.uuid4()),
                }
            )

        if entries:
            id_results[sd_token] = entries

    out_id = DST / version / "detection_id.json"
    out_id.write_text(json.dumps({"results": id_results, "meta": {"version": version}}, indent=2))
    print("✔ ID Detection JSON →", out_id)

    print("\nClone ready at:", DST)
    print(" - OOD boxes:", out_novel)
    print(" - ID boxes :", out_id)


def main():
    inject_ood_and_save_jsons(
        src_root=SRC_DATASET,
        dst_root=DST_DATASET,
        assets_dir=ASSETS_DIR,
        n_pastes=N_PASTES,
        version="v1.0-mini",
    )


if __name__ == "__main__":
    main()
