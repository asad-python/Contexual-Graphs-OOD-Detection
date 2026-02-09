"""

Requirements:
- nuscenes-devkit
- config/defaults.py should define SRC_DATASET/DST_DATASET as strings (paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nuscenes.nuscenes import NuScenes


# nuScenes has 6 camera streams in the standard setup
DEFAULT_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class CameraFrame:
    """One camera frame for a single sample."""
    sample_token: str
    cam_name: str
    sample_data_token: str
    filename: str          # relative path inside nuScenes (as stored in tables)
    abs_path: str          # resolved absolute path on disk
    timestamp: int


def _resolve_abs_path(dataset_root: Path, rel_filename: str) -> str:
    """Resolve nuScenes relative filename to an absolute path string."""
    return str((dataset_root / rel_filename).resolve())


def build_scene_sample_index(
    dataset_root: str,
    version: str,
    *,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    scene_names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build an index of scenes -> samples -> camera frames.

    Returns a dictionary with structure:

    index = {
      "meta": {...},
      "scenes": [
         {
           "scene_token": ...,
           "name": ...,
           "description": ...,
           "nbr_samples": ...,
           "samples": [
              {
                "sample_token": ...,
                "timestamp": ...,
                "cams": {
                   "CAM_FRONT": {
                      "sample_data_token": ...,
                      "filename": ...,
                      "abs_path": ...,
                      "timestamp": ...
                   },
                   ...
                }
              },
              ...
           ]
         },
         ...
      ]
    }
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"nuScenes dataset_root does not exist: {root}\n"
            "Set dataset_root in config/defaults.py (placeholder path)."
        )

    nusc = NuScenes(version=version, dataroot=str(root), verbose=verbose)

    # Select scenes
    all_scenes = nusc.scene
    if scene_names is not None:
        scene_names_set = set(scene_names)
        scenes = [s for s in all_scenes if s.get("name") in scene_names_set]
    else:
        scenes = list(all_scenes)

    out: Dict[str, Any] = {
        "meta": {
            "version": version,
            "dataset_root": str(root.resolve()),
            "cameras": list(cameras),
            "num_scenes": len(scenes),
        },
        "scenes": [],
    }

    for scene in scenes:
        scene_token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        nbr_samples = scene["nbr_samples"]

        scene_entry: Dict[str, Any] = {
            "scene_token": scene_token,
            "name": scene.get("name", ""),
            "description": scene.get("description", ""),
            "nbr_samples": nbr_samples,
            "samples": [],
        }

        # Walk the linked list of samples
        cur_token = first_sample_token
        while cur_token:
            sample = nusc.get("sample", cur_token)

            sample_entry: Dict[str, Any] = {
                "sample_token": cur_token,
                "timestamp": sample.get("timestamp", -1),
                "cams": {},
            }

            data = sample.get("data", {})
            for cam in cameras:
                if cam not in data:
                    # Some subsets might not have all cameras
                    continue

                sd_token = data[cam]
                sd = nusc.get("sample_data", sd_token)

                rel_fn = sd["filename"]
                sample_entry["cams"][cam] = {
                    "sample_data_token": sd_token,
                    "filename": rel_fn,
                    "abs_path": _resolve_abs_path(root, rel_fn),
                    "timestamp": sd.get("timestamp", -1),
                }

            scene_entry["samples"].append(sample_entry)

            # next sample
            cur_token = sample.get("next", "")

        out["scenes"].append(scene_entry)

    return out


def flatten_camera_frames(
    index: Dict[str, Any],
    *,
    cameras: Optional[Sequence[str]] = None,
) -> List[CameraFrame]:
    """
    Convert the hierarchical index into a flat list of CameraFrame objects.

    Useful for:
    - iterating all camera frames
    - sampling frames for injection
    - building per-camera video outputs
    """
    cams_filter = set(cameras) if cameras is not None else None

    frames: List[CameraFrame] = []
    for scene in index.get("scenes", []):
        for s in scene.get("samples", []):
            sample_token = s["sample_token"]
            for cam_name, cam_info in s.get("cams", {}).items():
                if cams_filter is not None and cam_name not in cams_filter:
                    continue

                frames.append(
                    CameraFrame(
                        sample_token=sample_token,
                        cam_name=cam_name,
                        sample_data_token=cam_info["sample_data_token"],
                        filename=cam_info["filename"],
                        abs_path=cam_info["abs_path"],
                        timestamp=int(cam_info.get("timestamp", -1)),
                    )
                )
    return frames
