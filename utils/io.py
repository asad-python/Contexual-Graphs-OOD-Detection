from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2


JsonLike = Union[Dict[str, Any], List[Any]]


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Union[str, Path]) -> JsonLike:
    """Load a JSON file and return the parsed Python object."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """Save a Python object as JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def read_text(path: Union[str, Path]) -> str:
    """Read a text file into a string."""
    p = Path(path)
    return p.read_text(encoding="utf-8")


def write_text(path: Union[str, Path], text: str) -> None:
    """Write a string to a text file."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def list_images(
    folder: Union[str, Path],
    *,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    recursive: bool = True,
) -> List[Path]:
    """Return all image paths in a folder."""
    folder = Path(folder)
    exts = {e.lower() for e in exts}

    if recursive:
        files = folder.rglob("*")
    else:
        files = folder.glob("*")

    out: List[Path] = []
    for p in files:
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def load_image_bgr(path: Union[str, Path]) -> Any:
    """
    Load an image in BGR format (OpenCV default).
    Raises FileNotFoundError / ValueError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {p}")
    return img


def load_image_rgba(path: Union[str, Path]) -> Any:
    """
    Load an image with alpha channel if available (RGBA).
    Useful for patch assets that include transparency.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {p}")
    # Ensure 4 channels
    if len(img.shape) == 2:
        # grayscale -> add channels + alpha
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # BGR -> BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def save_image(path: Union[str, Path], img: Any) -> None:
    """Save an image using OpenCV."""
    p = Path(path)
    ensure_dir(p.parent)
    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise ValueError(f"Failed to write image: {p}")
