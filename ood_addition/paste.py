

import random
import albumentations as A
import cv2


augment = A.Compose(
    [
        A.RandomScale(scale_limit=(0.3, 0.6), p=1.0),
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(p=0.45),
        A.HorizontalFlip(p=0.25),
    ],
    additional_targets={"mask": "mask"},
)


def paste_object(img_bgr, obj_rgba, lower_half_only=True):
    """
    Paste a single RGBA object patch into a BGR camera image.

    Args:
        img_bgr: nuScenes camera image (BGR)
        obj_rgba: patch image with alpha channel (BGRA/RGBA-like from cv2.IMREAD_UNCHANGED)
        lower_half_only: default True -> paste only in lower half (patch default behavior)

    Returns:
        (new_img_bgr, bbox) where bbox = (x1, y1, x2, y2)
    """

    h, w = img_bgr.shape[:2]

    # obj_rgba is usually BGRA if loaded with cv2.IMREAD_UNCHANGED
    obj_bgr = obj_rgba[:, :, :3]
    alpha = obj_rgba[:, :, 3]

    # Albumentations expects image + mask
    aug = augment(image=obj_bgr, mask=alpha)
    obj_bgr, alpha = aug["image"], aug["mask"]

    oh, ow = obj_bgr.shape[:2]
    if oh <= 0 or ow <= 0:
        return img_bgr, None

    # placement region
    x0 = random.randint(0, max(0, w - ow))

    if lower_half_only:
        y_min = int(h * 0.5)
    else:
        y_min = 0

    y0 = random.randint(y_min, max(y_min, h - oh))

    # Paste using alpha blending
    roi = img_bgr[y0:y0 + oh, x0:x0 + ow]
    alpha_f = (alpha[:, :, None] / 255.0)

    img_bgr[y0:y0 + oh, x0:x0 + ow] = (1 - alpha_f) * roi + alpha_f * obj_bgr

    bbox = (x0, y0, x0 + ow, y0 + oh)
    return img_bgr, bbox
