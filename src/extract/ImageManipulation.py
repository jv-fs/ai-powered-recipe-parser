from __future__ import annotations

import cv2
import numpy as np


def split_vertical_two_thirds(image_bytes: bytes) -> tuple[bytes, bytes]:
    if not image_bytes:
        raise ValueError("Empty image bytes provided.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")

    height = image.shape[0]
    if height < 3:
        raise ValueError("Image height is too small to split into two-thirds.")

    split_end_top = int(round(height * (2 / 3)))
    split_start_bottom = int(round(height * (1 / 3)))

    top_crop = image[:split_end_top, :]
    bottom_crop = image[split_start_bottom:, :]

    ok_top, top_encoded = cv2.imencode(".png", top_crop)
    ok_bottom, bottom_encoded = cv2.imencode(".png", bottom_crop)

    if not ok_top or not ok_bottom:
        raise RuntimeError("Failed to encode cropped images to PNG.")

    return top_encoded.tobytes(), bottom_encoded.tobytes()