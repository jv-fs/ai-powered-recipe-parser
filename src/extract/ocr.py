from __future__ import annotations

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .models import OcrLine
from .text import _norm


_CROP_PADDING = 8

def _run_ocr(image: bytes | None, ocr: RapidOCR) -> tuple[list[OcrLine] | None, Exception | None]:
    if not image:
        return [], None
    try:
        arr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return [], None
        out, _ = ocr(arr)
        lines: list[OcrLine] = []
        for entry in out or []:
            try:
                box, txt, score = entry[0], str(entry[1]), float(entry[2])
                xs = [int(round(p[0])) for p in box]
                ys = [int(round(p[1])) for p in box]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            except Exception:
                txt, bbox, score = str(entry[1]) if len(entry) > 1 else "", None, None
            t = _norm(txt)
            if t:
                lines.append(OcrLine(t, bbox, score))
        return lines, None
    except Exception as e:
        return None, e


def _crop(image: bytes, bbox: tuple[int, int, int, int] | None) -> bytes | None:
    if not bbox:
        return None
    arr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1 - _CROP_PADDING), max(0, y1 - _CROP_PADDING)
    x2, y2 = min(w, x2 + _CROP_PADDING), min(h, y2 + _CROP_PADDING)
    if x2 <= x1 or y2 <= y1:
        return None
    ok, buf = cv2.imencode(".png", arr[y1:y2, x1:x2])
    return buf.tobytes() if ok else None
