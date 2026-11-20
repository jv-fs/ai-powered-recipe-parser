from __future__ import annotations

import base64
import json
import re
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import cv2
import numpy as np
from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR

from ..ingest import IngestResult, Page
from .llm import _b64img, _call_llm, _native_completion, _transcribe, _transcribe_crop
from .models import (
    Evidence,
    ExtractResult,
    ExtractWarning,
    Field,
    ItemMention,
    OcrLine,
    Reading,
    Servings,
    Source,
    Step,
)
from .ocr import _CROP_PADDING, _crop, _run_ocr
from .pipeline import extract
from .prompts import (
    _CROP_PROMPT,
    _CROP_SCHEMA,
    _STRUCTURE_PROMPT,
    _STRUCTURE_SCHEMA,
    _TRANSCRIPT_PROMPT,
    _TRANSCRIPT_SCHEMA,
)
from .structure import _structure
from .text import (
    _DURATION,
    _FRAC_MAP,
    _FRACTIONS,
    _GERMAN_PREFIXES,
    _NULL_STRINGS,
    _PLAUSIBLE_UNITS,
    _QUANTITY_VALID,
    _TOKEN,
    _UNITS,
    _WS,
    _as_str,
    _canon,
    _canonical_quantity,
    _canonical_unit,
    _find_span,
    _json_loads,
    _merge_bboxes,
    _norm,
    _same,
    _tokens,
)
from .verification import (
    _match_ocr_to_segment,
    _process_page,
    _recheck_crop,
    _refine_item_bbox,
)

__all__ = [
    "Any",
    "Evidence",
    "ExtractResult",
    "ExtractWarning",
    "Field",
    "Fraction",
    "IngestResult",
    "ItemMention",
    "Literal",
    "OcrLine",
    "OpenAI",
    "Page",
    "RapidOCR",
    "Reading",
    "Servings",
    "Source",
    "Step",
    "base64",
    "cv2",
    "dataclass",
    "extract",
    "json",
    "np",
    "re",
    "urllib",
]
