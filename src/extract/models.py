from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Source = Literal[
    "text_layer",
    "ocr",
    "llm",
    "crop_ocr",
    "crop_llm",
    "accepted_evidence",
    "llm_structured",
]
Field = Literal[
    "transcript",
    "name",
    "quantity",
    "unit",
    "preparation",
    "step",
    "quote",
    "item",
    "servings",
]


@dataclass(frozen=True)
class Reading:
    source: Source
    value: str | None
    score: float | None = None


@dataclass(frozen=True)
class ExtractWarning:
    code: str
    message: str
    page: int | None = None
    source_index: int | None = None
    step_id: str | None = None
    item_id: str | None = None
    field: str | None = None
    quote: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    char_start: int | None = None
    char_end: int | None = None
    readings: tuple[Reading, ...] = ()
    selected: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Step:
    id: str
    index: int
    text: str
    page_index: int
    item_ids: tuple[str, ...] = ()
    duration: str | None = None
    conditions: tuple[str, ...] = ()
    continues_previous: bool = False


@dataclass(frozen=True)
class ItemMention:
    id: str
    step_id: str
    name: str
    type: str
    quantity: str | None
    unit: str | None
    preparation: str | None
    optional: bool
    inferred: bool
    inference_reason: str | None
    quote: str
    grounding: str
    page_index: int
    source_index: int
    bbox: tuple[int, int, int, int] | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class Servings:
    value: int | None
    inferred: bool = True
    quote: str | None = None


@dataclass(frozen=True)
class ExtractResult:
    steps: tuple[Step, ...]
    items: tuple[ItemMention, ...]
    servings: Servings
    warnings: tuple[ExtractWarning, ...]
    model: str


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: tuple[int, int, int, int] | None
    score: float | None


@dataclass(frozen=True)
class Evidence:
    id: str
    page: int
    source_index: int
    bbox: tuple[int, int, int, int] | None
    char_start: int | None
    char_end: int | None
    text: str
    is_recipe: bool
    grounding: str

