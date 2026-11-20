from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Any

from ..ingest import Page


_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w/.,½¼¾⅓⅔⅛⅜⅝⅞]+", re.UNICODE)
_QUANTITY_VALID = re.compile(
    r"^\s*(?:\d+\s+)?(?:\d+/\d+|\d+[.,]\d+|[½¼¾⅓⅔⅛⅜⅝⅞]|\d+)\s*$"
)
_DURATION = re.compile(
    r"(?:^|\b)(?:\d+(?:[.,]\d+)?\.?|ein|eine|einer|einem|einen|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)\s*"
    r"(?:min|minuten|minute|h|std|stunden|stunde|sek|sekunden|tag|tage|tagen|t)\b",
    re.IGNORECASE,
)
_FRAC_MAP = str.maketrans(
    {
        "½": "1/2",
        "¼": "1/4",
        "¾": "3/4",
        "⅓": "1/3",
        "⅔": "2/3",
        "⅛": "1/8",
        "⅜": "3/8",
        "⅝": "5/8",
        "⅞": "7/8",
    }
)
_FRACTIONS = {
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}
_UNITS = {
    "l": "l",
    "liter": "l",
    "litre": "l",
    "liters": "l",
    "litres": "l",
    "ml": "ml",
    "milliliter": "ml",
    "millilitre": "ml",
    "cl": "cl",
    "zentiliter": "cl",
    "g": "g",
    "gramm": "g",
    "gram": "g",
    "gramme": "g",
    "kg": "kg",
    "kilogramm": "kg",
    "kilogram": "kg",
    "mg": "mg",
    "milligramm": "mg",
    "el": "el",
    "esslöffel": "el",
    "essloeffel": "el",
    "tl": "tl",
    "teelöffel": "tl",
    "teeloeffel": "tl",
    "stk": "stk",
    "stück": "stk",
    "stueck": "stk",
    "tasse": "tasse",
    "tassen": "tasse",
    "prise": "prise",
    "prisen": "prise",
    "pfund": "pfund",
    "pfd": "pfund",
    "cm": "cm",
    "mm": "mm",
}
_PLAUSIBLE_UNITS = frozenset(_UNITS.values())
_NULL_STRINGS = frozenset({"null", "none", "n/a", ""})
_GERMAN_PREFIXES = frozenset(
    {
        "ab",
        "an",
        "auf",
        "aus",
        "bei",
        "ein",
        "fort",
        "her",
        "hin",
        "los",
        "mit",
        "nach",
        "vor",
        "weg",
        "zu",
        "zurück",
        "zusammen",
    }
)

def _norm(text: str | None) -> str:
    return _WS.sub(" ", (text or "")).strip()


def _as_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _norm(value)
    return None if cleaned.casefold() in _NULL_STRINGS else cleaned


def _canon(text: str | None) -> str:
    t = _norm(text).casefold().translate(_FRAC_MAP).replace(",", ".")
    return _WS.sub(" ", t).strip()


def _tokens(text: str | None) -> set[str]:
    return set(_TOKEN.findall(_canon(text)))


def _same(a: str | None, b: str | None) -> bool:
    ka, kb = _canon(a), _canon(b)
    return ka != "" and ka == kb


def _canonical_unit(value: str | None) -> str | None:
    if not value:
        return None
    tokens = _TOKEN.findall(value.casefold())
    return _UNITS.get(tokens[0], tokens[0]) if tokens else None


def _canonical_quantity(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().casefold().replace(",", ".")
    for character, fraction in _FRACTIONS.items():
        text = text.replace(character, f" {fraction} ")
    total = Fraction(0)
    matched = False
    for token in _TOKEN.findall(text):
        try:
            parsed = Fraction(token)
        except (ValueError, ZeroDivisionError):
            continue
        if parsed > 0:
            total += parsed
            matched = True
    return str(total) if matched else None


def _json_loads(content: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(content) if isinstance(content, str) else None
    except (TypeError, json.JSONDecodeError):
        data = None
    return data if isinstance(data, dict) else None


def _find_span(page: Page, text: str) -> tuple[int | None, int | None]:
    needle = _norm(text)
    if not needle:
        return None, None
    for layer in (page.raw_text, page.text):
        if not layer:
            continue
        idx = layer.find(needle)
        if idx >= 0 and page.char_start is not None:
            return page.char_start + idx, page.char_start + idx + len(needle)
        folded_layer = _norm(layer).casefold()
        if needle.casefold() in folded_layer:
            return page.char_start, page.char_end
    return None, None


def _merge_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )
