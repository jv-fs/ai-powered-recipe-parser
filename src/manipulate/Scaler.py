from __future__ import annotations

import copy
import re
from fractions import Fraction
from typing import Any

_UNICODE_FRACTIONS: dict[str, str] = {
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

_QTY_PATTERN = re.compile(
    r"^\s*(?P<num>(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?))(?P<suffix>.*)$"
)


def resolve_scale_factor(
    servings: dict[str, Any],
    *,
    target_servings: int | None = None,
    scale_factor: float | None = None,
    assumed_base_servings: int | None = None,
) -> tuple[float, int | None, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    original_value = servings.get("value")

    if scale_factor is not None:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")
        new_servings = (
            round(original_value * scale_factor)
            if isinstance(original_value, int) and original_value > 0
            else None
        )
        return float(scale_factor), new_servings, warnings

    if target_servings is None:
        return 1.0, original_value if isinstance(original_value, int) else None, warnings

    if target_servings <= 0:
        raise ValueError("target_servings must be greater than zero.")

    if isinstance(original_value, int) and original_value > 0:
        factor = target_servings / original_value
        return factor, target_servings, warnings

    if isinstance(assumed_base_servings, int) and assumed_base_servings > 0:
        factor = target_servings / assumed_base_servings
        warnings.append(
            {
                "code": "assumed_base_servings",
                "severity": "warning",
                "message": (
                    f"Original servings unknown; scaled from assumed base of "
                    f"{assumed_base_servings} to {target_servings}."
                ),
            }
        )
        return factor, target_servings, warnings

    raise ValueError(
        "Recipe has no stated servings; provide scale_factor or assumed_base_servings."
    )


def parse_quantity(raw: str | None) -> tuple[float | None, str]:
    if raw is None:
        return None, ""

    normalized = raw.strip()
    for char, ascii_frac in _UNICODE_FRACTIONS.items():
        normalized = normalized.replace(char, f" {ascii_frac} ")
    normalized = " ".join(normalized.split())

    match = _QTY_PATTERN.match(normalized)
    if not match:
        return None, raw

    num_str = match.group("num").replace(",", ".")
    suffix = match.group("suffix")

    try:
        if " " in num_str:
            whole, frac = num_str.split(" ", 1)
            value = float(int(whole) + Fraction(frac))
        elif "/" in num_str:
            value = float(Fraction(num_str))
        else:
            value = float(num_str)
    except (ValueError, ZeroDivisionError):
        return None, raw

    return value, suffix


def format_number(value: float) -> str:
    rounded = round(value, 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:g}"


def format_quantity(value: float, suffix: str) -> str:
    formatted_num = format_number(value)
    if not suffix:
        return formatted_num
    if suffix.startswith(" "):
        return f"{formatted_num}{suffix}"
    return f"{formatted_num} {suffix}"


def _replace_once_ws_tolerant(text: str, old: str, new: str) -> tuple[str, bool]:
    if not old:
        return text, False
    if old in text:
        return text.replace(old, new, 1), True

    parts = old.split()
    if not parts:
        return text, False
    pattern = r"\s+".join(re.escape(part) for part in parts)
    updated, count = re.subn(pattern, lambda _: new, text, count=1)
    return updated, count > 0


def scale_recipe(
    recipe: dict[str, Any],
    *,
    target_servings: int | None = None,
    scale_factor: float | None = None,
    assumed_base_servings: int | None = None,
) -> dict[str, Any]:
    scaled = copy.deepcopy(recipe)
    warnings: list[dict[str, Any]] = list(scaled.get("warnings", []))
    servings = scaled.setdefault(
        "servings",
        {
            "id": "serving-001",
            "value": None,
            "inferred": True,
            "inference_reason": None,
            "quote": None,
            "span": None,
        },
    )

    factor, new_servings, factor_warnings = resolve_scale_factor(
        servings,
        target_servings=target_servings,
        scale_factor=scale_factor,
        assumed_base_servings=assumed_base_servings,
    )
    warnings.extend(factor_warnings)

    servings["original_value"] = servings.get("value")
    servings["value"] = new_servings
    servings["scale_factor"] = round(factor, 4)

    if factor == 1.0:
        scaled["warnings"] = warnings
        return scaled

    items_by_step: dict[str, list[tuple[str, str]]] = {}

    for item in scaled.get("items", []):
        if item.get("type") != "ingredient":
            continue

        raw_qty = item.get("quantity")
        item["original_quantity"] = raw_qty

        if raw_qty is None:
            warnings.append(
                {
                    "code": "unquantified_item_unscaled",
                    "severity": "info",
                    "message": f"Ingredient '{item.get('name')}' has no explicit quantity and was not scaled.",
                    "step_id": item.get("step_id"),
                    "item_id": item.get("id"),
                }
            )
            continue

        numeric_qty, suffix = parse_quantity(raw_qty)
        if numeric_qty is None:
            warnings.append(
                {
                    "code": "unparseable_quantity",
                    "severity": "warning",
                    "message": f"Could not parse quantity '{raw_qty}' for '{item.get('name')}'.",
                    "step_id": item.get("step_id"),
                    "item_id": item.get("id"),
                }
            )
            continue

        new_qty = format_quantity(numeric_qty * factor, suffix)
        item["quantity"] = new_qty

        old_quote = item.get("quote") or ""
        if old_quote and raw_qty in old_quote:
            new_quote = old_quote.replace(raw_qty, new_qty, 1)
            item["quote"] = new_quote
            items_by_step.setdefault(item.get("step_id", ""), []).append(
                (old_quote, new_quote)
            )
        elif old_quote:
            items_by_step.setdefault(item.get("step_id", ""), []).append(
                (raw_qty, new_qty)
            )

    for step in scaled.get("steps", []):
        step_id = step.get("id", "")
        replacements = items_by_step.get(step_id, [])
        if not replacements:
            continue

        step_text = step.get("text", "")
        for old_sub, new_sub in replacements:
            step_text, replaced = _replace_once_ws_tolerant(step_text, old_sub, new_sub)
            if not replaced:
                warnings.append(
                    {
                        "code": "step_text_not_updated",
                        "severity": "warning",
                        "message": f"Could not update '{old_sub}' to '{new_sub}' in step text.",
                        "step_id": step_id,
                    }
                )
        step["text"] = step_text

    scaled["warnings"] = warnings
    return scaled