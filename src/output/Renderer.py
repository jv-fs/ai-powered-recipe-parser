from __future__ import annotations

from typing import Any


def _derive_title(recipe: dict[str, Any], fallback: str = "Extracted Recipe") -> str:
    text = (recipe.get("text") or "").strip()
    if not text:
        return fallback
    first_line = text.splitlines()[0].strip()
    if len(first_line) <= 90:
        return first_line
    return first_line[:87].rstrip() + "..."


def _format_item_line(item: dict[str, Any]) -> str:
    parts: list[str] = []
    qty = item.get("quantity")
    unit = item.get("unit")
    name = (item.get("name") or "").strip()
    prep = (item.get("preparation") or "").strip()

    if qty:
        qty_str = str(qty).strip()
        if unit and not qty_str.lower().endswith(str(unit).strip().lower()):
            parts.append(f"{qty_str} {str(unit).strip()}")
        else:
            parts.append(qty_str)

    parts.append(f"**{name}**" if name else "**(unnamed)**")

    details: list[str] = []
    if prep:
        details.append(prep)
    if item.get("optional"):
        details.append("optional")

    sub = item.get("vegan_substitution")
    if isinstance(sub, dict):
        orig_name = sub.get("original_name")
        orig_qty = sub.get("original_quantity")
        if orig_name:
            replaced_label = f"{orig_qty} {orig_name}".strip() if orig_qty else orig_name
            details.append(f"vegan substitute for: {replaced_label}")

    line = f"- {' '.join(parts)}"
    if details:
        line += f" *({'; '.join(details)})*"
    return line


def _format_step_line(index: int, step: dict[str, Any]) -> str:
    text = (step.get("text") or "").strip()
    meta: list[str] = []
    if step.get("duration"):
        meta.append(f"duration: {step['duration']}")
    conditions = step.get("conditions")
    if isinstance(conditions, list) and conditions:
        meta.append(f"conditions: {', '.join(str(c) for c in conditions)}")
    if step.get("inferred"):
        meta.append("inferred step")

    line = f"{index}. {text}"
    if meta:
        line += f"\n   - *({'; '.join(meta)})*"
    return line


def render_markdown(
    recipe: dict[str, Any],
    *,
    title: str | None = None,
    make_vegan: bool = False,
) -> str:
    recipe_title = title or _derive_title(recipe)
    servings = recipe.get("servings") or {}
    items = recipe.get("items") or []
    steps = recipe.get("steps") or []
    warnings = recipe.get("warnings") or []

    ingredients = [i for i in items if i.get("type") == "ingredient"]
    tools = [i for i in items if i.get("type") == "tool"]

    seen_tools: set[str] = set()
    unique_tools: list[dict[str, Any]] = []
    for tool in tools:
        key = (tool.get("name") or "").strip().casefold()
        if key and key not in seen_tools:
            seen_tools.add(key)
            unique_tools.append(tool)

    lines: list[str] = [f"# {recipe_title}", ""]

    # Metadata block
    meta_lines: list[str] = []
    serving_val = servings.get("value")
    orig_serving = servings.get("original_value")
    scale_factor = servings.get("scale_factor")

    if serving_val is not None:
        s_text = f"**Servings:** {serving_val}"
        if orig_serving is not None and orig_serving != serving_val:
            s_text += f" (scaled from {orig_serving})"
        meta_lines.append(f"- {s_text}")
    elif scale_factor and scale_factor != 1.0:
        meta_lines.append(f"- **Servings:** Not specified (scaled by {scale_factor}x)")
    else:
        meta_lines.append("- **Servings:** Not specified")

    if scale_factor and scale_factor != 1.0:
        meta_lines.append(f"- **Scale Factor:** {scale_factor}x")

    meta_lines.append(f"- **Dietary Mode:** {'Veganized' if make_vegan else 'Original'}")

    lines.extend(meta_lines)
    lines.append("")

    # Ingredients section
    lines.append("## Ingredients")
    lines.append("")
    if ingredients:
        for item in ingredients:
            lines.append(_format_item_line(item))
    else:
        lines.append("- *(No ingredients extracted)*")
    lines.append("")

    # Tools section
    lines.append("## Tools & Equipment")
    lines.append("")
    if unique_tools:
        for tool in unique_tools:
            t_name = (tool.get("name") or "").strip()
            tag = " *(inferred)*" if tool.get("inferred") else ""
            lines.append(f"- {t_name}{tag}")
    else:
        lines.append("- *(No specific tools listed)*")
    lines.append("")

    # Preparation steps
    lines.append("## Preparation")
    lines.append("")
    if steps:
        for idx, step in enumerate(steps, start=1):
            lines.append(_format_step_line(idx, step))
    else:
        lines.append("1. *(No preparation steps extracted)*")
    lines.append("")

    # User-relevant notes & warnings
    notable_warnings = [
        w
        for w in warnings
        if isinstance(w, dict)
        and w.get("code") not in {"crop_refined", "resized"}
    ]
    if notable_warnings:
        lines.append("## Notes & Audit Warnings")
        lines.append("")
        for w in notable_warnings:
            code = w.get("code", "info")
            msg = w.get("message", "")
            lines.append(f"- **[{code}]** {msg}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"