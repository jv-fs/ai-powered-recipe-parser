from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .Renderer import render_markdown


def _short(value: str, limit: int = 120) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _format_warning(warning: dict[str, Any]) -> str:
    code = warning.get("code", "unknown")
    fields: list[str] = []
    for name in ("severity", "page", "step_id", "item_id"):
        value = warning.get(name)
        if value is not None:
            fields.append(f"{name}={value}")
    quote = warning.get("quote")
    if quote is not None:
        fields.append(f"quote={_short(str(quote))!r}")
    span = warning.get("span")
    if isinstance(span, dict) and "start" in span and "end" in span:
        fields.append(f"span=({span['start']}, {span['end']})")

    header = " ".join([f"[{code}]", *fields]) if fields else f"[{code}]"
    lines = [header]
    if warning.get("message") is not None:
        lines.append(f"  message: {warning['message']}")
    if warning.get("reason") is not None:
        lines.append(f"  reason: {warning['reason']}")
    if warning.get("full_reading") is not None:
        lines.append(f"  full_reading: {warning['full_reading']!r}")
    if warning.get("crop_reading") is not None:
        lines.append(f"  crop_reading: {warning['crop_reading']!r}")
    return "\n".join(lines)


def export_recipe(
    recipe: dict[str, Any],
    *,
    source_label: str,
    page_count: int,
    ingest_warnings: list[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    warnings_path: Path,
    title: str | None = None,
    make_vegan: bool = False,
) -> dict[str, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_content = render_markdown(
        recipe,
        title=title,
        make_vegan=make_vegan,
    )
    markdown_path.write_text(md_content, encoding="utf-8")

    pipeline_warnings = recipe.get("warnings", [])
    report_lines = [
        f"source: {source_label}",
        f"pages: {page_count}",
        f"model: {recipe.get('model', 'unknown')}",
        "",
        f"[ingest] {len(ingest_warnings)} warning(s)",
    ]
    if ingest_warnings:
        for w in ingest_warnings:
            report_lines.append(
                f"  code={w.get('code')} page={w.get('page')} message={w.get('message')}"
            )
    else:
        report_lines.append("  (none)")

    report_lines.extend(
        [
            "",
            f"[extract + manipulate] {len(pipeline_warnings)} warning(s)",
            *(_format_warning(w) for w in pipeline_warnings),
        ]
    )
    if not pipeline_warnings:
        report_lines.append("  (none)")

    warnings_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "json": json_path,
        "markdown": markdown_path,
        "warnings": warnings_path,
    }