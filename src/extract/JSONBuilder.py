from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from ..config import LLM_BASE_URL
from .LLMMethods import extract_structure

_WS = re.compile(r"\s+")
_WORD_CHARS = re.compile(r"\w{2,}", flags=re.UNICODE)
_FUZZY_MATCH_THRESHOLD = 0.90
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


def _norm(text: str | None) -> str:
    return _WS.sub(" ", (text or "")).strip()


def _canonical(text: str | None) -> str:
    cleaned = _norm(text).casefold().translate(_FRAC_MAP).replace(",", ".")
    return _WS.sub(" ", cleaned).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = _norm(value)
        if not cleaned or cleaned.lower() == "null":
            return None
        return cleaned
    return _norm(str(value)) or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _find_span(haystack: str, needle: str | None) -> tuple[int, int] | None:
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx >= 0:
        return (idx, idx + len(needle))

    parts = _norm(needle).split()
    if parts:
        pattern = r"\s+".join(re.escape(part) for part in parts)
        match = re.search(pattern, haystack)
        if match:
            return (match.start(), match.end())

    c_needle = _canonical(needle)
    if not c_needle:
        return None
    n_tokens = c_needle.split()
    h_spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S+", haystack):
        ct = _canonical(m.group())
        if ct:
            for tok in ct.split():
                h_spans.append((m.start(), m.end(), tok))

    h_tokens = [tok for _, _, tok in h_spans]
    size = len(n_tokens)
    if size and len(h_tokens) >= size:
        best_score = 0.0
        best_span: tuple[int, int] | None = None

        for i in range(len(h_tokens) - size + 1):
            window_tokens = h_tokens[i : i + size]
            if window_tokens == n_tokens:
                return (h_spans[i][0], h_spans[i + size - 1][1])

            score = _similarity(c_needle, " ".join(window_tokens))
            if score > best_score:
                best_score = score
                best_span = (h_spans[i][0], h_spans[i + size - 1][1])

        if best_span is not None and best_score >= _FUZZY_MATCH_THRESHOLD:
            return best_span

    return None


def _page_for_offset(pages: list[dict[str, Any]], start: int | None) -> int:
    if not pages or start is None:
        return 0
    for page in pages:
        p_start = int(page.get("char_start", 0))
        p_end = int(page.get("char_end", 0))
        if p_start <= start < p_end:
            return int(page.get("page", 0))
    return int(pages[-1].get("page", 0))


def build_structured_recipe(
    full_text: str,
    pages: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    ollama_base_url: str = LLM_BASE_URL,
    default_client: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    raw_data, err = extract_structure(
        client=client,
        model=model,
        full_text=full_text,
        ollama_base_url=ollama_base_url,
        default_client=default_client,
    )

    if err or raw_data is None:
        warnings.append(
            {
                "code": "structure_extraction_failed",
                "severity": "error",
                "message": f"Failed to structure recipe text: {err or 'invalid_json'}",
                "page": None,
            }
        )
        return (
            {
                "steps": [],
                "items": [],
                "servings": {
                    "id": "serving-001",
                    "value": None,
                    "inferred": True,
                    "inference_reason": "Structuring failed",
                    "quote": None,
                },
            },
            warnings,
        )

    raw_steps = raw_data.get("steps") if isinstance(raw_data.get("steps"), list) else []
    raw_items = raw_data.get("items") if isinstance(raw_data.get("items"), list) else []
    raw_servings = raw_data.get("servings")

    step_candidates: list[dict[str, Any]] = []
    for i, rs in enumerate(raw_steps):
        if not isinstance(rs, dict):
            continue

        orig_idx = rs.get("index", i)
        if not isinstance(orig_idx, int) or isinstance(orig_idx, bool):
            orig_idx = i

        text = _as_str(rs.get("text")) or ""
        quote = _as_str(rs.get("quote")) or ""
        inferred = _as_bool(rs.get("inferred", False))
        inference_reason = _as_str(rs.get("inference_reason"))
        duration = _as_str(rs.get("duration"))
        raw_conditions = (
            rs.get("conditions") if isinstance(rs.get("conditions"), list) else []
        )
        conditions = [_norm(str(c)) for c in raw_conditions if _norm(str(c))]
        continues_previous = _as_bool(rs.get("continues_previous", False))

        if not text:
            warnings.append(
                {
                    "code": "empty_step_dropped",
                    "severity": "warning",
                    "message": "Dropped step with empty text.",
                    "page": None,
                }
            )
            continue

        span = _find_span(full_text, quote) if quote else None
        page_idx = _page_for_offset(pages, span[0] if span else None)

        if not inferred:
            if not quote or span is None:
                warnings.append(
                    {
                        "code": "quote_not_found",
                        "severity": "warning",
                        "message": "Step quote not found in verified transcript; step dropped.",
                        "page": page_idx,
                        "quote": quote or None,
                    }
                )
                continue
        else:
            if not inference_reason:
                inference_reason = "Inferred step without explicit reason"
                warnings.append(
                    {
                        "code": "inference_without_reason",
                        "severity": "info",
                        "message": "Inferred step was missing an inference_reason.",
                        "page": page_idx,
                    }
                )
            if quote:
                warnings.append(
                    {
                        "code": "inferred_quote_cleared",
                        "severity": "info",
                        "message": "Cleared literal quote from inferred step.",
                        "page": page_idx,
                        "quote": quote,
                    }
                )
                quote = ""
                span = None

        if duration and _canonical(duration) not in _canonical(text):
            warnings.append(
                {
                    "code": "duration_not_in_step",
                    "severity": "warning",
                    "message": "Duration was not found in step text and was cleared.",
                    "page": page_idx,
                    "quote": duration,
                }
            )
            duration = None

        valid_conditions: list[str] = []
        for cond in conditions:
            if _canonical(cond) in _canonical(text):
                valid_conditions.append(cond)
            else:
                warnings.append(
                    {
                        "code": "condition_not_in_step",
                        "severity": "warning",
                        "message": "Condition was not found in step text and was removed.",
                        "page": page_idx,
                        "quote": cond,
                    }
                )

        step_candidates.append(
            {
                "orig_index": orig_idx,
                "order": i,
                "text": text,
                "quote": quote,
                "span": span,
                "page_index": page_idx,
                "inferred": inferred,
                "inference_reason": inference_reason,
                "duration": duration,
                "conditions": valid_conditions,
                "continues_previous": continues_previous,
            }
        )

    step_candidates.sort(key=lambda s: (s["orig_index"], s["order"]))

    steps: list[dict[str, Any]] = []
    step_id_by_orig: dict[int, str] = {}
    step_span_by_id: dict[str, tuple[int, int]] = {}

    for new_idx, cand in enumerate(step_candidates):
        sid = f"step-{new_idx + 1:03d}"
        step_id_by_orig.setdefault(cand["orig_index"], sid)
        if cand["span"] is not None:
            step_span_by_id[sid] = cand["span"]

        continues = cand["continues_previous"]
        if continues and new_idx == 0:
            continues = False
            warnings.append(
                {
                    "code": "invalid_continues_previous",
                    "severity": "info",
                    "message": "Reset continues_previous=True on the first step.",
                    "page": cand["page_index"],
                    "step_id": sid,
                }
            )

        steps.append(
            {
                "id": sid,
                "index": new_idx,
                "text": cand["text"],
                "quote": cand["quote"],
                "inferred": cand["inferred"],
                "inference_reason": cand["inference_reason"],
                "page_index": cand["page_index"],
                "span": (
                    {"start": cand["span"][0], "end": cand["span"][1]}
                    if cand["span"]
                    else None
                ),
                "item_ids": [],
                "duration": cand["duration"],
                "conditions": cand["conditions"],
                "continues_previous": continues,
            }
        )

    located_spans = sorted(step_span_by_id.values(), key=lambda pair: pair[0])
    for (_, prev_end), (next_start, _) in zip(located_spans, located_spans[1:]):
        if next_start <= prev_end:
            continue
        gap = full_text[prev_end:next_start]
        if _WORD_CHARS.search(gap):
            g_start = prev_end + (len(gap) - len(gap.lstrip()))
            g_end = next_start - (len(gap) - len(gap.rstrip()))
            warnings.append(
                {
                    "code": "unmapped_text",
                    "severity": "warning",
                    "message": "Text segment between steps was skipped during extraction.",
                    "page": _page_for_offset(pages, g_start),
                    "quote": full_text[g_start:g_end],
                    "span": {"start": g_start, "end": g_end},
                }
            )

    def _resolve_step_id(raw_idx: Any) -> str | None:
        if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
            if raw_idx in step_id_by_orig:
                return step_id_by_orig[raw_idx]
            if 0 <= raw_idx < len(steps):
                return steps[raw_idx]["id"]
        return steps[0]["id"] if steps else None

    items: list[dict[str, Any]] = []
    seen_items: set[tuple[str, str, str, str, str, str]] = set()

    for ri in raw_items:
        if not isinstance(ri, dict):
            continue

        sid = _resolve_step_id(ri.get("step_index", 0))
        if sid is None:
            warnings.append(
                {
                    "code": "orphan_item_dropped",
                    "severity": "warning",
                    "message": "Dropped item because no valid steps exist.",
                    "page": None,
                }
            )
            continue

        name = _as_str(ri.get("name")) or ""
        if not name:
            continue

        item_type = ri.get("type") if ri.get("type") in ("ingredient", "tool") else "ingredient"
        quantity = _as_str(ri.get("quantity"))
        unit = _as_str(ri.get("unit"))
        preparation = _as_str(ri.get("preparation"))
        optional = _as_bool(ri.get("optional", False))
        inferred = _as_bool(ri.get("inferred", False))
        inference_reason = _as_str(ri.get("inference_reason"))
        quote = _as_str(ri.get("quote")) or ""

        step_span = step_span_by_id.get(sid)
        span: tuple[int, int] | None = None

        if not inferred:
            if quote and step_span is not None:
                local_span = _find_span(full_text[step_span[0] : step_span[1]], quote)
                if local_span is not None:
                    span = (step_span[0] + local_span[0], step_span[0] + local_span[1])

            if span is None and quote:
                global_span = _find_span(full_text, quote)
                if global_span is not None:
                    span = global_span
                    warnings.append(
                        {
                            "code": "step_scope_mismatch",
                            "severity": "warning",
                            "message": "Item quote matched outside its assigned step boundaries.",
                            "page": _page_for_offset(pages, span[0]),
                            "step_id": sid,
                            "quote": quote,
                        }
                    )

            if not quote or span is None:
                warnings.append(
                    {
                        "code": "quote_not_found",
                        "severity": "warning",
                        "message": "Explicit item quote not found in text; item dropped.",
                        "page": _page_for_offset(pages, step_span[0] if step_span else None),
                        "step_id": sid,
                        "quote": quote or None,
                    }
                )
                continue
        else:
            if not inference_reason:
                inference_reason = "Implied by recipe step"
            if quote:
                quote = ""
            span = None

        page_idx = _page_for_offset(
            pages, span[0] if span else (step_span[0] if step_span else None)
        )

        dedup_key = (
            sid,
            item_type,
            _canonical(name),
            _canonical(quantity),
            _canonical(unit),
            _canonical(preparation),
        )
        if dedup_key in seen_items:
            warnings.append(
                {
                    "code": "duplicate_merged",
                    "severity": "info",
                    "message": "Duplicate item within the same step removed.",
                    "page": page_idx,
                    "step_id": sid,
                    "quote": quote or name,
                }
            )
            continue
        seen_items.add(dedup_key)

        iid = f"item-{len(items) + 1:03d}"
        items.append(
            {
                "id": iid,
                "step_id": sid,
                "name": name,
                "type": item_type,
                "quantity": quantity,
                "unit": unit,
                "preparation": preparation,
                "optional": optional,
                "inferred": inferred,
                "inference_reason": inference_reason,
                "quote": quote,
                "page_index": page_idx,
                "span": {"start": span[0], "end": span[1]} if span else None,
            }
        )

    items_by_step: dict[str, list[str]] = {}
    for item in items:
        items_by_step.setdefault(item["step_id"], []).append(item["id"])
    for step in steps:
        step["item_ids"] = items_by_step.get(step["id"], [])

    rs_obj = raw_servings if isinstance(raw_servings, dict) else {}
    s_val = _as_int(rs_obj.get("value"))
    s_quote = _as_str(rs_obj.get("quote"))
    s_inferred = _as_bool(rs_obj.get("inferred", True))
    s_reason = _as_str(rs_obj.get("inference_reason"))
    s_span = _find_span(full_text, s_quote) if s_quote else None

    if s_quote and s_span is None:
        warnings.append(
            {
                "code": "quote_not_found",
                "severity": "warning",
                "message": "Servings quote not found in verified transcript; reset to inferred.",
                "page": None,
                "quote": s_quote,
            }
        )
        s_val = None
        s_quote = None
        s_span = None
        s_inferred = True
        s_reason = "Servings quote not found in text"

    if s_val is None or s_val <= 0:
        s_val = None
        s_inferred = True
        s_quote = None
        s_span = None
        if not s_reason:
            s_reason = "No serving count stated in the text"
        warnings.append(
            {
                "code": "servings_inferred",
                "severity": "info",
                "message": "Serving count not stated in recipe text.",
                "page": None,
                "reason": s_reason,
            }
        )
    elif s_inferred:
        if not s_reason:
            s_reason = "Servings inferred from recipe proportions"
        if s_quote:
            s_quote = None
            s_span = None
        warnings.append(
            {
                "code": "servings_inferred",
                "severity": "info",
                "message": "Serving count was inferred rather than explicitly stated.",
                "page": None,
                "reason": s_reason,
            }
        )

    servings = {
        "id": "serving-001",
        "value": s_val,
        "inferred": s_inferred,
        "inference_reason": s_reason,
        "quote": s_quote,
        "span": {"start": s_span[0], "end": s_span[1]} if s_span else None,
    }

    return {"steps": steps, "items": items, "servings": servings}, warnings