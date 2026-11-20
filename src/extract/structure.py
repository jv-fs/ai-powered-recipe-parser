from __future__ import annotations

import json

from openai import OpenAI

from .llm import _call_llm
from .models import Evidence, ExtractWarning, ItemMention, OcrLine, Reading, Servings, Step
from .prompts import _STRUCTURE_PROMPT, _STRUCTURE_SCHEMA
from .text import (
    _DURATION,
    _GERMAN_PREFIXES,
    _PLAUSIBLE_UNITS,
    _QUANTITY_VALID,
    _as_str,
    _canon,
    _canonical_unit,
    _norm,
    _tokens,
)
from .verification import _refine_item_bbox


def _structure(
    evidences: list[Evidence],
    ocr_lines_by_page: dict[int, list[OcrLine]],
    client: OpenAI,
    model: str,
    ollama_base_url: str,
    default_client: bool,
) -> tuple[list[Step], list[ItemMention], Servings, list[ExtractWarning]]:
    warnings: list[ExtractWarning] = []
    if not evidences:
        return [], [], Servings(None, True, None), warnings

    recipe_evidences = [e for e in evidences if e.is_recipe] or evidences
    ev_index = {e.id: e for e in evidences}
    payload = {
        "evidence": [
            {"id": e.id, "text": e.text, "page": e.page} for e in recipe_evidences
        ]
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _STRUCTURE_PROMPT},
                {
                    "type": "text",
                    "text": "<source>\n" + json.dumps(payload, ensure_ascii=False) + "\n</source>",
                },
            ],
        }
    ]
    data, err = _call_llm(
        client, model, messages, ollama_base_url, default_client, _STRUCTURE_SCHEMA
    )
    if err or data is None:
        code = "invalid_json" if err == "invalid_json" else "model_error"
        warnings.append(
            ExtractWarning(code, f"Structuring failed: {err}", field="item")
        )
        for e in recipe_evidences:
            warnings.append(
                ExtractWarning(
                    "structure_mismatch",
                    "Evidence not structured due to model failure",
                    page=e.page,
                    source_index=e.source_index,
                    field="item",
                    quote=e.text,
                    bbox=e.bbox,
                    char_start=e.char_start,
                    char_end=e.char_end,
                    readings=(Reading("accepted_evidence", e.text),),
                    reason="failure in structuring stage",
                )
            )
        return [], [], Servings(None, True, None), warnings

    raw_steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    steps: list[Step] = []
    step_id_by_index: dict[int, str] = {}
    referenced_evs: set[str] = set()
    seen_step_texts: set[str] = set()

    for i, rs in enumerate(raw_steps):
        if not isinstance(rs, dict):
            continue
        idx = rs.get("index", i)
        if not isinstance(idx, int) or isinstance(idx, bool):
            idx = i
        sid = f"step-{len(steps) + 1:03d}"
        step_id_by_index[idx] = sid
        text = _norm(str(rs.get("text") or ""))
        ev_ids = (
            [eid for eid in rs.get("evidence_ids", []) if isinstance(eid, str)]
            if isinstance(rs.get("evidence_ids"), list)
            else []
        )
        matched_evs = [ev_index[eid] for eid in ev_ids if eid in ev_index]
        for e in matched_evs:
            referenced_evs.add(e.id)
        first_ev = matched_evs[0] if matched_evs else None
        page_idx = first_ev.page if first_ev else (recipe_evidences[0].page if recipe_evidences else 0)

        norm_step = _canon(text)
        if not norm_step:
            warnings.append(
                ExtractWarning(
                    "step_conflict",
                    "Step extracted with empty text",
                    page=page_idx,
                    step_id=sid,
                    field="step",
                    reason="empty step text",
                )
            )
        elif norm_step in seen_step_texts:
            warnings.append(
                ExtractWarning(
                    "step_order_conflict",
                    "Duplicate step detected",
                    page=page_idx,
                    step_id=sid,
                    field="step",
                    quote=text,
                    bbox=first_ev.bbox if first_ev else None,
                    readings=(Reading("llm_structured", text),),
                    selected=text,
                    reason="step text identical to previous step",
                )
            )
        seen_step_texts.add(norm_step)

        if not matched_evs:
            warnings.append(
                ExtractWarning(
                    "step_conflict",
                    f"Step {sid} does not reference any valid evidence_id",
                    page=page_idx,
                    step_id=sid,
                    field="step",
                    quote=text or None,
                    reason="missing or invalid evidence_ids on step",
                )
            )
        elif norm_step:
            ev_combined = " ".join(e.text for e in matched_evs)
            s_toks = _tokens(text)
            e_toks = _tokens(ev_combined)
            overlap = len(s_toks & e_toks) / max(len(s_toks), 1) if s_toks else 0.0
            if overlap < 0.3:
                warnings.append(
                    ExtractWarning(
                        "step_conflict",
                        f"Step {sid} text does not match referenced evidence_ids",
                        page=page_idx,
                        step_id=sid,
                        field="step",
                        quote=text,
                        bbox=first_ev.bbox if first_ev else None,
                        readings=(
                            Reading("accepted_evidence", ev_combined),
                            Reading("llm_structured", text),
                        ),
                        selected=text,
                        reason="step text not grounded in referenced evidence",
                    )
                )

        duration = _as_str(rs.get("duration"))
        if duration:
            dur_tokens = _tokens(duration)
            step_tokens = _tokens(text)
            if dur_tokens and not (dur_tokens & step_tokens):
                warnings.append(
                    ExtractWarning(
                        "step_conflict",
                        f"Duration not found in step text: {duration!r}",
                        page=page_idx,
                        step_id=sid,
                        field="step",
                        quote=text,
                        bbox=first_ev.bbox if first_ev else None,
                        readings=(
                            Reading("accepted_evidence", text),
                            Reading("llm_structured", duration),
                        ),
                        selected=None,
                        reason="duration not grounded in step text",
                    )
                )
                duration = None
            elif not _DURATION.search(duration):
                warnings.append(
                    ExtractWarning(
                        "step_conflict",
                        f"Duration does not match expected pattern: {duration!r}",
                        page=page_idx,
                        step_id=sid,
                        field="step",
                        quote=text,
                        bbox=first_ev.bbox if first_ev else None,
                        readings=(Reading("llm_structured", duration),),
                        selected=None,
                        reason="invalid duration format",
                    )
                )
                duration = None
        elif norm_step and _DURATION.search(text):
            warnings.append(
                ExtractWarning(
                    "step_conflict",
                    f"Temporal expression detected in step {sid} text, but duration is null",
                    page=page_idx,
                    step_id=sid,
                    field="step",
                    quote=text,
                    bbox=first_ev.bbox if first_ev else None,
                    readings=(
                        Reading("accepted_evidence", text),
                        Reading("llm_structured", None),
                    ),
                    selected=None,
                    reason="duration omitted despite temporal expression in step text",
                )
            )

        raw_conditions = rs.get("conditions")
        conditions = (
            tuple(_norm(c) for c in raw_conditions if isinstance(c, str) and _norm(c))
            if isinstance(raw_conditions, list)
            else ()
        )
        continues_previous = bool(rs.get("continues_previous"))
        if continues_previous and not steps:
            warnings.append(
                ExtractWarning(
                    "step_order_conflict",
                    "Initial step marked as continuation without a preceding step",
                    page=page_idx,
                    step_id=sid,
                    field="step",
                    quote=text,
                    reason="continuation without preceding step",
                )
            )
            continues_previous = False

        steps.append(
            Step(
                sid,
                len(steps),
                text,
                page_idx,
                (),
                duration,
                conditions,
                continues_previous,
            )
        )

    step_by_id = {s.id: s for s in steps}
    raw_items = data.get("items") if isinstance(data.get("items"), list) else []
    items: list[ItemMention] = []
    seen_items: set[tuple[str, str, str, str]] = set()

    for ri in raw_items:
        if not isinstance(ri, dict):
            continue
        ev_id = ri.get("evidence_id")
        ev = ev_index.get(ev_id) if isinstance(ev_id, str) else None
        step_idx = ri.get("step_index", 0)
        sid = step_id_by_index.get(step_idx)
        iid = f"item-{len(items) + 1:03d}"
        name = _as_str(ri.get("name"))
        itype = ri.get("type") if ri.get("type") in ("ingredient", "tool") else "ingredient"
        qty = _as_str(ri.get("quantity"))
        unit = _as_str(ri.get("unit"))
        prep = _as_str(ri.get("preparation"))
        optional = bool(ri.get("optional"))
        inferred = bool(ri.get("inferred"))
        inference_reason = _as_str(ri.get("inference_reason"))
        quote = _norm(str(ri.get("quote") or ""))

        if sid is None:
            warnings.append(
                ExtractWarning(
                    "step_conflict",
                    f"Item references non-existent step_index: {step_idx}",
                    page=ev.page if ev else None,
                    source_index=ev.source_index if ev else None,
                    item_id=iid,
                    field="step",
                    quote=quote or None,
                    bbox=ev.bbox if ev else None,
                    reason="invalid step_index",
                )
            )
            if steps:
                sid = steps[0].id
            else:
                continue

        if ev is None:
            warnings.append(
                ExtractWarning(
                    "structure_mismatch",
                    f"Item references non-existent evidence_id: {ev_id}",
                    step_id=sid,
                    item_id=iid,
                    field="item",
                    quote=quote or None,
                    readings=(Reading("llm_structured", json.dumps(ri, ensure_ascii=False)),),
                    reason="invalid evidence_id",
                )
            )
            continue

        referenced_evs.add(ev.id)
        step_obj = step_by_id.get(sid)
        grounding = ev.grounding
        item_bbox = (
            _refine_item_bbox(quote, ocr_lines_by_page.get(ev.page, []), ev.bbox)
            if quote
            else ev.bbox
        )

        if inferred and not inference_reason:
            inference_reason = "Value marked as inferred without providing a reason"
            warnings.append(
                ExtractWarning(
                    "model_json_invalid",
                    "Inferred item has no inference_reason",
                    page=ev.page,
                    source_index=ev.source_index,
                    step_id=sid,
                    item_id=iid,
                    field="item",
                    quote=quote or None,
                    bbox=item_bbox,
                    reason="missing inference_reason",
                )
            )

        if not quote:
            grounding = "unverified"
            warnings.append(
                ExtractWarning(
                    "missing_quote",
                    f"Item '{name or ''}' has empty quote",
                    page=ev.page,
                    source_index=ev.source_index,
                    step_id=sid,
                    item_id=iid,
                    field="quote",
                    bbox=item_bbox,
                    readings=(
                        Reading("accepted_evidence", ev.text),
                        Reading("llm_structured", ""),
                    ),
                    selected=None,
                    reason="empty quote",
                )
            )
        else:
            in_ev = _canon(quote) in _canon(ev.text)
            in_step = bool(step_obj and _canon(quote) in _canon(step_obj.text))
            if step_obj and not in_step:
                grounding = "unverified"
                warnings.append(
                    ExtractWarning(
                        "step_conflict",
                        f"Quote for item '{name or ''}' not found in assigned step {sid}",
                        page=ev.page,
                        source_index=ev.source_index,
                        step_id=sid,
                        item_id=iid,
                        field="step",
                        quote=quote,
                        bbox=item_bbox,
                        char_start=ev.char_start,
                        char_end=ev.char_end,
                        readings=(
                            Reading("accepted_evidence", step_obj.text),
                            Reading("llm_structured", quote),
                        ),
                        selected=step_obj.text,
                        reason="item linked to step whose text does not contain quote",
                    )
                )
            if not in_ev:
                grounding = "unverified"
                warnings.append(
                    ExtractWarning(
                        "structure_mismatch",
                        f"Quote for item '{name or ''}' not found in referenced evidence {ev.id}",
                        page=ev.page,
                        source_index=ev.source_index,
                        step_id=sid,
                        item_id=iid,
                        field="quote",
                        quote=quote,
                        bbox=item_bbox,
                        char_start=ev.char_start,
                        char_end=ev.char_end,
                        readings=(
                            Reading("accepted_evidence", ev.text),
                            Reading("llm_structured", quote),
                        ),
                        selected=ev.text,
                        reason="quote not found verbatim in referenced evidence",
                    )
                )

        if qty is not None and not _QUANTITY_VALID.match(qty):
            warnings.append(
                ExtractWarning(
                    "quantity_conflict",
                    f"Non-numeric quantity for item '{name or ''}': {qty!r}",
                    page=ev.page,
                    source_index=ev.source_index,
                    step_id=sid,
                    item_id=iid,
                    field="quantity",
                    quote=quote or ev.text,
                    bbox=item_bbox,
                    readings=(Reading("llm_structured", qty),),
                    selected=qty,
                    reason="quantity contains non-numeric text",
                )
            )

        if unit is not None:
            canon_u = _canonical_unit(unit)
            if qty is None:
                warnings.append(
                    ExtractWarning(
                        "unit_conflict",
                        f"Unit '{unit}' discarded for item '{name or ''}' due to missing quantity",
                        page=ev.page,
                        source_index=ev.source_index,
                        step_id=sid,
                        item_id=iid,
                        field="unit",
                        quote=quote or ev.text,
                        bbox=item_bbox,
                        readings=(Reading("llm_structured", unit),),
                        selected=None,
                        reason="unit without associated quantity",
                    )
                )
                unit = None
            elif canon_u not in _PLAUSIBLE_UNITS:
                warnings.append(
                    ExtractWarning(
                        "unit_conflict",
                        f"Implausible unit '{unit}' discarded for item '{name or ''}'",
                        page=ev.page,
                        source_index=ev.source_index,
                        step_id=sid,
                        item_id=iid,
                        field="unit",
                        quote=quote or ev.text,
                        bbox=item_bbox,
                        readings=(Reading("llm_structured", unit),),
                        selected=None,
                        reason="unit not in known measurement units",
                    )
                )
                unit = None
            else:
                unit = canon_u

        if prep is not None:
            prep_fold = prep.casefold()
            if prep_fold in _GERMAN_PREFIXES or len(prep_fold) <= 2:
                warnings.append(
                    ExtractWarning(
                        "suspicious_preparation",
                        f"Suspicious preparation (prefix or fragment) for item '{name or ''}': {prep!r}",
                        page=ev.page,
                        source_index=ev.source_index,
                        step_id=sid,
                        item_id=iid,
                        field="preparation",
                        quote=quote or ev.text,
                        bbox=item_bbox,
                        readings=(Reading("llm_structured", prep),),
                        selected=prep,
                        reason="possible isolated verb particle",
                    )
                )

        dedupe_key = (
            _canon(name),
            _canon(unit),
            _canon(prep),
            sid,
        )
        if dedupe_key in seen_items:
            warnings.append(
                ExtractWarning(
                    "duplicate_merged",
                    f"Duplicate item ignored in step {sid}: '{name or ''}'",
                    page=ev.page,
                    source_index=ev.source_index,
                    step_id=sid,
                    item_id=iid,
                    field="item",
                    quote=quote or None,
                    bbox=item_bbox,
                    reason="duplicate item in the same step",
                )
            )
            continue
        seen_items.add(dedupe_key)

        items.append(
            ItemMention(
                id=iid,
                step_id=sid,
                name=name or "",
                type=itype,
                quantity=qty,
                unit=unit,
                preparation=prep,
                optional=optional,
                inferred=inferred,
                inference_reason=inference_reason,
                quote=quote,
                grounding=grounding,
                page_index=ev.page,
                source_index=ev.source_index,
                bbox=item_bbox,
                char_start=ev.char_start,
                char_end=ev.char_end,
            )
        )

    for e in recipe_evidences:
        if e.id not in referenced_evs:
            warnings.append(
                ExtractWarning(
                    "structure_mismatch",
                    "Recipe evidence not represented in structuring",
                    page=e.page,
                    source_index=e.source_index,
                    field="item",
                    quote=e.text,
                    bbox=e.bbox,
                    char_start=e.char_start,
                    char_end=e.char_end,
                    readings=(
                        Reading("accepted_evidence", e.text),
                        Reading("llm_structured", None),
                    ),
                    selected=e.text,
                    reason="recipe segment omitted during structuring",
                )
            )

    rsrv = data.get("servings")
    if isinstance(rsrv, list):
        warnings.append(
            ExtractWarning(
                "model_json_invalid",
                "Servings returned as a list; using first object",
                field="servings",
                reason="invalid type for servings",
            )
        )
        rsrv = next((entry for entry in rsrv if isinstance(entry, dict)), {})
    elif not isinstance(rsrv, dict):
        rsrv = {}

    sval = rsrv.get("value")
    sinferred = bool(rsrv.get("inferred", True))
    squote = _as_str(rsrv.get("quote"))
    if not isinstance(sval, int) or isinstance(sval, bool) or sval <= 0:
        sval = None
        sinferred = True
        squote = None
        warnings.append(
            ExtractWarning(
                "servings_inferred",
                "Servings missing or not positive",
                field="servings",
                reason="servings missing or not positive",
            )
        )
    elif sinferred:
        warnings.append(
            ExtractWarning(
                "servings_inferred",
                "Servings inferred by the model",
                field="servings",
                quote=squote,
                reason="servings inferred by the model",
            )
        )
    servings = Servings(sval, sinferred, squote)

    by_step: dict[str, list[str]] = {}
    for it in items:
        by_step.setdefault(it.step_id, []).append(it.id)
    for s in steps:
        if not by_step.get(s.id):
            warnings.append(
                ExtractWarning(
                    "step_conflict",
                    f"Step {s.id} has no linked items",
                    page=s.page_index,
                    step_id=s.id,
                    field="step",
                    quote=s.text,
                    reason="no ingredients or tools extracted for this step",
                )
            )
    steps = [
        Step(
            s.id,
            s.index,
            s.text,
            s.page_index,
            tuple(by_step.get(s.id, ())),
            s.duration,
            s.conditions,
            s.continues_previous,
        )
        for s in steps
    ]

    return steps, items, servings, warnings