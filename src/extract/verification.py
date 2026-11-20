from __future__ import annotations

from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR

from ..ingest import Page
from .llm import _transcribe, _transcribe_crop
from .models import Evidence, ExtractWarning, OcrLine, Reading
from .ocr import _crop, _run_ocr
from .text import _find_span, _merge_bboxes, _norm, _same, _tokens


def _match_ocr_to_segment(
    segment_text: str,
    ocr_lines: list[OcrLine],
) -> tuple[str | None, tuple[int, int, int, int] | None, float | None]:
    seg_tokens = _tokens(segment_text)
    if not seg_tokens or not ocr_lines:
        return None, None, None
    matched_lines: list[OcrLine] = []
    for ol in ocr_lines:
        ot = _tokens(ol.text)
        if not ot:
            continue
        overlap = len(seg_tokens & ot) / len(ot)
        if overlap >= 0.45:
            matched_lines.append(ol)
    if not matched_lines:
        return None, None, None
    joined_text = _norm(" ".join(l.text for l in matched_lines))
    bbox = _merge_bboxes([l.bbox for l in matched_lines if l.bbox is not None])
    scores = [l.score for l in matched_lines if l.score is not None]
    avg_score = sum(scores) / len(scores) if scores else None
    return joined_text, bbox, avg_score


def _refine_item_bbox(
    quote: str,
    ocr_lines: list[OcrLine],
    fallback_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    q_tokens = _tokens(quote)
    if not q_tokens or not ocr_lines:
        return fallback_bbox
    best_bbox: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for ol in ocr_lines:
        if ol.bbox is None:
            continue
        ot = _tokens(ol.text)
        if not ot:
            continue
        score = len(q_tokens & ot) / len(q_tokens)
        if score > best_score:
            best_score = score
            best_bbox = ol.bbox
    return best_bbox if best_score >= 0.5 and best_bbox is not None else fallback_bbox


def _recheck_crop(
    client: OpenAI,
    model: str,
    image: bytes,
    bbox: tuple[int, int, int, int] | None,
    ocr: RapidOCR,
    ollama_base_url: str,
    default_client: bool,
) -> tuple[Reading, Reading]:
    img = _crop(image, bbox) if bbox else None
    if img:
        lines, err = _run_ocr(img, ocr)
        if err or lines is None:
            ocr_val, ocr_score = None, None
        else:
            ocr_val = _norm(" ".join(l.text for l in lines)) or None
            scores = [l.score for l in lines if l.score is not None]
            ocr_score = sum(scores) / len(scores) if scores else None
        llm_val, err = _transcribe_crop(client, model, img, ollama_base_url, default_client)
        if err or not llm_val:
            llm_val = None
    else:
        ocr_val, ocr_score, llm_val = None, None, None
    return Reading("crop_ocr", ocr_val, ocr_score), Reading("crop_llm", llm_val, None)


def _process_page(
    page: Page,
    client: OpenAI,
    model: str,
    ocr: RapidOCR,
    ollama_base_url: str,
    default_client: bool,
) -> tuple[list[Evidence], list[ExtractWarning], list[OcrLine]]:
    warnings: list[ExtractWarning] = []
    evidences: list[Evidence] = []

    if not page.image:
        if page.text:
            evidences.append(
                Evidence(
                    "",
                    page.index,
                    page.source_index,
                    None,
                    page.char_start,
                    page.char_end,
                    page.text,
                    True,
                    "text_layer",
                )
            )
        return evidences, warnings, []

    ocr_lines, ocr_err = _run_ocr(page.image, ocr)
    if ocr_err or ocr_lines is None:
        warnings.append(
            ExtractWarning(
                "ocr_error",
                f"OCR failed: {ocr_err}",
                page=page.index,
                source_index=page.source_index,
                field="transcript",
            )
        )
        ocr_lines = []

    llm_segments, llm_err = _transcribe(
        client, model, page, ollama_base_url, default_client
    )
    if llm_err:
        code = "invalid_json" if llm_err == "invalid_json" else "model_error"
        warnings.append(
            ExtractWarning(
                code,
                f"Transcript failed: {llm_err}",
                page=page.index,
                source_index=page.source_index,
                field="transcript",
            )
        )
        llm_segments = []

    has_text_layer = bool(page.text or page.raw_text)

    if llm_segments:
        for seg_text, is_recipe in llm_segments:
            ocr_val, bbox, ocr_score = _match_ocr_to_segment(seg_text, ocr_lines)
            cs, ce = _find_span(page, seg_text) if has_text_layer else (None, None)
            in_text_layer = cs is not None
            r_llm = Reading("llm", seg_text)
            r_ocr = Reading("ocr", ocr_val, ocr_score) if ocr_val else None
            base_readings = tuple(r for r in (r_llm, r_ocr) if r is not None)

            grounding = "text_layer" if in_text_layer else ("full_ocr" if ocr_val else "vision")

            if is_recipe and ocr_val and not _same(ocr_val, seg_text):
                if bbox and page.image:
                    cr_ocr, cr_llm = _recheck_crop(
                        client, model, page.image, bbox, ocr, ollama_base_url, default_client
                    )
                    readings = base_readings + (cr_ocr, cr_llm)
                    if cr_ocr.value and _same(cr_ocr.value, seg_text):
                        grounding = "crop_ocr"
                    else:
                        warnings.append(
                            ExtractWarning(
                                "mismatch",
                                "Mismatch between visual reading (LLM) and OCR in segment",
                                page=page.index,
                                source_index=page.source_index,
                                field="transcript",
                                quote=seg_text,
                                bbox=bbox,
                                char_start=cs,
                                char_end=ce,
                                readings=readings,
                                selected=seg_text,
                                reason="kept full visual reading after OCR mismatch",
                            )
                        )
                else:
                    warnings.append(
                        ExtractWarning(
                            "mismatch",
                            "Mismatch between visual reading (LLM) and OCR without bbox for crop",
                            page=page.index,
                            source_index=page.source_index,
                            field="transcript",
                            quote=seg_text,
                            bbox=None,
                            char_start=cs,
                            char_end=ce,
                            readings=base_readings,
                            selected=seg_text,
                            reason="kept full visual reading",
                        )
                    )

            evidences.append(
                Evidence(
                    "",
                    page.index,
                    page.source_index,
                    bbox,
                    cs,
                    ce,
                    seg_text,
                    is_recipe,
                    grounding,
                )
            )
    elif ocr_lines:
        for ol in ocr_lines:
            cs, ce = _find_span(page, ol.text) if has_text_layer else (None, None)
            evidences.append(
                Evidence(
                    "",
                    page.index,
                    page.source_index,
                    ol.bbox,
                    cs,
                    ce,
                    ol.text,
                    True,
                    "full_ocr",
                )
            )
    elif has_text_layer:
        txt = _norm(page.text or page.raw_text)
        if txt:
            evidences.append(
                Evidence(
                    "",
                    page.index,
                    page.source_index,
                    None,
                    page.char_start,
                    page.char_end,
                    txt,
                    True,
                    "text_layer",
                )
            )

    return evidences, warnings, ocr_lines
