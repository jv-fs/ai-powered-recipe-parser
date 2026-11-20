from __future__ import annotations

from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR

from ..ingest import IngestResult
from .models import Evidence, ExtractResult, ExtractWarning
from .structure import _structure
from .verification import _process_page


def extract(
    result: IngestResult,
    *,
    model: str = "qwen2.5vl:7b",
    client: OpenAI | None = None,
    ocr: RapidOCR | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
) -> ExtractResult:
    default_client = client is None
    client = client or OpenAI(base_url=ollama_base_url, api_key="ollama")
    ocr = ocr or RapidOCR()
    warnings: list[ExtractWarning] = []
    evidences: list[Evidence] = []
    ocr_lines_by_page: dict[int, list[OcrLine]] = {}

    for page in result.pages:
        evs, page_warnings, page_ocr_lines = _process_page(
            page, client, model, ocr, ollama_base_url, default_client
        )
        warnings.extend(page_warnings)
        evidences.extend(evs)
        ocr_lines_by_page[page.index] = page_ocr_lines

    evidences = [
        Evidence(
            f"ev-{i:03d}",
            e.page,
            e.source_index,
            e.bbox,
            e.char_start,
            e.char_end,
            e.text,
            e.is_recipe,
            e.grounding,
        )
        for i, e in enumerate(evidences, 1)
    ]

    steps, items, servings, struct_warnings = _structure(
        evidences, ocr_lines_by_page, client, model, ollama_base_url, default_client
    )
    warnings.extend(struct_warnings)
    return ExtractResult(tuple(steps), tuple(items), servings, tuple(warnings), model)
