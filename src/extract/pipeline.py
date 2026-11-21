from __future__ import annotations

from typing import Any

from openai import OpenAI

from ..config import LLM_API_KEY, LLM_BASE_URL, VISION_MODEL
from .Collector import collect_document_readings
from .LLMMethods import is_default_client
from .TextVerifier import verify_transcript
from .JSONBuilder import build_structured_recipe


def extract(
    result: Any,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    active_model = model or VISION_MODEL
    active_base_url = ollama_base_url or LLM_BASE_URL
    default_client = client is None or is_default_client(client, active_base_url)
    if client is None:
        client = OpenAI(base_url=active_base_url, api_key=LLM_API_KEY)

    pages = getattr(result, "pages", [])
    readings_list = collect_document_readings(
        pages=pages,
        client=client,
        model=active_model,
        ollama_base_url=active_base_url,
        default_client=default_client,
    )

    verified_pages: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    assembled_parts: list[str] = []
    offset = 0

    for idx, reading in enumerate(readings_list):
        if idx > 0:
            assembled_parts.append("\n")
            offset += 1

        if reading.native_text is not None:
            page_text = reading.native_text
            page_warnings: list[dict[str, Any]] = []
            origin = "text_layer"
        else:
            page_text, page_warnings = verify_transcript(
                full_text=reading.full_text,
                top_text=reading.top_text,
                bottom_text=reading.bottom_text,
                page_index=reading.page_index,
            )
            origin = "llm_verified"

        for err in reading.errors:
            all_warnings.append(
                {
                    "code": "collection_error",
                    "severity": "warning",
                    "message": err,
                    "page": reading.page_index,
                }
            )

        all_warnings.extend(page_warnings)

        start = offset
        end = offset + len(page_text)
        assembled_parts.append(page_text)
        offset = end

        verified_pages.append(
            {
                "page": reading.page_index,
                "source_index": reading.source_index,
                "text": page_text,
                "origin": origin,
                "char_start": start,
                "char_end": end,
                "readings": {
                    "native_text": reading.native_text,
                    "full_text": reading.full_text,
                    "top_text": reading.top_text,
                    "bottom_text": reading.bottom_text,
                },
            }
        )

    full_text = "".join(assembled_parts)

    structured, struct_warnings = build_structured_recipe(
        full_text=full_text,
        pages=verified_pages,
        client=client,
        model=active_model,
        ollama_base_url=active_base_url,
        default_client=default_client,
    )
    all_warnings.extend(struct_warnings)

    return {
        "text": full_text,
        "pages": verified_pages,
        "steps": structured["steps"],
        "items": structured["items"],
        "servings": structured["servings"],
        "warnings": all_warnings,
        "model": active_model,
    }