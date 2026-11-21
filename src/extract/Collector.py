from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ..config import LLM_BASE_URL
from .ImageManipulation import split_vertical_two_thirds
from .LLMMethods import transcribe_image


@dataclass(frozen=True)
class PageReadings:
    page_index: int
    source_index: int
    native_text: str | None
    full_text: str
    top_text: str
    bottom_text: str
    errors: tuple[str, ...]


def collect_page_readings(
    page: Any,
    client: OpenAI,
    model: str,
    ollama_base_url: str = LLM_BASE_URL,
    default_client: bool = True,
) -> PageReadings:
    page_index = getattr(page, "index", 0)
    source_index = getattr(page, "source_index", 0)
    native_text = getattr(page, "text", None)
    image_bytes = getattr(page, "image", None)

    if native_text is not None and native_text.strip():
        cleaned = " ".join(native_text.split())
        return PageReadings(
            page_index=page_index,
            source_index=source_index,
            native_text=cleaned,
            full_text=cleaned,
            top_text="",
            bottom_text="",
            errors=(),
        )

    if not image_bytes:
        return PageReadings(
            page_index=page_index,
            source_index=source_index,
            native_text=None,
            full_text="",
            top_text="",
            bottom_text="",
            errors=("missing_page_image",),
        )

    errors: list[str] = []

    full_text, full_err = transcribe_image(
        client=client,
        model=model,
        image_bytes=image_bytes,
        ollama_base_url=ollama_base_url,
        default_client=default_client,
    )
    if full_err:
        errors.append(f"full_transcript_error: {full_err}")

    try:
        top_bytes, bottom_bytes = split_vertical_two_thirds(image_bytes)
    except Exception as exc:
        errors.append(f"crop_split_error: {exc}")
        return PageReadings(
            page_index=page_index,
            source_index=source_index,
            native_text=None,
            full_text=full_text or "",
            top_text="",
            bottom_text="",
            errors=tuple(errors),
        )

    top_text, top_err = transcribe_image(
        client=client,
        model=model,
        image_bytes=top_bytes,
        ollama_base_url=ollama_base_url,
        default_client=default_client,
    )
    if top_err:
        errors.append(f"top_transcript_error: {top_err}")

    bottom_text, bottom_err = transcribe_image(
        client=client,
        model=model,
        image_bytes=bottom_bytes,
        ollama_base_url=ollama_base_url,
        default_client=default_client,
    )
    if bottom_err:
        errors.append(f"bottom_transcript_error: {bottom_err}")

    return PageReadings(
        page_index=page_index,
        source_index=source_index,
        native_text=None,
        full_text=full_text or "",
        top_text=top_text or "",
        bottom_text=bottom_text or "",
        errors=tuple(errors),
    )


def collect_document_readings(
    pages: list[Any],
    client: OpenAI,
    model: str,
    ollama_base_url: str = LLM_BASE_URL,
    default_client: bool = True,
) -> list[PageReadings]:
    return [
        collect_page_readings(
            page=page,
            client=client,
            model=model,
            ollama_base_url=ollama_base_url,
            default_client=default_client,
        )
        for page in pages
    ]