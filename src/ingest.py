from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
import pymupdf

Kind = Literal["pdf", "image", "text"]

MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_PAGES = 100
MAX_LONG_EDGE = 2000

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceInfo:
    kind: Kind
    name: str | None
    mime: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Warning:
    code: str
    message: str
    page: int | None = None


@dataclass(frozen=True)
class Page:
    index: int
    source_index: int
    name: str | None = None
    image: bytes | None = None
    text: str | None = None
    raw_text: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    width: int | None = None
    height: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class IngestResult:
    sources: tuple[SourceInfo, ...]
    pages: tuple[Page, ...]
    text: str
    warnings: tuple[Warning, ...]


def ingest(
    source: str | Path | bytes | Sequence[Path | bytes],
    *,
    name: str | None = None,
) -> IngestResult:
    inputs = [source] if isinstance(source, (str, bytes, Path)) else list(source)
    sources: list[SourceInfo] = []
    pages: list[Page] = []
    warnings: list[Warning] = []

    for item in inputs:
        source_name = name if len(inputs) == 1 else None
        _consume(item, source_name, sources, pages, warnings)

    finalized, text = _with_offsets(pages)
    return IngestResult(tuple(sources), finalized, text, tuple(warnings))


def _consume(
    item: object,
    name: str | None,
    sources: list[SourceInfo],
    pages: list[Page],
    warnings: list[Warning],
) -> None:
    data, derived_name = _read(item, name)

    if data is None:
        warnings.append(Warning("unsupported_input", "Unsupported input type"))
        return
    if not data:
        warnings.append(Warning("empty_input", "Input is empty"))
        return
    if len(data) > MAX_INPUT_BYTES:
        warnings.append(Warning("input_too_large", f"Input exceeds {MAX_INPUT_BYTES} bytes"))
        return

    if data.startswith(b"%PDF") or _has_suffix(derived_name, ".pdf"):
        _consume_pdf(data, derived_name, sources, pages, warnings, MAX_PAGES - len(pages))
        return
    if _has_suffix(derived_name, ".txt"):
        decoded_text = _decode_text(data)
        if decoded_text is None:
            warnings.append(Warning("decode_failed", "Could not decode text file"))
            return
        _consume_text(data, derived_name, sources, pages, warnings, decoded_text)
        return
    if _image_mime(data) is not None:
        _consume_image(data, derived_name, sources, pages, warnings)
        return

    decoded = _decode_text(data)
    if decoded is not None:
        _consume_text(data, derived_name, sources, pages, warnings, decoded)
        return

    warnings.append(Warning("unsupported_input", "Input is neither PDF, image nor text"))


def _read(item: object, name: str | None) -> tuple[bytes | None, str | None]:
    if isinstance(item, str):
        return item.encode("utf-8"), name
    if isinstance(item, Path):
        try:
            return item.read_bytes(), name or item.name
        except OSError:
            return None, name or item.name
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), name
    return None, name


def _consume_text(
    data: bytes,
    name: str | None,
    sources: list[SourceInfo],
    pages: list[Page],
    warnings: list[Warning],
    text: str,
) -> None:
    source_index = len(sources)
    sources.append(_source("text", name, "text/plain", data))
    pages.append(
        Page(
            index=len(pages),
            source_index=source_index,
            name=name,
            text=_normalize(text) or None,
        )
    )


def _consume_image(
    data: bytes,
    name: str | None,
    sources: list[SourceInfo],
    pages: list[Page],
    warnings: list[Warning],
) -> None:
    mime = _image_mime(data) or "application/octet-stream"
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        warnings.append(Warning("decode_failed", "Could not decode image"))
        return

    height, width = array.shape[:2]
    scale = min(1.0, MAX_LONG_EDGE / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        warnings.append(Warning("decode_failed", "Could not encode image as PNG"))
        return
    image = encoded.tobytes()
    height, width = array.shape[:2]

    source_index = len(sources)
    sources.append(_source("image", name, mime, data))
    page_index = len(pages)
    if scale < 1.0:
        warnings.append(
            Warning("resized", f"Image downscaled to long edge {MAX_LONG_EDGE}", page_index)
        )
    pages.append(
        Page(
            index=page_index,
            source_index=source_index,
            name=name,
            image=image,
            width=width,
            height=height,
            sha256=hashlib.sha256(image).hexdigest(),
        )
    )


def _consume_pdf(
    data: bytes,
    name: str | None,
    sources: list[SourceInfo],
    pages: list[Page],
    warnings: list[Warning],
    page_budget: int,
) -> None:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        warnings.append(Warning("unsupported_input", "Could not open PDF"))
        return

    with document:
        if document.needs_pass:
            warnings.append(Warning("encrypted_pdf", "PDF is encrypted"))
            return
        if document.page_count == 0:
            warnings.append(Warning("empty_input", "PDF has no pages"))
            return
        if document.page_count > page_budget:
            warnings.append(Warning("input_too_large", f"PDF exceeds the {MAX_PAGES}-page run limit"))
            return

        source_index = len(sources)
        sources.append(_source("pdf", name, "application/pdf", data))

        for document_page in document:
            page_index = len(pages)
            try:
                raw_text = document_page.get_text("text", sort=True)
                rect = document_page.rect
                scale = min(1.0, MAX_LONG_EDGE / max(rect.width, rect.height))
                pixmap = document_page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale),
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                image = pixmap.tobytes("png")
            except Exception:
                warnings.append(Warning("page_unreadable", "Could not read page", page_index))
                continue

            if scale < 1.0:
                warnings.append(
                    Warning("resized", f"Page downscaled to long edge {MAX_LONG_EDGE}", page_index)
                )
            pages.append(
                Page(
                    index=page_index,
                    source_index=source_index,
                    name=name,
                    image=image,
                    text=_normalize(raw_text) or None,
                    raw_text=raw_text or None,
                    width=pixmap.width,
                    height=pixmap.height,
                    sha256=hashlib.sha256(image).hexdigest(),
                )
            )


def _with_offsets(pages: list[Page]) -> tuple[tuple[Page, ...], str]:
    parts: list[str] = []
    spans: list[tuple[int | None, int | None]] = []
    cursor = 0

    for page in pages:
        if page.text:
            spans.append((cursor, cursor + len(page.text)))
            parts.append(page.text)
            cursor += len(page.text) + 1
        else:
            spans.append((None, None))

    finalized = tuple(
        replace(page, char_start=start, char_end=end)
        for page, (start, end) in zip(pages, spans)
    )
    return finalized, "\n".join(parts)


def _source(kind: Kind, name: str | None, mime: str, data: bytes) -> SourceInfo:
    return SourceInfo(
        kind=kind,
        name=name,
        mime=mime,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _decode_text(data: bytes) -> str | None:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not text.strip():
            return None
        printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
        if printable / len(text) >= 0.95:
            return text
    return None


def _has_suffix(name: str | None, suffix: str) -> bool:
    return bool(name and name.lower().endswith(suffix))


def _image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
