from __future__ import annotations

import struct
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ingest as ingest_module  # noqa: E402
from src.ingest import ingest  # noqa: E402


def _png(color: tuple[int, int, int] = (0, 0, 255), size: tuple[int, int] = (40, 20)) -> bytes:
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _image_file(tmp_path: Path, filename: str, color=(0, 0, 255), size=(40, 20)) -> Path:
    path = tmp_path / filename
    path.write_bytes(_png(color, size))
    return path


def _pdf_file(tmp_path: Path, filename: str, page_count: int = 1) -> Path:
    document = pymupdf.open()
    for index in range(page_count):
        page = document.new_page(width=300, height=200)
        page.insert_text((40, 50 + index * 10), f"Line {index}", fontsize=12)
    path = tmp_path / filename
    document.save(str(path))
    document.close()
    return path


def _mixed_pdf_file(tmp_path: Path) -> Path:
    document = pymupdf.open()
    first = document.new_page(width=300, height=200)
    first.insert_text((40, 100), "Zweite Zeile", fontsize=12)
    first.insert_text((40, 50), "Erste Zeile", fontsize=12)
    second = document.new_page(width=300, height=200)
    second.insert_image(pymupdf.Rect(20, 20, 280, 180), stream=_png())
    path = tmp_path / "mixed.pdf"
    document.save(str(path))
    document.close()
    return path


def _exif_jpeg(orientation: int, size: tuple[int, int] = (40, 20)) -> bytes:
    image = np.full((size[1], size[0], 3), (0, 0, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()

    tiff = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    tiff += struct.pack("<H", 1)
    tiff += struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<H", orientation) + b"\x00\x00"
    tiff += struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:2] + app1 + jpeg[2:]


def test_text_input_is_single_page() -> None:
    result = ingest("  Sauerbraten\n\n mit  Knoblauch  ")

    assert result.sources[0].kind == "text"
    assert result.sources[0].mime == "text/plain"
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.image is None
    assert page.raw_text is None
    assert page.text == "Sauerbraten mit Knoblauch"
    assert result.text == page.text
    assert result.text[page.char_start : page.char_end] == page.text


def test_text_file_uses_cp1252_fallback(tmp_path: Path) -> None:
    path = tmp_path / "rezepte.txt"
    path.write_bytes("Großmutter Übung".encode("cp1252"))

    result = ingest(path)

    assert result.text == "Großmutter Übung"
    assert result.sources[0].name == "rezepte.txt"


def test_pdf_emits_both_channels_on_every_page(tmp_path: Path) -> None:
    result = ingest(_mixed_pdf_file(tmp_path))

    assert result.sources[0].kind == "pdf"
    assert [page.index for page in result.pages] == [0, 1]
    assert all(page.image and page.image.startswith(b"\x89PNG") for page in result.pages)
    assert all(max(page.width, page.height) <= 2000 for page in result.pages)

    text_page, image_page = result.pages
    assert text_page.text and text_page.text.startswith("Erste Zeile Zweite Zeile")
    assert text_page.raw_text and "Erste Zeile" in text_page.raw_text
    assert image_page.text is None
    assert image_page.raw_text is None
    assert result.text[text_page.char_start : text_page.char_end] == text_page.text
    assert image_page.char_start is None and image_page.char_end is None


def test_pdf_render_is_byte_identical(tmp_path: Path) -> None:
    data = _pdf_file(tmp_path, "deterministic.pdf").read_bytes()

    first = ingest(data)
    second = ingest(data)

    assert first == second
    assert first.pages[0].sha256 == second.pages[0].sha256


def test_image_inputs_keep_order_and_rgb_colors(tmp_path: Path) -> None:
    first = _image_file(tmp_path, "a.png", color=(0, 0, 255), size=(40, 20))
    second = _image_file(tmp_path, "b.png", color=(255, 0, 0), size=(20, 40))

    result = ingest([first, second])

    assert [page.name for page in result.pages] == ["a.png", "b.png"]
    assert [page.source_index for page in result.pages] == [0, 1]
    assert all(page.text is None and page.raw_text is None for page in result.pages)
    assert all(page.image.startswith(b"\x89PNG") for page in result.pages)

    decoded = cv2.imdecode(np.frombuffer(result.pages[0].image, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    assert tuple(rgb[0, 0]) == (255, 0, 0)


def test_exif_orientation_is_applied() -> None:
    result = ingest(_exif_jpeg(orientation=6))

    assert result.pages[0].width == 20
    assert result.pages[0].height == 40


def test_small_images_are_not_upscaled(tmp_path: Path) -> None:
    result = ingest(_image_file(tmp_path, "small.png", size=(10, 8)))

    assert (result.pages[0].width, result.pages[0].height) == (10, 8)


def test_encrypted_pdf_is_skipped_with_warning(tmp_path: Path) -> None:
    document = pymupdf.open()
    document.new_page().insert_text((40, 50), "secret", fontsize=12)
    path = tmp_path / "locked.pdf"
    document.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()

    result = ingest(path)

    assert result.pages == ()
    assert any(warning.code == "encrypted_pdf" for warning in result.warnings)


def test_empty_and_unsupported_inputs_warn_without_raising() -> None:
    assert ingest(b"").warnings[0].code == "empty_input"
    assert ingest(b"\x00\x01\x02\x03\xff\xfe").warnings[0].code == "unsupported_input"
    assert ingest(b"\x00\x01", name="broken.txt").warnings[0].code == "decode_failed"


def test_bad_source_does_not_discard_good_ones(tmp_path: Path) -> None:
    good = _image_file(tmp_path, "good.png")

    result = ingest([good, b"\x00\x01\x02", good])

    assert [page.index for page in result.pages] == [0, 1]
    assert any(warning.code == "unsupported_input" for warning in result.warnings)


def test_limits_are_configurable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_max_bytes = ingest_module.MAX_INPUT_BYTES
    monkeypatch.setattr(ingest_module, "MAX_INPUT_BYTES", 10)
    assert ingest(b"a" * 20).warnings[0].code == "input_too_large"

    monkeypatch.setattr(ingest_module, "MAX_INPUT_BYTES", original_max_bytes)
    monkeypatch.setattr(ingest_module, "MAX_PAGES", 1)
    result = ingest(_pdf_file(tmp_path, "long.pdf", page_count=2))
    assert any(warning.code == "input_too_large" for warning in result.warnings)

    single_page = _pdf_file(tmp_path, "single.pdf", page_count=1)
    cumulative = ingest([single_page, single_page])
    assert len(cumulative.pages) == 1
    assert any(warning.code == "input_too_large" for warning in cumulative.warnings)


def test_result_is_frozen_and_writes_nothing_to_disk(tmp_path: Path) -> None:
    result = ingest(b"plain bytes")

    with pytest.raises(FrozenInstanceError):
        result.text = "changed"

    assert list(tmp_path.iterdir()) == []
