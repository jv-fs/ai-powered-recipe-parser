from __future__ import annotations

import base64
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import Collector, JSONBuilder, LLMMethods, TextVerifier, pipeline  # noqa: E402
from src.extract.Collector import PageReadings  # noqa: E402
from src.extract.ImageManipulation import split_vertical_two_thirds  # noqa: E402


def _png(height: int = 6, width: int = 4) -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _row_png(rows: int = 6, width: int = 5) -> bytes:
    image = np.zeros((rows, width, 3), dtype=np.uint8)
    for row in range(rows):
        image[row, :, :] = (row + 1) * 30
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _decode_png(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _page(
    page_index: int = 0,
    source_index: int = 0,
    text: str | None = None,
    image: bytes | None = None,
) -> Any:
    return SimpleNamespace(
        index=page_index,
        source_index=source_index,
        name=f"page-{page_index}",
        text=text,
        image=image,
    )


class _FakeCompletions:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0) if self.results else '{"ok": true}'
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))]
        )


class _FakeOpenAI:
    def __init__(self, *results: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(list(results)))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


class _BoomClient:
    @property
    def chat(self) -> Any:
        raise AssertionError("client should not be used")


def _raw_step(index: int, text: str, quote: str = "", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "index": index,
        "text": text,
        "quote": quote,
        "duration": None,
        "conditions": [],
        "continues_previous": False,
        "inferred": False,
        "inference_reason": None,
    }
    data.update(overrides)
    return data


def _raw_item(step_index: int, name: str, quote: str = "", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "step_index": step_index,
        "name": name,
        "type": "ingredient",
        "quantity": None,
        "unit": None,
        "preparation": None,
        "optional": False,
        "quote": quote,
        "inferred": False,
        "inference_reason": None,
    }
    data.update(overrides)
    return data


def _raw_servings(
    value: int | None = 4,
    quote: str | None = None,
    inferred: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "quote": quote,
        "inferred": inferred,
        "inference_reason": reason,
    }


def _structure(
    steps: list[dict[str, Any]] | None = None,
    items: list[dict[str, Any]] | None = None,
    servings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "steps": list(steps or []),
        "items": list(items or []),
        "servings": servings if servings is not None else _raw_servings(inferred=True, reason="test"),
    }


def _pages(full_text: str, page: int = 0, source_index: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "page": page,
            "source_index": source_index,
            "text": full_text,
            "origin": "text_layer",
            "char_start": 0,
            "char_end": len(full_text),
            "readings": {},
        }
    ]


def _build(
    monkeypatch: pytest.MonkeyPatch,
    full_text: str,
    raw_data: dict[str, Any],
    pages: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    monkeypatch.setattr(JSONBuilder, "extract_structure", lambda **kwargs: (raw_data, None))
    return JSONBuilder.build_structured_recipe(
        full_text=full_text,
        pages=pages if pages is not None else _pages(full_text),
        client=object(),
        model="test-model",
    )


def _reading(
    page_index: int = 0,
    source_index: int = 0,
    native_text: str | None = None,
    full_text: str = "",
    top_text: str = "",
    bottom_text: str = "",
    errors: tuple[str, ...] = (),
) -> PageReadings:
    return PageReadings(
        page_index=page_index,
        source_index=source_index,
        native_text=native_text,
        full_text=full_text,
        top_text=top_text,
        bottom_text=bottom_text,
        errors=errors,
    )


def _structured() -> dict[str, Any]:
    return {
        "steps": [{"id": "step-001"}],
        "items": [{"id": "item-001"}],
        "servings": {"value": 2},
    }


def test_split_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="Empty image bytes"):
        split_vertical_two_thirds(b"")

    with pytest.raises(ValueError, match="Unable to decode"):
        split_vertical_two_thirds(b"not an image")

    with pytest.raises(ValueError, match="height is too small"):
        split_vertical_two_thirds(_png(height=2))


def test_split_returns_overlapping_top_and_bottom_crops() -> None:
    source = _row_png(rows=6, width=5)

    top, bottom = split_vertical_two_thirds(source)

    original = _decode_png(source)
    top_image = _decode_png(top)
    bottom_image = _decode_png(bottom)

    assert top_image.shape == (4, 5, 3)
    assert bottom_image.shape == (4, 5, 3)
    np.testing.assert_array_equal(top_image, original[:4])
    np.testing.assert_array_equal(bottom_image, original[2:])


def test_is_default_client() -> None:
    default = SimpleNamespace(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    trailing_slash = SimpleNamespace(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
    )
    wrong_key = SimpleNamespace(
        base_url="http://localhost:11434/v1",
        api_key="sk-test",
    )
    wrong_url = SimpleNamespace(base_url="http://example.test/v1", api_key="ollama")

    assert LLMMethods.is_default_client(default, "http://localhost:11434/v1") is True
    assert LLMMethods.is_default_client(trailing_slash, "http://localhost:11434/v1") is True
    assert LLMMethods.is_default_client(wrong_key, "http://localhost:11434/v1") is False
    assert LLMMethods.is_default_client(wrong_url, "http://localhost:11434/v1") is False
    assert LLMMethods.is_default_client(object(), "http://localhost:11434/v1") is False


def test_parse_json_obj() -> None:
    assert LLMMethods._parse_json_obj({"a": 1}) == {"a": 1}
    assert LLMMethods._parse_json_obj('{"a": 1}') == {"a": 1}
    assert LLMMethods._parse_json_obj("[1, 2]") is None
    assert LLMMethods._parse_json_obj("not json") is None
    assert LLMMethods._parse_json_obj(123) is None


def test_call_llm_uses_json_schema_first_then_json_object() -> None:
    schema = {"type": "object"}
    client = _FakeOpenAI(RuntimeError("schema unsupported"), '{"ok": true}')
    messages = [{"role": "user", "content": "hello"}]

    data, err = LLMMethods.call_llm(
        client=client,
        model="test-model",
        messages=messages,
        default_client=False,
        schema=schema,
    )

    assert data == {"ok": True}
    assert err is None
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "strict": True, "schema": schema},
    }
    assert client.calls[1]["response_format"] == {"type": "json_object"}
    for call in client.calls:
        assert call["model"] == "test-model"
        assert call["temperature"] == 0
        assert call["max_tokens"] == 4096
        assert call["extra_body"] == {"options": {"num_ctx": 16384, "num_predict": 4096}}
        assert call["messages"] is messages


def test_call_llm_stops_at_first_valid_response() -> None:
    client = _FakeOpenAI('{"a": 1}', '{"b": 2}')

    data, err = LLMMethods.call_llm(client, "test-model", [], default_client=False)

    assert data == {"a": 1}
    assert err is None
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_call_llm_falls_back_to_native_ollama_for_default_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {"type": "object"}
    native_calls: list[tuple[str, list[dict[str, Any]], str, dict[str, Any] | None]] = []

    def fake_native(
        model: str,
        messages: list[dict[str, Any]],
        ollama_base_url: str,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        native_calls.append((model, messages, ollama_base_url, format_schema))
        if format_schema is not None:
            return "not json"
        return '{"native": true}'

    monkeypatch.setattr(LLMMethods, "_native_ollama_chat", fake_native)
    client = _FakeOpenAI(RuntimeError("schema failed"), RuntimeError("json failed"))

    data, err = LLMMethods.call_llm(
        client=client,
        model="test-model",
        messages=[],
        ollama_base_url="http://ollama.test/v1",
        default_client=True,
        schema=schema,
    )

    assert data == {"native": True}
    assert err is None
    assert len(client.calls) == 2
    assert [call[3] for call in native_calls] == [schema, None]
    assert native_calls[0][0] == "test-model"
    assert native_calls[0][2] == "http://ollama.test/v1"


def test_call_llm_non_default_invalid_json_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_native(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("native fallback must not run")

    monkeypatch.setattr(LLMMethods, "_native_ollama_chat", forbidden_native)
    client = _FakeOpenAI("not json")

    data, err = LLMMethods.call_llm(client, "test-model", [], default_client=False)

    assert data is None
    assert err == "invalid_json"
    assert len(client.calls) == 1


def test_call_llm_reports_last_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_native(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("native fallback must not run")

    monkeypatch.setattr(LLMMethods, "_native_ollama_chat", forbidden_native)
    client = _FakeOpenAI(RuntimeError("boom"), ValueError("bad"))

    data, err = LLMMethods.call_llm(
        client,
        "test-model",
        [],
        default_client=False,
        schema={"type": "object"},
    )

    assert data is None
    assert err == "ValueError: bad"
    assert len(client.calls) == 2


def test_extract_segments_handles_supported_shapes() -> None:
    assert LLMMethods._extract_segments(
        {"segments": ["  a   b ", "", {"text": " c "}, 3]}
    ) == ["a b", "c"]
    assert LLMMethods._extract_segments({"text": "  hi  "}) == ["hi"]
    assert LLMMethods._extract_segments({"transcript": " yo "}) == ["yo"]
    assert LLMMethods._extract_segments({"lines": [" a ", "", "b"]}) == ["a", "b"]
    assert LLMMethods._extract_segments({}) == []


def test_native_ollama_chat_converts_messages_and_calls_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": '{"ok": 1}'}}).encode("utf-8")

    def fake_urlopen(req: Any, timeout: int) -> _Response:
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(LLMMethods.urllib.request, "urlopen", fake_urlopen)
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "ignored"},
            ],
        },
    ]
    schema = {"type": "object"}

    raw = LLMMethods._native_ollama_chat(
        "test-model",
        messages,
        "http://localhost:11434/v1",
        schema,
    )

    assert raw == '{"ok": 1}'
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["timeout"] == 600
    assert seen["body"] == {
        "model": "test-model",
        "stream": False,
        "format": schema,
        "options": {"num_ctx": 16384, "num_predict": 4096, "temperature": 0},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a\nb", "images": ["QUJD"]},
        ],
    }

    def failing_urlopen(req: Any, timeout: int) -> None:
        raise OSError("offline")

    monkeypatch.setattr(LLMMethods.urllib.request, "urlopen", failing_urlopen)
    assert LLMMethods._native_ollama_chat("test-model", [], "http://localhost:11434/v1") is None


def test_transcribe_image_builds_prompt_and_image_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_call_llm(**kwargs: Any) -> tuple[dict[str, Any], str | None]:
        captured.update(kwargs)
        return {"segments": ["  Line   one ", "Line two"]}, None

    monkeypatch.setattr(LLMMethods, "call_llm", fake_call_llm)
    client = object()

    text, err = LLMMethods.transcribe_image(
        client=client,
        model="test-model",
        image_bytes=b"image-bytes",
        ollama_base_url="http://ollama.test/v1",
        default_client=False,
    )

    assert text == "Line one Line two"
    assert err is None
    assert captured["client"] is client
    assert captured["schema"] is LLMMethods.TRANSCRIBE_SCHEMA
    assert captured["default_client"] is False
    assert captured["ollama_base_url"] == "http://ollama.test/v1"
    parts = captured["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": LLMMethods.TRANSCRIBE_PROMPT}
    encoded = base64.b64encode(b"image-bytes").decode("ascii")
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }

    monkeypatch.setattr(LLMMethods, "call_llm", lambda **kwargs: (None, "boom"))
    assert LLMMethods.transcribe_image(object(), "test-model", b"x") == (None, "boom")


def test_extract_structure_fences_source_and_forwards_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_call_llm(**kwargs: Any) -> tuple[dict[str, Any], str | None]:
        captured.update(kwargs)
        return {"steps": []}, None

    monkeypatch.setattr(LLMMethods, "call_llm", fake_call_llm)
    client = object()

    data, err = LLMMethods.extract_structure(
        client=client,
        model="test-model",
        full_text="Some text",
        ollama_base_url="http://ollama.test/v1",
        default_client=True,
    )

    assert data == {"steps": []}
    assert err is None
    assert captured["client"] is client
    assert captured["schema"] is LLMMethods.STRUCTURE_SCHEMA
    assert captured["default_client"] is True
    parts = captured["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": LLMMethods.STRUCTURE_PROMPT}
    assert parts[1] == {
        "type": "text",
        "text": "<source>\n```\nSome text\n```\n</source>",
    }


def test_page_readings_is_frozen() -> None:
    reading = _reading()

    with pytest.raises(FrozenInstanceError):
        reading.full_text = "changed"


def test_native_text_short_circuits_client() -> None:
    reading = Collector.collect_page_readings(
        page=_page(page_index=2, source_index=7, text="  A   B \n C ", image=b"unused"),
        client=_BoomClient(),
        model="test-model",
    )

    assert reading.page_index == 2
    assert reading.source_index == 7
    assert reading.native_text == "A B C"
    assert reading.full_text == "A B C"
    assert reading.top_text == ""
    assert reading.bottom_text == ""
    assert reading.errors == ()


def test_missing_image_warns_without_client() -> None:
    reading = Collector.collect_page_readings(
        page=_page(text="   ", image=None),
        client=_BoomClient(),
        model="test-model",
    )

    assert reading.native_text is None
    assert reading.full_text == ""
    assert reading.errors == ("missing_page_image",)


def test_image_page_runs_three_transcriptions_and_captures_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_bytes = b"FULL"
    top_bytes = b"TOP"
    bottom_bytes = b"BOTTOM"
    calls: list[bytes] = []

    monkeypatch.setattr(
        Collector,
        "split_vertical_two_thirds",
        lambda data: (top_bytes, bottom_bytes),
    )

    def fake_transcribe(
        *,
        client: Any,
        model: str,
        image_bytes: bytes,
        ollama_base_url: str,
        default_client: bool,
    ) -> tuple[str | None, str | None]:
        calls.append(image_bytes)
        if image_bytes == full_bytes:
            return "Full text", "full_err"
        if image_bytes == top_bytes:
            return None, "top_err"
        return "Bottom text", None

    monkeypatch.setattr(Collector, "transcribe_image", fake_transcribe)

    reading = Collector.collect_page_readings(
        page=_page(page_index=1, source_index=3, image=full_bytes),
        client=object(),
        model="test-model",
        ollama_base_url="http://ollama.test/v1",
        default_client=True,
    )

    assert calls == [full_bytes, top_bytes, bottom_bytes]
    assert reading.native_text is None
    assert reading.full_text == "Full text"
    assert reading.top_text == ""
    assert reading.bottom_text == "Bottom text"
    assert reading.errors == (
        "full_transcript_error: full_err",
        "top_transcript_error: top_err",
    )


def test_crop_split_failure_keeps_full_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_split(data: bytes) -> tuple[bytes, bytes]:
        raise ValueError("bad crop")

    monkeypatch.setattr(Collector, "split_vertical_two_thirds", failing_split)
    calls: list[bytes] = []

    def fake_transcribe(
        *,
        client: Any,
        model: str,
        image_bytes: bytes,
        ollama_base_url: str,
        default_client: bool,
    ) -> tuple[str | None, str | None]:
        calls.append(image_bytes)
        return "Full text", None

    monkeypatch.setattr(Collector, "transcribe_image", fake_transcribe)

    reading = Collector.collect_page_readings(
        page=_page(image=b"FULL"),
        client=object(),
        model="test-model",
    )

    assert calls == [b"FULL"]
    assert reading.full_text == "Full text"
    assert reading.top_text == ""
    assert reading.bottom_text == ""
    assert reading.errors == ("crop_split_error: bad crop",)


def test_collect_document_readings_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def fake_collect(**kwargs: Any) -> PageReadings:
        page = kwargs["page"]
        seen.append(page)
        return _reading(page_index=page.index, source_index=page.source_index)

    monkeypatch.setattr(Collector, "collect_page_readings", fake_collect)
    pages = [_page(page_index=0), _page(page_index=1)]

    readings = Collector.collect_document_readings(
        pages=pages,
        client=object(),
        model="test-model",
        ollama_base_url="http://ollama.test/v1",
        default_client=False,
    )

    assert seen == pages
    assert [reading.page_index for reading in readings] == [0, 1]


def test_merge_crops_empty_overlap_and_concat() -> None:
    assert TextVerifier.merge_crops("", "  c d  ") == "c d"
    assert TextVerifier.merge_crops("  a b ", "") == "a b"
    assert TextVerifier.merge_crops("a b c d e", "c d e f g") == "a b c d e f g"
    assert TextVerifier.merge_crops("a b", "c d") == "a b c d"


def test_verify_transcript_empty_inputs() -> None:
    assert TextVerifier.verify_transcript("", "", "") == ("", [])

    text, warnings = TextVerifier.verify_transcript("", "a b", "", page_index=2)
    assert text == "a b"
    assert warnings[0]["code"] == "full_transcript_empty"
    assert warnings[0]["page"] == 2
    assert warnings[0]["crop_reading"] == "a b"

    text, warnings = TextVerifier.verify_transcript("  full  text ", "", "", page_index=3)
    assert text == "full  text"
    assert warnings[0]["code"] == "crop_transcript_empty"
    assert warnings[0]["page"] == 3
    assert warnings[0]["full_reading"] == "full  text"


def test_verify_transcript_equal_reading_has_no_warnings() -> None:
    text, warnings = TextVerifier.verify_transcript(
        full_text="one two three",
        top_text="one two three",
        bottom_text="",
    )

    assert text == "one two three"
    assert warnings == []


def test_verify_transcript_refines_minor_discrepancy() -> None:
    text, warnings = TextVerifier.verify_transcript(
        full_text="a butter c",
        top_text="a butters c",
        bottom_text="",
        page_index=1,
    )

    assert text == "a butters c"
    assert len(warnings) == 1
    assert warnings[0]["code"] == "crop_refined"
    assert warnings[0]["severity"] == "info"
    assert warnings[0]["page"] == 1
    assert warnings[0]["full_reading"] == "butter"
    assert warnings[0]["crop_reading"] == "butters"


def test_verify_transcript_rejects_major_discrepancy() -> None:
    text, warnings = TextVerifier.verify_transcript(
        full_text="a butter c",
        top_text="a margarine c",
        bottom_text="",
        page_index=4,
    )

    assert text == "a butter c"
    assert len(warnings) == 1
    assert warnings[0]["code"] == "verification_failed"
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["page"] == 4
    assert warnings[0]["full_reading"] == "butter"
    assert warnings[0]["crop_reading"] == "margarine"


def test_verify_transcript_reports_delete_and_insert() -> None:
    text, warnings = TextVerifier.verify_transcript(
        full_text="a b c",
        top_text="a c",
        bottom_text="",
        page_index=5,
    )
    assert text == "a b c"
    assert [warning["code"] for warning in warnings] == ["missing_in_crop"]
    assert warnings[0]["full_reading"] == "b"
    assert warnings[0]["crop_reading"] == ""

    text, warnings = TextVerifier.verify_transcript(
        full_text="a c",
        top_text="a b c",
        bottom_text="",
        page_index=6,
    )
    assert text == "a b c"
    assert [warning["code"] for warning in warnings] == ["recovered_by_crop"]
    assert warnings[0]["full_reading"] == ""
    assert warnings[0]["crop_reading"] == "b"


def test_json_builder_helpers() -> None:
    assert JSONBuilder._norm("  a  \n b ") == "a b"
    assert JSONBuilder._canonical(" ½ 1,5 ") == "1/2 1.5"
    assert JSONBuilder._as_str(" null ") is None
    assert JSONBuilder._as_str(7) == "7"
    assert JSONBuilder._as_bool("YES") is True
    assert JSONBuilder._as_bool("no") is False
    assert JSONBuilder._as_int(True) is None
    assert JSONBuilder._as_int(" 4 ") == 4
    assert JSONBuilder._as_int("x") is None
    assert JSONBuilder._find_span("Ich nehme ½ Liter.", "½ Liter") == (10, 17)
    assert JSONBuilder._find_span("a  b", "a b") == (0, 4)
    assert JSONBuilder._find_span("nothing here", "missing") is None

    pages = [
        {"char_start": 0, "char_end": 3, "page": 10},
        {"char_start": 4, "char_end": 7, "page": 11},
    ]
    assert JSONBuilder._page_for_offset(pages, 0) == 10
    assert JSONBuilder._page_for_offset(pages, 3) == 11
    assert JSONBuilder._page_for_offset([], 0) == 0


def test_structure_failure_returns_fallback_and_error_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(JSONBuilder, "extract_structure", lambda **kwargs: (None, "boom"))

    structured, warnings = JSONBuilder.build_structured_recipe(
        full_text="some text",
        pages=[],
        client=object(),
        model="test-model",
    )

    assert structured["steps"] == []
    assert structured["items"] == []
    assert structured["servings"] == {
        "id": "serving-001",
        "value": None,
        "inferred": True,
        "inference_reason": "Structuring failed",
        "quote": None,
    }
    assert warnings == [
        {
            "code": "structure_extraction_failed",
            "severity": "error",
            "message": "Failed to structure recipe text: boom",
            "page": None,
        }
    ]


def test_steps_sorted_validated_and_ids_assigned(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "First step. Second step."
    raw = _structure(
        steps=[
            _raw_step(2, "Second step.", "Second step."),
            _raw_step(1, "First step.", "First step.", continues_previous=True),
        ]
    )

    structured, warnings = _build(monkeypatch, full_text, raw)
    steps = structured["steps"]

    assert [step["id"] for step in steps] == ["step-001", "step-002"]
    assert [step["index"] for step in steps] == [0, 1]
    assert [step["text"] for step in steps] == ["First step.", "Second step."]
    assert steps[0]["span"] == {"start": 0, "end": 11}
    assert steps[1]["span"] == {"start": 12, "end": 24}
    assert steps[0]["continues_previous"] is False
    assert any(
        warning["code"] == "invalid_continues_previous"
        and warning["step_id"] == "step-001"
        for warning in warnings
    )


def test_explicit_step_quote_not_found_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "Real step."
    raw = _structure(steps=[_raw_step(0, "Real step.", "not in text")])

    structured, warnings = _build(monkeypatch, full_text, raw)

    assert structured["steps"] == []
    assert any(
        warning["code"] == "quote_not_found" and warning["quote"] == "not in text"
        for warning in warnings
    )


def test_inferred_step_and_invalid_fields_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_text = "Boil 10 min until tender."
    raw = _structure(
        steps=[
            _raw_step(
                0,
                full_text,
                "bogus quote",
                inferred=True,
                duration="20 min",
                conditions=["until golden", "until tender"],
                continues_previous=True,
            )
        ]
    )

    structured, warnings = _build(monkeypatch, full_text, raw)
    step = structured["steps"][0]
    codes = [warning["code"] for warning in warnings]

    assert step["quote"] == ""
    assert step["span"] is None
    assert step["inference_reason"] == "Inferred step without explicit reason"
    assert step["duration"] is None
    assert step["conditions"] == ["until tender"]
    assert step["continues_previous"] is False
    assert "inference_without_reason" in codes
    assert "inferred_quote_cleared" in codes
    assert "duration_not_in_step" in codes
    assert "condition_not_in_step" in codes
    assert "invalid_continues_previous" in codes


def test_unmapped_text_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "Alpha. skipped words. Beta."
    raw = _structure(
        steps=[
            _raw_step(0, "Alpha.", "Alpha."),
            _raw_step(1, "Beta.", "Beta."),
        ]
    )

    _, warnings = _build(monkeypatch, full_text, raw)
    unmapped = [warning for warning in warnings if warning["code"] == "unmapped_text"]

    assert len(unmapped) == 1
    assert unmapped[0]["quote"] == "skipped words."
    assert unmapped[0]["span"] == {"start": 7, "end": 21}


def test_items_are_resolved_deduped_and_linked(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "Chop 2 onions in a pan. Add salt."
    raw = _structure(
        steps=[
            _raw_step(0, "Chop 2 onions in a pan.", "Chop 2 onions in a pan."),
            _raw_step(1, "Add salt.", "Add salt."),
        ],
        items=[
            _raw_item(0, "onion", "2 onions", quantity="2"),
            _raw_item(0, "onion", "2 onions", quantity="2"),
            _raw_item(0, "pan", "a pan", type="tool"),
            _raw_item(1, "salt", "salt"),
            _raw_item(0, "oil", "bogus quote", inferred=True),
            _raw_item(0, "ghost", "not in text"),
        ],
    )

    structured, warnings = _build(monkeypatch, full_text, raw)
    items = structured["items"]

    assert [item["id"] for item in items] == [
        "item-001",
        "item-002",
        "item-003",
        "item-004",
    ]
    assert [item["step_id"] for item in items] == [
        "step-001",
        "step-001",
        "step-002",
        "step-001",
    ]
    assert [item["type"] for item in items] == [
        "ingredient",
        "tool",
        "ingredient",
        "ingredient",
    ]
    assert items[0]["span"] == {"start": 5, "end": 13}
    assert items[1]["span"] == {"start": 17, "end": 22}
    assert items[2]["span"] == {"start": 28, "end": 32}
    assert items[3]["quote"] == ""
    assert items[3]["span"] is None
    assert items[3]["inference_reason"] == "Implied by recipe step"
    assert structured["steps"][0]["item_ids"] == [
        "item-001",
        "item-002",
        "item-004",
    ]
    assert structured["steps"][1]["item_ids"] == ["item-003"]
    assert any(warning["code"] == "duplicate_merged" for warning in warnings)
    assert any(
        warning["code"] == "quote_not_found" and warning["quote"] == "not in text"
        for warning in warnings
    )


def test_item_quote_outside_step_scope_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "Use oil. Add oil."
    raw = _structure(
        steps=[
            _raw_step(0, "Use oil.", "Use oil."),
            _raw_step(1, "Add oil.", "Add oil."),
        ],
        items=[_raw_item(0, "oil", "Add oil.")],
    )

    structured, warnings = _build(monkeypatch, full_text, raw)

    assert structured["items"][0]["span"] == {"start": 9, "end": 17}
    assert any(
        warning["code"] == "step_scope_mismatch"
        and warning["step_id"] == "step-001"
        for warning in warnings
    )


def test_orphan_item_dropped_when_no_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _structure(items=[_raw_item(0, "salt", "salt")])

    structured, warnings = _build(monkeypatch, "salt", raw)

    assert structured["items"] == []
    assert any(warning["code"] == "orphan_item_dropped" for warning in warnings)


def test_servings_explicit_quote_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    full_text = "Serves 4."
    raw = _structure(
        servings=_raw_servings(value=4, quote="Serves 4", inferred=False, reason=None)
    )

    structured, warnings = _build(monkeypatch, full_text, raw)

    assert structured["servings"] == {
        "id": "serving-001",
        "value": 4,
        "inferred": False,
        "inference_reason": None,
        "quote": "Serves 4",
        "span": {"start": 0, "end": 8},
    }
    assert not any(warning["code"] == "servings_inferred" for warning in warnings)


def test_servings_bad_quote_resets_to_inferred(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _structure(
        servings=_raw_servings(value=4, quote="Serves 4", inferred=False, reason=None)
    )

    structured, warnings = _build(monkeypatch, "Boil water.", raw)
    servings = structured["servings"]
    codes = [warning["code"] for warning in warnings]

    assert servings["value"] is None
    assert servings["inferred"] is True
    assert servings["inference_reason"] == "Servings quote not found in text"
    assert servings["quote"] is None
    assert servings["span"] is None
    assert "quote_not_found" in codes
    assert "servings_inferred" in codes


def test_servings_inferred_value_clears_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _structure(
        servings=_raw_servings(value=4, quote="Serves 4", inferred=True, reason=None)
    )

    structured, warnings = _build(monkeypatch, "Serves 4.", raw)

    assert structured["servings"]["value"] == 4
    assert structured["servings"]["inferred"] is True
    assert structured["servings"]["quote"] is None
    assert structured["servings"]["span"] is None
    assert structured["servings"]["inference_reason"] == (
        "Servings inferred from recipe proportions"
    )
    assert any(warning["code"] == "servings_inferred" for warning in warnings)


def test_extract_uses_native_text_and_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = (_page(page_index=0, source_index=10), _page(page_index=1, source_index=11))
    readings = [
        _reading(
            page_index=0,
            source_index=10,
            native_text="AAA",
            full_text="ignored",
            errors=("native_err",),
        ),
        _reading(page_index=1, source_index=11, native_text="BBB"),
    ]
    collect_seen: dict[str, Any] = {}
    build_seen: dict[str, Any] = {}

    def fake_collect(**kwargs: Any) -> list[PageReadings]:
        collect_seen.update(kwargs)
        return readings

    def forbidden_verify(**kwargs: Any) -> None:
        raise AssertionError("native pages must not be verified")

    def fake_build(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        build_seen.update(kwargs)
        return _structured(), [{"code": "struct_warning"}]

    monkeypatch.setattr(pipeline, "collect_document_readings", fake_collect)
    monkeypatch.setattr(pipeline, "verify_transcript", forbidden_verify)
    monkeypatch.setattr(pipeline, "build_structured_recipe", fake_build)
    client = object()

    extracted = pipeline.extract(
        SimpleNamespace(pages=pages),
        model="test-model",
        client=client,
        ollama_base_url="http://ollama.test/v1",
    )

    assert collect_seen["pages"] is pages
    assert collect_seen["client"] is client
    assert collect_seen["model"] == "test-model"
    assert collect_seen["ollama_base_url"] == "http://ollama.test/v1"
    assert extracted["text"] == "AAA\nBBB"
    assert [page["origin"] for page in extracted["pages"]] == ["text_layer", "text_layer"]
    assert extracted["pages"][0]["char_start"] == 0
    assert extracted["pages"][0]["char_end"] == 3
    assert extracted["pages"][1]["char_start"] == 4
    assert extracted["pages"][1]["char_end"] == 7
    assert extracted["pages"][0]["readings"]["native_text"] == "AAA"
    assert extracted["steps"] == [{"id": "step-001"}]
    assert extracted["items"] == [{"id": "item-001"}]
    assert extracted["servings"] == {"value": 2}
    assert extracted["model"] == "test-model"
    assert extracted["warnings"] == [
        {
            "code": "collection_error",
            "severity": "warning",
            "message": "native_err",
            "page": 0,
        },
        {"code": "struct_warning"},
    ]
    assert build_seen["full_text"] == "AAA\nBBB"
    assert build_seen["pages"] is extracted["pages"]
    assert build_seen["client"] is client


def test_extract_verifies_image_pages_and_warning_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading = _reading(
        page_index=2,
        source_index=5,
        full_text="full",
        top_text="top",
        bottom_text="bottom",
        errors=("missing",),
    )
    verify_seen: dict[str, Any] = {}

    def fake_verify(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        verify_seen.update(kwargs)
        return "verified text", [{"code": "crop_refined"}]

    monkeypatch.setattr(pipeline, "collect_document_readings", lambda **kwargs: [reading])
    monkeypatch.setattr(pipeline, "verify_transcript", fake_verify)
    monkeypatch.setattr(
        pipeline,
        "build_structured_recipe",
        lambda **kwargs: (_structured(), [{"code": "struct_warning"}]),
    )

    extracted = pipeline.extract(SimpleNamespace(pages=(_page(),)), client=object())

    assert verify_seen == {
        "full_text": "full",
        "top_text": "top",
        "bottom_text": "bottom",
        "page_index": 2,
    }
    assert extracted["text"] == "verified text"
    assert extracted["pages"][0]["origin"] == "llm_verified"
    assert extracted["pages"][0]["text"] == "verified text"
    assert extracted["pages"][0]["readings"] == {
        "native_text": None,
        "full_text": "full",
        "top_text": "top",
        "bottom_text": "bottom",
    }
    assert extracted["warnings"] == [
        {
            "code": "collection_error",
            "severity": "warning",
            "message": "missing",
            "page": 2,
        },
        {"code": "crop_refined"},
        {"code": "struct_warning"},
    ]


def test_extract_default_client_creation_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, Any]] = []
    factory_client = object()

    def fake_openai(**kwargs: Any) -> object:
        created.append(kwargs)
        return factory_client

    default_calls: list[tuple[Any, str]] = []

    def fake_is_default(client: Any, url: str) -> bool:
        default_calls.append((client, url))
        return False

    collect_seen: dict[str, Any] = {}
    build_seen: dict[str, Any] = {}

    def fake_collect(**kwargs: Any) -> list[PageReadings]:
        collect_seen.clear()
        collect_seen.update(kwargs)
        return []

    def fake_build(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        build_seen.clear()
        build_seen.update(kwargs)
        return {"steps": [], "items": [], "servings": {"value": None}}, []

    monkeypatch.setattr(pipeline, "OpenAI", fake_openai)
    monkeypatch.setattr(pipeline, "is_default_client", fake_is_default)
    monkeypatch.setattr(pipeline, "collect_document_readings", fake_collect)
    monkeypatch.setattr(pipeline, "build_structured_recipe", fake_build)

    extracted = pipeline.extract(
        SimpleNamespace(pages=[]),
        ollama_base_url="http://ollama.test/v1",
    )

    assert created == [
        {"base_url": "http://ollama.test/v1", "api_key": "ollama"},
    ]
    assert default_calls == []
    assert collect_seen["pages"] == []
    assert collect_seen["client"] is factory_client
    assert collect_seen["default_client"] is True
    assert build_seen["client"] is factory_client
    assert build_seen["default_client"] is True
    assert extracted["text"] == ""
    assert extracted["pages"] == []

    provided_client = object()
    monkeypatch.setattr(pipeline, "is_default_client", lambda client, url: True)

    pipeline.extract(
        SimpleNamespace(pages=[]),
        client=provided_client,
        ollama_base_url="http://other.test/v1",
    )

    assert created == [
        {"base_url": "http://ollama.test/v1", "api_key": "ollama"},
    ]
    assert collect_seen["client"] is provided_client
    assert collect_seen["default_client"] is True
    assert build_seen["client"] is provided_client
    assert build_seen["default_client"] is True
