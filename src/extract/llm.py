from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any

from openai import OpenAI

from ..ingest import Page
from .prompts import _CROP_PROMPT, _CROP_SCHEMA, _TRANSCRIPT_PROMPT, _TRANSCRIPT_SCHEMA
from .text import _json_loads, _norm


def _b64img(image: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")},
    }


def _native_completion(
    model: str,
    messages: list[dict[str, Any]],
    ollama_base_url: str,
    schema: dict[str, Any] | None = None,
) -> str | None:
    native_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            native_messages.append({"role": message.get("role", "user"), "content": content})
            continue
        text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        images = []
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if "," in url:
                images.append(url.split(",", 1)[1])
        native_messages.append(
            {"role": message.get("role", "user"), "content": text, "images": images}
        )
    body = {
        "model": model,
        "stream": False,
        "format": schema if schema is not None else "json",
        "options": {"num_ctx": 16384, "num_predict": 4096, "temperature": 0},
        "messages": native_messages,
    }
    url = ollama_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        request = urllib.request.Request(
            url + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        return data["message"]["content"]
    except Exception:
        return None


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    ollama_base_url: str,
    default_client: bool,
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    content: str | None = None
    last_err: str | None = None
    formats: list[dict[str, Any]] = []
    if schema is not None:
        formats.append(
            {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            }
        )
    formats.append({"type": "json_object"})

    for fmt in formats:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                extra_body={"options": {"num_ctx": 16384, "num_predict": 4096}},
                response_format=fmt,
                messages=messages,
            )
            content = resp.choices[0].message.content
            last_err = None
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    data = _json_loads(content)
    if data is None and default_client:
        native_content = _native_completion(model, messages, ollama_base_url, schema)
        data = _json_loads(native_content)
        if data is None and schema is not None:
            data = _json_loads(_native_completion(model, messages, ollama_base_url, None))
    if data is None:
        return None, last_err or "invalid_json"
    return data, None


def _transcribe(
    client: OpenAI,
    model: str,
    page: Page,
    ollama_base_url: str,
    default_client: bool,
) -> tuple[list[tuple[str, bool]] | None, str | None]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": _TRANSCRIPT_PROMPT}]
    meta = {
        "page_index": page.index,
        "source_index": page.source_index,
        "char_start": page.char_start,
        "char_end": page.char_end,
    }
    meta_text = "<source>\npage_metadata: " + json.dumps(meta, ensure_ascii=False)
    if page.text is not None:
        meta_text += f"\ntext_layer:\n```\n{page.text}\n```"
    if page.raw_text is not None:
        meta_text += f"\nraw_text_layer:\n```\n{page.raw_text}\n```"
    meta_text += "\n</source>"
    parts.append({"type": "text", "text": meta_text})
    if page.image:
        parts.append(_b64img(page.image))
    data, err = _call_llm(
        client,
        model,
        [{"role": "user", "content": parts}],
        ollama_base_url,
        default_client,
        _TRANSCRIPT_SCHEMA,
    )
    if err or data is None:
        return None, err

    segments = data.get("segments")
    if isinstance(segments, list):
        parsed: list[tuple[str, bool]] = []
        for entry in segments:
            if isinstance(entry, dict):
                txt = _norm(str(entry.get("text") or ""))
                if txt:
                    parsed.append((txt, bool(entry.get("is_recipe", True))))
            elif isinstance(entry, str) and _norm(entry):
                parsed.append((_norm(entry), True))
        return parsed, None

    lines = data.get("lines")
    if isinstance(lines, list):
        return [(_norm(str(x)), True) for x in lines if _norm(str(x))], None
    if isinstance(data.get("transcript"), str):
        t = _norm(data["transcript"])
        return ([(t, True)] if t else []), None
    return None, "invalid_json"


def _transcribe_crop(
    client: OpenAI,
    model: str,
    image: bytes,
    ollama_base_url: str,
    default_client: bool,
) -> tuple[str | None, str | None]:
    data, err = _call_llm(
        client,
        model,
        [{"role": "user", "content": [{"type": "text", "text": _CROP_PROMPT}, _b64img(image)]}],
        ollama_base_url,
        default_client,
        _CROP_SCHEMA,
    )
    if err or data is None:
        return None, err
    if isinstance(data.get("text"), str):
        return _norm(data["text"]), None
    lines = data.get("lines")
    if isinstance(lines, list) and lines:
        return _norm(" ".join(_norm(str(x)) for x in lines if _norm(str(x)))), None
    if isinstance(data.get("transcript"), str):
        return _norm(data["transcript"]), None
    return "", None
