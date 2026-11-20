# SPEC.md

## Scope

This document specifies the cookbook pipeline one stage at a time. **Stage 1 — Ingest** and **Stage 2 — Extract** are specified below; Manipulate and Output will be added in later revisions.

**Language convention:** all artifacts written to this repository are authored in English — this document, README, AI log, source code (including identifiers, comments and docstrings), tests, config examples, commit messages and prose in example outputs. Portuguese is used only for conversation with the user and is never written to disk. Source recipe content stays verbatim in its original language; ingest does not translate it.

### Environment

All development and test runs use the conda environment **`finlyt-task`** (Python 3.11). Dependencies are declared in `requirements.txt`:

```bash
conda activate finlyt-task
python -m pip install -r requirements.txt
python -m pytest
```

`requirements.txt` pins the versions already installed in `finlyt-task` for packages used directly by the project, plus OpenCV for image decoding. Extract calls a local Ollama server with `qwen2.5vl:7b` pulled through the OpenAI-compatible client; no API key is used, and the client points to `http://localhost:11434/v1` with `api_key="ollama"`. No OCR dependency is used.

---

## Stage 1 — Ingest

### Purpose

Turn any supported input (plain text, image files, PDF) into a single predictable object that the Extract stage can consume. Ingest performs **no** interpretation, **no** OCR and **no** LLM calls: it only detects the input kind, extracts whatever text layer exists, and produces normalized page images.

### Public API

```python
def ingest(
    source: str | Path | bytes | Sequence[Path | bytes],
    *,
    name: str | None = None,
) -> IngestResult: ...
```

- A bare `str` is always treated as plain text; a file path must be passed as `Path`.
- A `Sequence` is processed in caller order, one source per element.
- `name` overrides the derived source name when the input is raw bytes.

### Data model

```python
from dataclasses import dataclass
from typing import Literal, Sequence

Kind = Literal["pdf", "image", "text"]

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
    name: str | None
    image: bytes | None
    text: str | None        # normalized
    raw_text: str | None    # verbatim PyMuPDF output
    char_start: int | None  # offset into IngestResult.text
    char_end: int | None    # exclusive end offset
    width: int | None
    height: int | None
    sha256: str | None

@dataclass(frozen=True)
class IngestResult:
    sources: tuple[SourceInfo, ...]
    pages: tuple[Page, ...]
    text: str
    warnings: tuple[Warning, ...]
```

Immutability is deliberate: the result is a handoff contract, not a working buffer. Page indices are stable and assigned in output order.

### Routing

| Input | Detection | Reading strategy |
| --- | --- | --- |
| Plain text | `str` | Single text page |
| Text file | `.txt` path or bytes | Single text page |
| Image | Non-`%PDF` bytes decodable by OpenCV | One page per file |
| PDF | `%PDF` magic bytes | PyMuPDF, one page per PDF page |

### Rules

**Plain text.** Accept `str` or `.txt`. Decode UTF-8 (with BOM tolerance), then fall back to cp1252, which is common for German text. Produce exactly one `Page` with `image=None`, `raw_text=None`, and `width`/`height`/`sha256` set to `None`. The text is normalized and copied into the global anchor.

**Images.** Each input file becomes one page in **input order**; filenames are not sorted. Decode with OpenCV, which applies EXIF orientation automatically, then resize so the long edge is at most **1280 px**, preserving aspect ratio. Encode as PNG bytes into `Page.image`; OpenCV converts BGR to RGB while writing the PNG. Set `width`/`height` to the final image dimensions and `sha256` to the hash of the PNG bytes; text fields stay `None`. No OCR runs here: handwriting is read by the vision-capable model in Extract.

**PDFs.** Open with PyMuPDF. For **every** page, in order:

1. `raw_text = page.get_text("text", sort=True)` — `sort=True` preserves reading order.
2. `text = normalize(raw_text)` — collapse whitespace; keep `raw_text` for verbatim evidence. A page counts as having text only when `text.strip()` is non-empty.
3. Render the **full page** with `page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)`, where `scale` is chosen so the long edge is at most 1280 px. Store the PNG bytes in `Page.image` with matching dimensions and hash.

Both channels are always emitted, including for pure-text PDFs. Full-page rendering (rather than extracting embedded images only) is required because drawings, arrows and margin notes carry meaning. Detection of embedded images is advisory metadata only and never changes the render decision.

**Global anchor.** `IngestResult.text` is `"\n".join(page.text for page in pages if page.text)`. `Page.char_start` and `Page.char_end` mark the exact slice of that string belonging to each page; page separator characters belong to no page. Pages without text have both offsets set to `None`.

**Limits and failures.** Maximum 50 MB per input and 100 pages per run. Invalid, encrypted or unreadable inputs are skipped with a structured `Warning` and the run continues; nothing is dropped silently. Warning codes: `input_too_large`, `unsupported_input`, `encrypted_pdf`, `page_unreadable`, `decode_failed`, `empty_input`, `resized`. `Warning.page` carries the affected `Page.index` when applicable.

**Determinism.** Fixed render scale, no timestamps in output, and per-page hashes mean identical inputs produce identical bytes. This makes ingest directly testable without an LLM.

### Handoff to Extract

The contract is index-addressable: `pages[i].image` (PNG bytes), `pages[i].text` (normalized), `pages[i].raw_text` (verbatim), plus `IngestResult.text` as the global anchor. `Page.source_index` maps back to `sources`; `char_start`/`char_end` map any quote in the global anchor back to its page. Nothing is written to disk.

### Testing

`pytest` runs inside `finlyt-task`. Ingest is deterministic, so it is covered by unit tests: no network, no LLM and no real recipe files. Fixtures are generated in `tmp_path` with PyMuPDF and OpenCV. `Recipe.pdf` is gitignored and is not a test dependency; the user runs it manually against the finished software.

```
tests/
  conftest.py        # helpers: make_pdf(), make_image(), make_txt()
  test_routing.py    # str/Path/bytes/%PDF/Sequence, unsupported, empty, source_index
  test_text.py       # BOM, cp1252, normalization, offset slices
  test_images.py     # EXIF, channel order, 1280 px without upscale, PNG, input order
  test_pdf.py        # sort=True reading order, both channels, render every page, encryption, bad page
  test_contract.py   # frozen dataclasses, equality, hashes, no disk writes
  test_failures.py   # size/page caps via monkeypatch, best-effort warnings with code and page
```

Coverage targets:

- **Routing:** each input kind maps to the correct `SourceInfo.kind`; `Sequence` order and `source_index` are preserved; unsupported and empty inputs produce warnings instead of exceptions.
- **Contract:** `pages[i].index == i`; every hash matches its bytes; `text[char_start:char_end] == page.text` for every page with text; `ingest(x) == ingest(x)`; no files are written to disk.
- **Text:** UTF-8 with BOM and cp1252 umlauts decode correctly; whitespace normalization does not alter `raw_text`.
- **Images:** EXIF orientation is applied; a red source pixel stays red in the PNG; the long edge is at most 1280 px with aspect ratio preserved and no upscaling; output is valid PNG.
- **PDFs:** `sort=True` reading order; an image-only page has `text=None` and emits no warning; a mixed PDF still produces an image for pure-text pages; render scale matches the 1280 px cap; encrypted PDFs are skipped with `encrypted_pdf`; an unreadable page yields `page_unreadable` with its index; repeated runs produce byte-identical images.
- **Failures:** size and page caps are tested by monkeypatching the limits, avoiding large fixtures; one bad source does not discard the good ones.

Not covered here: handwriting accuracy and hallucination checks. Those belong to the Extract stage and need an evaluation set, not unit tests.

### Explicitly not in this stage

- OCR in ingest (considered and rejected; the vision model reads the handwriting).
- Writing page images to numbered files (considered and rejected in favor of in-memory bytes).
- Extracting only embedded images from a PDF (loses layout and decoration).
- Pydantic models (a frozen dataclass keeps the boundary dependency-free).

## Stage 2 — Extract

### Purpose

Turn an `IngestResult` into a traceable plain dictionary containing final page texts, ordered steps, ingredients, tools, servings and warnings. Extract is the first interpreting stage. Deterministic code collects readings and validates output; the local Ollama-hosted `qwen2.5vl:7b` model is used only to transcribe page images and to turn the assembled text into structured data. There is no OCR stage, no bounding boxes and no immutable `ExtractResult` dataclass.

The implementation is split across six modules:

| Module | Responsibility |
| --- | --- |
| `src/extract/pipeline.py` | Orchestrates collection, verification, assembly, structuring and the final dict. |
| `src/extract/Collector.py` | Defines `PageReadings` and produces native-text or LLM image readings. |
| `src/extract/ImageManipulation.py` | Splits an image into overlapping vertical crops. |
| `src/extract/LLMMethods.py` | Prompts, schemas, OpenAI-compatible calls, native Ollama fallback and parsing helpers. |
| `src/extract/TextVerifier.py` | Merges crop readings and reconciles them with the full-page reading. |
| `src/extract/JSONBuilder.py` | Validates raw structured output, assigns ids/spans and emits warnings. |

The pipeline has five fixed phases:

1. **Collect:** use the native text layer when present; otherwise transcribe the full image and two overlapping crops with the LLM.
2. **Verify:** reconcile the full-page transcript with the merged crop transcript using word-level sequence matching.
3. **Assemble:** join page texts with newline separators and record character offsets for every page.
4. **Structure:** send the assembled text to one schema-constrained LLM call.
5. **Validate:** deterministic code drops or clears invalid objects, assigns ids, rebuilds links and emits warnings.

No manipulation happens here.

### Public API

```python
def extract(
    result: Any,
    *,
    model: str = "qwen2.5vl:7b",
    client: OpenAI | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
) -> dict[str, Any]: ...
```

- `result` only needs a `pages` attribute; `page.index`, `page.source_index`, `page.text` and `page.image` are read with `getattr` defaults.
- `client` is injectable so unit tests need no Ollama server, network or real recipe. When it is omitted, the default is `openai.OpenAI(base_url=ollama_base_url, api_key="ollama")`.
- A supplied client is treated as the default Ollama client when its `base_url` contains `ollama_base_url` and its `api_key` is `"ollama"`; this enables the native Ollama fallback.
- The model is called with `temperature=0`, `max_tokens=4096` and `extra_body={"options": {"num_ctx": 16384, "num_predict": 4096}}`.
- The returned dictionary always has `text`, `pages`, `steps`, `items`, `servings`, `warnings` and `model`.

### Data model

`PageReadings` is the only frozen dataclass in Extract:

```python
@dataclass(frozen=True)
class PageReadings:
    page_index: int
    source_index: int
    native_text: str | None
    full_text: str
    top_text: str
    bottom_text: str
    errors: tuple[str, ...]
```

The other artifacts are plain dictionaries. A verified page has this shape:

```python
{
    "page": int,
    "source_index": int,
    "text": str,
    "origin": "text_layer" | "llm_verified",
    "char_start": int,
    "char_end": int,
    "readings": {
        "native_text": str | None,
        "full_text": str,
        "top_text": str,
        "bottom_text": str,
    },
}
```

Steps, items, servings and warnings are dictionaries produced by `JSONBuilder.py`. Ids (`step-001`, `item-001`, `serving-001`) are assigned by validation, never trusted from the model. Spans are dictionaries `{"start": int, "end": int}` into `extract()["text"]`; inferred objects have neither a literal quote nor a span.

A step carries `id`, `index`, `text`, `quote`, `inferred`, `inference_reason`, `page_index`, `span`, `item_ids`, `duration`, `conditions` and `continues_previous`. An item carries `id`, `step_id`, `name`, `type` (`ingredient` or `tool`), `quantity`, `unit`, `preparation`, `optional`, `inferred`, `inference_reason`, `quote`, `page_index` and `span`. Servings carries `id`, `value`, `inferred`, `inference_reason`, `quote` and `span`.

Warnings are dictionaries with `code`, `severity`, `message` and `page`, plus optional fields such as `quote`, `span`, `step_id`, `reason` or the two reading strings used by `TextVerifier.py`.

### Reading channels

| Page state | LLM transcription | Final text | Origin | Verification |
| --- | --- | --- | --- | --- |
| Non-empty native text layer | None | `page.text`, whitespace-normalized | `text_layer` | Not run |
| Image, no native text | Full image plus two overlapping crops | Reconciled full/crop transcript | `llm_verified` | Word-level sequence matching |
| Text-only page (`image=None`) | None | Empty string with `missing_page_image` collected as an error | `llm_verified` | Not applicable |

The crops are made by `split_vertical_two_thirds`: the top crop covers rows `0:round(2h/3)` and the bottom crop covers rows `round(h/3):h`, so they overlap in the middle third. The full page and both crops are transcribed independently with the same verbatim-transcription prompt; no text layer or alternative reading is included in that prompt.

### Rules

**Collection.** A non-empty `page.text` is copied verbatim after whitespace collapsing and short-circuits all LLM work. Without native text, `Collector.py` first transcribes `page.image`. It then splits the image and transcribes the top and bottom crops. Transcription errors are preserved as `full_transcript_error: ...`, `top_transcript_error: ...` or `bottom_transcript_error: ...`; a crop split failure keeps the full-page reading and records `crop_split_error: ...`. All collector errors are surfaced by `pipeline.py` as `collection_error` warnings.

**Transcription.** `TRANSCRIBE_PROMPT` asks for verbatim text in the source language as strict JSON `{"segments": ["..."]}`. `LLMMethods._extract_segments` also accepts `text`, `transcript`, `lines` and `{"text": ...}` segment objects so that minor model-shape variations are normalized. Segment whitespace is collapsed and empty segments are dropped.

**Crop merging.** `merge_crops` tokenizes the top and bottom readings and uses `SequenceMatcher` to find their longest common run. A run of at least three words is treated as the overlap and removed from the front of the merged result; shorter or absent overlap means the readings are concatenated in reading order.

**Verification.** `verify_transcript` compares the words of the full-page reading with the merged crop reading. Equal token sequences keep the full reading without a warning. A replacement whose casefolded span similarity is at least `0.75` uses the crop wording and emits `crop_refined`; a lower similarity keeps the full wording and emits `verification_failed`. Deletions keep the full segment with `missing_in_crop`; insertions add the crop segment with `recovered_by_crop`. Empty full or crop readings emit `full_transcript_empty` or `crop_transcript_empty`. Every verification warning carries `page`, `full_reading` and `crop_reading`.

**Assembly.** Page texts are joined with `\n` in page order into `extract()["text"]`. The separator belongs to no page; each page's `char_start` and `char_end` address only its own text inside that assembled string.

**Canonicalization.** `JSONBuilder._canonical` collapses whitespace, casefolds, translates Unicode fractions to ASCII and replaces decimal commas with dots. It is used only for validation of quotes, durations, conditions and duplicates; returned text keeps the model's original spelling.

**Structuring.** `STRUCTURE_PROMPT` sends the assembled text fenced as data and requests `steps`, `items` and `servings` with all fields listed in the schema. Unknown values must be JSON `null`. Explicit objects need a literal `quote`; inferred objects need `inferred=true`, an `inference_reason` and an empty quote. `call_llm` first tries `response_format={"type": "json_schema", ...}`, then `{"type": "json_object"}`. When the client is the default Ollama client and parsing still fails, it retries Ollama's native `/api/chat` with `format` set to the schema and then without a schema.

**Validation.** `build_structured_recipe` drops empty steps, sorts steps by their original `index`, and keeps explicit steps only when their quote is canonically present in the full text. Inferred steps lose any quote and receive a default reason if missing. Durations and conditions are kept only when they appear verbatim (canonically) in the step text, `continues_previous` is reset on the first step, and word-bearing gaps between located step quotes produce `unmapped_text` warnings. Item quotes are first matched inside the assigned step span, then globally with `step_scope_mismatch`; unmatched quotes drop the item. Duplicate items are merged only within the same step using canonical `type + name + quantity + unit + preparation`. Servings must be a positive integer or `null`; bad quotes reset the value to inferred. When the LLM structure call fails entirely, the result is empty steps/items plus an inferred serving with reason `Structuring failed`.

Warning codes emitted by Extract are: `collection_error`, `structure_extraction_failed`, `full_transcript_empty`, `crop_transcript_empty`, `crop_refined`, `verification_failed`, `missing_in_crop`, `recovered_by_crop`, `empty_step_dropped`, `quote_not_found`, `inference_without_reason`, `inferred_quote_cleared`, `duration_not_in_step`, `condition_not_in_step`, `invalid_continues_previous`, `unmapped_text`, `orphan_item_dropped`, `step_scope_mismatch`, `duplicate_merged` and `servings_inferred`. Warning severities are `info`, `warning` or `error`.

### Example result

```json
{
  "text": "Chop 2 onions in a pan.",
  "pages": [
    {
      "page": 0,
      "source_index": 0,
      "text": "Chop 2 onions in a pan.",
      "origin": "text_layer",
      "char_start": 0,
      "char_end": 23,
      "readings": {
        "native_text": "Chop 2 onions in a pan.",
        "full_text": "Chop 2 onions in a pan.",
        "top_text": "",
        "bottom_text": ""
      }
    }
  ],
  "steps": [
    {
      "id": "step-001",
      "index": 0,
      "text": "Chop 2 onions in a pan.",
      "quote": "Chop 2 onions in a pan.",
      "inferred": false,
      "inference_reason": null,
      "page_index": 0,
      "span": { "start": 0, "end": 23 },
      "item_ids": ["item-001", "item-002"],
      "duration": null,
      "conditions": [],
      "continues_previous": false
    }
  ],
  "items": [
    {
      "id": "item-001",
      "step_id": "step-001",
      "name": "onion",
      "type": "ingredient",
      "quantity": "2",
      "unit": null,
      "preparation": null,
      "optional": false,
      "inferred": false,
      "inference_reason": null,
      "quote": "2 onions",
      "page_index": 0,
      "span": { "start": 5, "end": 13 }
    },
    {
      "id": "item-002",
      "step_id": "step-001",
      "name": "pan",
      "type": "tool",
      "quantity": null,
      "unit": null,
      "preparation": null,
      "optional": false,
      "inferred": false,
      "inference_reason": null,
      "quote": "a pan",
      "page_index": 0,
      "span": { "start": 17, "end": 22 }
    }
  ],
  "servings": {
    "id": "serving-001",
    "value": null,
    "inferred": true,
    "inference_reason": "No serving count stated in the text",
    "quote": null,
    "span": null
  },
  "warnings": [
    {
      "code": "servings_inferred",
      "severity": "info",
      "message": "Serving count not stated in recipe text.",
      "page": null,
      "reason": "No serving count stated in the text"
    }
  ],
  "model": "qwen2.5vl:7b"
}
```

### Handoff to CLI

`main.py` consumes the returned dictionary directly. It writes the full dictionary as UTF-8 JSON to `-o/--output` (default `output.json`) and a readable warnings report to `-w/--warnings` (default `warnings.txt`), then prints page, step, item and warning counts. No Manipulate stage is implemented yet, and no downstream code depends on frozen Extract dataclasses.

### Testing

Default `pytest` is fully offline. `tests/extract_test.py` uses a fake OpenAI client plus monkeypatched `transcribe_image`, `split_vertical_two_thirds`, `call_llm`, `_native_ollama_chat`, `collect_document_readings`, `verify_transcript`, `build_structured_recipe` and `extract_structure`. No Ollama server, network, OCR engine or `Recipe.pdf` is required, and there is no opt-in integration suite.

Coverage targets:

- Crop splitting rejects empty, undecodable and too-small images and returns the expected overlapping row ranges.
- `LLMMethods` covers client detection, JSON parsing, response-format fallback, native Ollama fallback, segment extraction, prompt/image message construction and fenced source text.
- `Collector` covers native-text short-circuiting, missing images, all three transcription calls, error prefixes, crop-split failure and document order.
- `TextVerifier` covers crop merging, empty readings, exact matches, minor refinements, major discrepancies, deletions and insertions.
- `JSONBuilder` covers step sorting, ids, spans, inferred-field sanitization, unmapped text, item scoping and deduplication, orphan items and all servings branches.
- `pipeline.extract` covers native-text origin, image-page verification, newline assembly, character offsets, warning order, default-client creation and the final dictionary shape.

Model accuracy is evaluated separately with a golden set, not with exact-output unit tests.

### Explicitly not in this stage

- OCR, rapidocr_onnxruntime, bounding boxes, padded crops and crop re-reads by a second reader: verification uses only full and cropped LLM transcripts.
- An immutable `ExtractResult` or frozen dataclasses for steps/items: the public contract is plain dictionaries.
- An LLM judge shown both readings: verification is deterministic sequence matching over words.
- Canonical-exact acceptance for transcript text: full/crop reconciliation uses a `0.75` similarity threshold for replacements; canonicalization is reserved for structured-data validation.
- Model-provided ids: all ids are assigned by deterministic validation.
- OCR-specific warning codes, quantity/unit conflicts, step-order conflicts and text-layer-authoritative warnings: they are not emitted by this implementation.
- A Manipulate handoff: `main.py` writes JSON and warnings files directly.
- Translation, cross-dimension unit conversion, scaling and substitution: not part of Extract.
- Pydantic models and a second model: one LLM plus one frozen `PageReadings` keeps the boundary simple.
