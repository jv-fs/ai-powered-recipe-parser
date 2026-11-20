# SPEC.md

## Scope

This document specifies the cookbook pipeline one stage at a time. Only **Stage 1 — Ingest** is specified below; Extract, Manipulate and Output will be added in later revisions.

**Language convention:** all artifacts written to this repository are authored in English — this document, README, AI log, source code (including identifiers, comments and docstrings), tests, config examples, commit messages and prose in example outputs. Portuguese is used only for conversation with the user and is never written to disk. Source recipe content stays verbatim in its original language; ingest does not translate it.

### Environment

All development and test runs use the conda environment **`finlyt-task`** (Python 3.11). Dependencies are declared in `requirements.txt`:

```bash
conda activate finlyt-task
python -m pip install -r requirements.txt
python -m pytest
```

`requirements.txt` pins the versions already installed in `finlyt-task` for packages used directly by the project, plus OpenCV for image decoding. No second environment is needed.

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

**Images.** Each input file becomes one page in **input order**; filenames are not sorted. Decode with OpenCV, which applies EXIF orientation automatically, then resize so the long edge is at most **2000 px**, preserving aspect ratio. Encode as PNG bytes into `Page.image`; OpenCV converts BGR to RGB while writing the PNG. Set `width`/`height` to the final image dimensions and `sha256` to the hash of the PNG bytes; text fields stay `None`. No OCR runs here: handwriting is read by the vision-capable model in Extract.

**PDFs.** Open with PyMuPDF. For **every** page, in order:

1. `raw_text = page.get_text("text", sort=True)` — `sort=True` preserves reading order.
2. `text = normalize(raw_text)` — collapse whitespace; keep `raw_text` for verbatim evidence. A page counts as having text only when `text.strip()` is non-empty.
3. Render the **full page** with `page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)`, where `scale` is chosen so the long edge is at most 2000 px. Store the PNG bytes in `Page.image` with matching dimensions and hash.

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
  test_images.py     # EXIF, channel order, 2000 px without upscale, PNG, input order
  test_pdf.py        # sort=True reading order, both channels, render every page, encryption, bad page
  test_contract.py   # frozen dataclasses, equality, hashes, no disk writes
  test_failures.py   # size/page caps via monkeypatch, best-effort warnings with code and page
```

Coverage targets:

- **Routing:** each input kind maps to the correct `SourceInfo.kind`; `Sequence` order and `source_index` are preserved; unsupported and empty inputs produce warnings instead of exceptions.
- **Contract:** `pages[i].index == i`; every hash matches its bytes; `text[char_start:char_end] == page.text` for every page with text; `ingest(x) == ingest(x)`; no files are written to disk.
- **Text:** UTF-8 with BOM and cp1252 umlauts decode correctly; whitespace normalization does not alter `raw_text`.
- **Images:** EXIF orientation is applied; a red source pixel stays red in the PNG; the long edge is at most 2000 px with aspect ratio preserved and no upscaling; output is valid PNG.
- **PDFs:** `sort=True` reading order; an image-only page has `text=None` and emits no warning; a mixed PDF still produces an image for pure-text pages; render scale matches the 2000 px cap; encrypted PDFs are skipped with `encrypted_pdf`; an unreadable page yields `page_unreadable` with its index; repeated runs produce byte-identical images.
- **Failures:** size and page caps are tested by monkeypatching the limits, avoiding large fixtures; one bad source does not discard the good ones.

Not covered here: handwriting accuracy and hallucination checks. Those belong to the Extract stage and need an evaluation set, not unit tests.

### Explicitly not in this stage

- OCR in ingest (considered and rejected; the vision model reads the handwriting).
- Writing page images to numbered files (considered and rejected in favor of in-memory bytes).
- Extracting only embedded images from a PDF (loses layout and decoration).
- Pydantic models (a frozen dataclass keeps the boundary dependency-free).
