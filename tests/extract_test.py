from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np

from src.extract import extract
from src.extract.models import ExtractWarning, Reading
from src.ingest import IngestResult, Page


# ---- fakes ----

class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        content = r if isinstance(r, str) else json.dumps(r)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class FakeOCR:
    def __init__(self, results):
        # results: list of list[(box,text,score)] or Exception per call
        self.results = list(results)

    def __call__(self, image):
        r = self.results.pop(0) if self.results else []
        if isinstance(r, Exception):
            raise r
        return r, 0.0


def png():
    img = np.full((40, 200, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def box(x=5, y=5, w=80, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def page_text(text="1/4 l Essig", idx=0):
    return Page(index=idx, source_index=0, text=text, raw_text=text,
                char_start=0, char_end=len(text), width=200, height=40, sha256="s")


def page_image(text_layer=None, idx=0):
    t = text_layer
    return Page(index=idx, source_index=0, image=png(),
                text=t, raw_text=t,
                char_start=0 if t else None, char_end=len(t) if t else None,
                width=200, height=40, sha256="s")


def ingest(*pages):
    return IngestResult(sources=(), pages=tuple(pages), text="", warnings=())


def transcript(lines):
    return {"lines": lines}


def structure(items, steps=None, servings=None):
    if steps is None:
        steps = [{"index": 0, "text": "step", "evidence_ids": [it.get("evidence_id", "ev-001") for it in items]}]
    if servings is None:
        servings = {"value": None, "quote": None, "evidence_id": None}
    return {"steps": steps, "items": items, "servings": servings}


def warnings_of(res, code=None):
    ws = res.warnings
    return [w for w in ws if (code is None or w.code == code)]


# ---- tests ----

def test_match_no_warning():
    # ocr == llm, no text layer -> no mismatch warning
    ocr = FakeOCR([[ (box(), "1/4 l Essig", 0.99) ]])
    client = FakeClient([
        transcript(["1/4 l Essig"]),  # page transcript
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Essig",
                     "type": "ingredient", "quantity": "1/4", "unit": "l", "quote": "1/4 l Essig" }]),
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    assert warnings_of(res, "mismatch") == []
    assert len(res.items) == 1
    assert res.items[0].name == "Essig"


def test_text_layer_match_grounds_segment():
    # matching text layer anchors the segment and its character offsets
    text = "1/4 l Essig"
    ocr = FakeOCR([[ (box(), text, 0.95) ]])
    client = FakeClient([
        transcript([text]),
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Essig",
                     "type": "ingredient", "quantity": "1/4", "unit": "l", "quote": text }],
                  steps=[{"index": 0, "text": text, "evidence_ids": ["ev-001"]}]),
    ])
    res = extract(ingest(page_image(text_layer=text)), client=client, ocr=ocr)
    assert warnings_of(res, "mismatch") == []
    assert res.items and res.items[0].grounding == "text_layer"
    assert res.items[0].char_start == 0
    assert res.items[0].char_end == len(text)


def test_mismatch_recheck_crop_distinction_warns_with_location():
    # ocr!=llm, crop_ocr!=crop_llm -> mismatch with bbox, readings (full+crop), selected
    ocr = FakeOCR([
        [ (box(10,10,60,18), "1/2 l Essig", 0.90) ],  # full
        [ (box(0,0,50,15), "1/2 l Essig", 0.80) ],   # crop ocr
    ])
    client = FakeClient([
        transcript(["1/4 l Essig"]),  # full llm
        transcript(["1/3 l Essig"]),  # crop llm
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Essig",
                     "type": "ingredient", "quantity": "1/3", "unit": "l", "quote": "1/3 l Essig" }]),
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    ms = warnings_of(res, "mismatch")
    assert ms, "expected mismatch after failed recheck"
    w = ms[0]
    assert w.page == 0
    assert w.bbox == (10, 10, 70, 28)
    assert w.quote
    assert w.selected is not None
    assert "OCR mismatch" in (w.reason or "")
    sources = {r.source for r in w.readings}
    assert {"ocr", "llm", "crop_ocr", "crop_llm"}.issubset(sources)


def test_recheck_resolves_no_warning():
    # ocr!=llm but crop_ocr==crop_llm -> resolved, no mismatch
    ocr = FakeOCR([
        [ (box(), "1/2 l Essig", 0.90) ],
        [ (box(), "1/4 l Essig", 0.88) ],
    ])
    client = FakeClient([
        transcript(["1/4 l Essig"]),
        transcript(["1/4 l Essig"]),
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Essig",
                     "type": "ingredient", "quantity": "1/4", "unit": "l", "quote": "1/4 l Essig" }]),
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    assert warnings_of(res, "mismatch") == []


def test_transcription_before_json_call_order():
    ocr = FakeOCR([[ (box(), "Salz", 0.99) ]])
    client = FakeClient([
        transcript(["Salz"]),
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Salz",
                     "type": "ingredient", "quantity": None, "unit": None, "quote": "Salz" }]),
    ])
    extract(ingest(page_image()), client=client, ocr=ocr)
    calls = client.chat.completions.calls
    assert len(calls) == 2
    first = json.dumps(calls[0]["messages"], ensure_ascii=False)
    second = json.dumps(calls[1]["messages"], ensure_ascii=False)
    assert "Transcribe" in first or "transcrev" in first.lower() or "lines" in first
    assert "evidence" in second.lower() or "steps" in second


def test_structure_mismatch_item3_lost():
    # evidence has 3 items but structuring returns only 2 -> lost item warned with location
    ocr = FakeOCR([[ (box(0,0,50,10), "Salz", 0.99),
                     (box(0,12,50,10), "Pfeffer", 0.99),
                     (box(0,24,50,10), "Zucker", 0.99) ]])
    client = FakeClient([
        transcript(["Salz", "Pfeffer", "Zucker"]),
        structure([
            { "step_index": 0, "evidence_id": "ev-001", "name": "Salz", "type": "ingredient",
              "quantity": None, "unit": None, "quote": "Salz" },
            { "step_index": 0, "evidence_id": "ev-002", "name": "Pfeffer", "type": "ingredient",
              "quantity": None, "unit": None, "quote": "Pfeffer" },
            # ev-003 Zucker missing -> lost
        ], steps=[{"index":0,"text":"step","evidence_ids":["ev-001","ev-002"]}]),
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    sms = warnings_of(res, "structure_mismatch")
    assert sms, "expected structure_mismatch for lost evidence/item 3"
    lost = [w for w in sms if "Zucker" in (w.quote or "") or "perd" in (w.message or "").lower() or "não represent" in (w.message or "").lower()]
    assert lost, sms
    w = lost[0]
    assert w.page == 0
    assert w.bbox == (0, 24, 50, 34)
    sources = {r.source for r in w.readings}
    assert "accepted_evidence" in sources
    assert "llm_structured" in sources


def test_ocr_error_warns():
    ocr = FakeOCR([Exception("boom")])
    client = FakeClient([
        transcript(["x"]),  # transcript still attempted
        {"steps": [], "items": [], "servings": {"value": None, "quote": None, "evidence_id": None}},
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    assert warnings_of(res, "ocr_error")


def test_model_error_warns():
    ocr = FakeOCR([[ (box(), "x", 0.9) ]])
    client = FakeClient([
        Exception("down"),  # transcript fails
        {"steps": [], "items": [], "servings": {"value": None, "quote": None, "evidence_id": None}},
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    assert warnings_of(res, "model_error")


def test_invalid_json_warns():
    ocr = FakeOCR([[ (box(), "x", 0.9) ]])
    client = FakeClient([
        "not json",
        {"steps": [], "items": [], "servings": {"value": None, "quote": None, "evidence_id": None}},
    ])
    res = extract(ingest(page_image()), client=client, ocr=ocr)
    assert warnings_of(res, "invalid_json")


def test_text_only_page_no_image():
    # text layer only -> evidence from text, no ocr/llm calls needed
    client = FakeClient([
        structure([{ "step_index": 0, "evidence_id": "ev-001", "name": "Salz",
                     "type": "ingredient", "quantity": None, "unit": None, "quote": "Salz" }],
                  steps=[{"index":0,"text":"Salz","evidence_ids":["ev-001"]}])
    ])
    res = extract(ingest(page_text("Salz")), client=client, ocr=FakeOCR([]))
    assert warnings_of(res, "mismatch") == []
    assert res.items and res.items[0].quote == "Salz"
