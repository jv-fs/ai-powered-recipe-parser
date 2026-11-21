from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402


def test_env_example_contains_provider_variables() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "LLM_BASE_URL=" in text
    assert "LLM_API_KEY=" in text
    assert "VISION_MODEL=" in text
    assert "TEXT_MODEL=" in text


def test_config_reads_provider_variables_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("VISION_MODEL", "vision-test")
    monkeypatch.setenv("TEXT_MODEL", "text-test")

    try:
        reloaded = importlib.reload(config)
        assert reloaded.LLM_BASE_URL == "https://api.example.test/v1"
        assert reloaded.LLM_API_KEY == "sk-test"
        assert reloaded.VISION_MODEL == "vision-test"
        assert reloaded.TEXT_MODEL == "text-test"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
