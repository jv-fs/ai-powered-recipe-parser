from __future__ import annotations

from typing import Any

from openai import OpenAI

from .Scaler import scale_recipe
from .Veganizer import veganize_recipe
from .VeganVerifier import verify_veganized_recipe


def manipulate(
    recipe: dict[str, Any],
    *,
    target_servings: int | None = None,
    scale_factor: float | None = None,
    assumed_base_servings: int | None = None,
    make_vegan: bool = False,
    multipliers: dict[str, float] | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
) -> dict[str, Any]:
    scaled = scale_recipe(
        recipe,
        target_servings=target_servings,
        scale_factor=scale_factor,
        assumed_base_servings=assumed_base_servings,
    )

    if not make_vegan:
        return scaled

    veganized = veganize_recipe(
        scaled,
        client=client,
        model=model,
        ollama_base_url=ollama_base_url,
        multipliers=multipliers,
    )
    return verify_veganized_recipe(scaled, veganized)