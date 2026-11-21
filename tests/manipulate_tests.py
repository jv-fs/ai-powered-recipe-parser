from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manipulate.Scaler import (  # noqa: E402
    format_number,
    format_quantity,
    parse_quantity,
    resolve_scale_factor,
    scale_recipe,
)


def _servings(value: int | None = 4) -> dict[str, Any]:
    return {
        "id": "serving-001",
        "value": value,
        "inferred": value is None,
        "inference_reason": None,
        "quote": None,
        "span": None,
    }


def _recipe() -> dict[str, Any]:
    return {
        "text": "Mix 2 cups flour. Add 1/2 tsp salt. Use a large bowl.",
        "steps": [
            {
                "id": "step-001",
                "text": "Mix 2 cups flour.",
                "quote": "Mix 2 cups flour.",
                "item_ids": ["item-001"],
            },
            {
                "id": "step-002",
                "text": "Add 1/2 tsp salt.",
                "quote": "Add 1/2 tsp salt.",
                "item_ids": ["item-002", "item-003"],
            },
        ],
        "items": [
            {
                "id": "item-001",
                "type": "ingredient",
                "name": "flour",
                "quantity": "2 cups",
                "quote": "2 cups flour",
                "step_id": "step-001",
            },
            {
                "id": "item-002",
                "type": "ingredient",
                "name": "salt",
                "quantity": "1/2 tsp",
                "quote": "1/2 tsp salt",
                "step_id": "step-002",
            },
            {
                "id": "item-003",
                "type": "tool",
                "name": "bowl",
                "quantity": "1",
                "quote": "a large bowl",
                "step_id": "step-002",
            },
        ],
        "servings": _servings(),
        "warnings": [
            {
                "code": "existing_warning",
                "severity": "info",
                "message": "kept",
            }
        ],
    }


def test_resolve_scale_factor_uses_target_and_scale_factor() -> None:
    assert resolve_scale_factor({"value": 4}, target_servings=6) == (1.5, 6, [])
    assert resolve_scale_factor({"value": 4}, scale_factor=0.5) == (0.5, 2, [])
    assert resolve_scale_factor(
        {"value": 4}, target_servings=10, scale_factor=0.5
    ) == (0.5, 2, [])


def test_resolve_scale_factor_without_change_or_unknown_base() -> None:
    assert resolve_scale_factor({"value": 4}) == (1.0, 4, [])
    assert resolve_scale_factor({"value": None}) == (1.0, None, [])

    factor, new_servings, warnings = resolve_scale_factor(
        {"value": None}, target_servings=6, assumed_base_servings=4
    )

    assert factor == 1.5
    assert new_servings == 6
    assert warnings == [
        {
            "code": "assumed_base_servings",
            "severity": "warning",
            "message": (
                "Original servings unknown; scaled from assumed base of 4 to 6."
            ),
        }
    ]


def test_resolve_scale_factor_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="target_servings must be greater than zero"):
        resolve_scale_factor({"value": 4}, target_servings=0)

    with pytest.raises(ValueError, match="scale_factor must be greater than zero"):
        resolve_scale_factor({"value": 4}, scale_factor=-1)

    with pytest.raises(ValueError, match="no stated servings"):
        resolve_scale_factor({"value": None}, target_servings=6)

    with pytest.raises(ValueError, match="no stated servings"):
        resolve_scale_factor({"value": 0}, target_servings=6)


@pytest.mark.parametrize(
    ("raw", "expected_value", "expected_suffix"),
    [
        ("2 cups", 2.0, " cups"),
        ("2,5 EI", 2.5, " EI"),
        ("3/4 tsp", 0.75, " tsp"),
        ("1 1/2 cups", 1.5, " cups"),
        ("½ cup", 0.5, " cup"),
        ("1½ cups", 1.5, " cups"),
        ("  2   cups  ", 2.0, " cups"),
    ],
)
def test_parse_quantity_supported_formats(
    raw: str, expected_value: float, expected_suffix: str
) -> None:
    assert parse_quantity(raw) == (expected_value, expected_suffix)


@pytest.mark.parametrize("raw", [None, "", "a pinch", "1/0 cup"])
def test_parse_quantity_unparseable(raw: str | None) -> None:
    assert parse_quantity(raw) == (None, raw or "")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (2.0, "2"),
        (2.5, "2.5"),
        (1.234, "1.23"),
        (1.236, "1.24"),
        (1 / 3, "0.33"),
    ],
)
def test_format_number(value: float, expected: str) -> None:
    assert format_number(value) == expected


def test_format_quantity_adds_suffix_with_expected_spacing() -> None:
    assert format_quantity(2.0, "") == "2"
    assert format_quantity(2.0, " cups") == "2 cups"
    assert format_quantity(2.0, "cups") == "2 cups"
    assert format_quantity(0.75, " tsp") == "0.75 tsp"


def test_scale_recipe_scales_ingredients_and_step_text() -> None:
    recipe = _recipe()
    original = copy.deepcopy(recipe)

    scaled = scale_recipe(recipe, target_servings=6)

    assert recipe == original
    assert scaled["servings"]["original_value"] == 4
    assert scaled["servings"]["value"] == 6
    assert scaled["servings"]["scale_factor"] == 1.5

    flour, salt, bowl = scaled["items"]
    assert flour["quantity"] == "3 cups"
    assert flour["original_quantity"] == "2 cups"
    assert flour["quote"] == "3 cups flour"
    assert salt["quantity"] == "0.75 tsp"
    assert salt["original_quantity"] == "1/2 tsp"
    assert salt["quote"] == "0.75 tsp salt"
    assert bowl["quantity"] == "1"
    assert "original_quantity" not in bowl

    assert scaled["steps"][0]["text"] == "Mix 3 cups flour."
    assert scaled["steps"][1]["text"] == "Add 0.75 tsp salt."
    assert scaled["warnings"] == [
        {
            "code": "existing_warning",
            "severity": "info",
            "message": "kept",
        }
    ]


def test_scale_recipe_reports_unscaled_and_unparseable_ingredients() -> None:
    recipe = _recipe()
    recipe["items"] = [
        {
            "id": "item-001",
            "type": "ingredient",
            "name": "salt",
            "quantity": None,
            "quote": "salt",
            "step_id": "step-001",
        },
        {
            "id": "item-002",
            "type": "ingredient",
            "name": "pepper",
            "quantity": "a pinch",
            "quote": "a pinch pepper",
            "step_id": "step-001",
        },
        {
            "id": "item-003",
            "type": "tool",
            "name": "spoon",
            "quantity": None,
            "quote": "a spoon",
            "step_id": "step-001",
        },
    ]

    scaled = scale_recipe(recipe, scale_factor=2.0)

    assert scaled["items"][0]["quantity"] is None
    assert scaled["items"][0]["original_quantity"] is None
    assert scaled["items"][1]["quantity"] == "a pinch"
    assert scaled["items"][1]["original_quantity"] == "a pinch"
    assert "original_quantity" not in scaled["items"][2]
    assert [warning["code"] for warning in scaled["warnings"]] == [
        "existing_warning",
        "unquantified_item_unscaled",
        "unparseable_quantity",
    ]
    assert scaled["warnings"][1]["item_id"] == "item-001"
    assert scaled["warnings"][1]["step_id"] == "step-001"
    assert scaled["warnings"][2]["item_id"] == "item-002"


def test_scale_recipe_warns_when_step_text_cannot_be_updated() -> None:
    recipe = _recipe()
    recipe["items"] = [
        {
            "id": "item-001",
            "type": "ingredient",
            "name": "flour",
            "quantity": "2 cups",
            "quote": "2 cups flour",
            "step_id": "step-001",
        }
    ]
    recipe["steps"] = [
        {
            "id": "step-001",
            "text": "Mix flour.",
            "quote": "Mix flour.",
            "item_ids": ["item-001"],
        }
    ]

    scaled = scale_recipe(recipe, scale_factor=1.5)

    assert scaled["steps"][0]["text"] == "Mix flour."
    assert [warning["code"] for warning in scaled["warnings"]] == [
        "existing_warning",
        "step_text_not_updated",
    ]
    assert scaled["warnings"][1]["step_id"] == "step-001"


def test_scale_recipe_updates_step_text_with_irregular_whitespace() -> None:
    recipe = _recipe()
    recipe["items"] = [
        {
            "id": "item-001",
            "type": "ingredient",
            "name": "flour",
            "quantity": "2 cups",
            "quote": "2   cups flour",
            "step_id": "step-001",
        }
    ]
    recipe["steps"] = [
        {
            "id": "step-001",
            "text": "Mix 2    cups flour.",
            "quote": "Mix 2    cups flour.",
            "item_ids": ["item-001"],
        }
    ]

    scaled = scale_recipe(recipe, scale_factor=1.5)

    assert scaled["steps"][0]["text"] == "Mix 3 cups flour."
    assert [warning["code"] for warning in scaled["warnings"]] == ["existing_warning"]


def test_scale_recipe_no_op_updates_only_servings_metadata() -> None:
    recipe = _recipe()
    recipe["items"] = [
        {
            "id": "item-001",
            "type": "ingredient",
            "name": "flour",
            "quantity": "2 cups",
            "quote": "2 cups flour",
            "step_id": "step-001",
        },
        {
            "id": "item-002",
            "type": "ingredient",
            "name": "salt",
            "quantity": None,
            "quote": "salt",
            "step_id": "step-001",
        },
    ]

    scaled = scale_recipe(recipe, scale_factor=1.0)

    assert scaled["servings"]["original_value"] == 4
    assert scaled["servings"]["value"] == 4
    assert scaled["servings"]["scale_factor"] == 1.0
    assert scaled["items"][0]["quantity"] == "2 cups"
    assert "original_quantity" not in scaled["items"][0]
    assert "original_quantity" not in scaled["items"][1]
    assert [warning["code"] for warning in scaled["warnings"]] == ["existing_warning"]


def test_scale_recipe_rounds_metadata_factor_but_scales_with_full_factor() -> None:
    recipe = _recipe()
    recipe["servings"]["value"] = 3
    recipe["items"] = [
        {
            "id": "item-001",
            "type": "ingredient",
            "name": "eggs",
            "quantity": "2",
            "quote": "2 eggs",
            "step_id": "step-001",
        }
    ]
    recipe["steps"] = [
        {
            "id": "step-001",
            "text": "Use 2 eggs.",
            "quote": "Use 2 eggs.",
            "item_ids": ["item-001"],
        }
    ]

    scaled = scale_recipe(recipe, target_servings=5)

    assert scaled["servings"]["value"] == 5
    assert scaled["servings"]["scale_factor"] == 1.6667
    assert scaled["items"][0]["quantity"] == "3.33"
    assert scaled["steps"][0]["text"] == "Use 3.33 eggs."
