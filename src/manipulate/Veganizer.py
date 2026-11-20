from __future__ import annotations

import copy
import json
from typing import Any

from openai import OpenAI

from .Scaler import format_quantity, parse_quantity

VEGAN_MULTIPLIERS: dict[str, float] = {
    "protein_meat": 1.5,
    "dairy_fat": 1.0,
    "egg": 1.0,
    "broth_stock": 1.0,
    "honey": 1.0,
    "other_animal": 1.0,
}

IDENTIFY_ITEMS_PROMPT = (
    "Analyze the provided list of recipe ingredients and identify ONLY those of animal origin "
    "(meat, poultry, fish, seafood, animal fat, dairy, eggs, honey, gelatin, meat stock/fond). "
    "For each animal-derived item, provide a suitable vegan culinary substitute in the EXACT same language as the ingredient. "
    "Return strict JSON with key \"replacements\": a list of objects with fields:\n"
    "- item_id: exact id from input\n"
    "- animal_origin_reason: brief reason why it is animal-derived\n"
    "- category: one of [\"protein_meat\", \"dairy_fat\", \"egg\", \"broth_stock\", \"honey\", \"other_animal\"]\n"
    "- substitute_name: vegan substitute name in the source language\n"
    "- substitute_preparation: adapted preparation in source language or null\n"
    "Do NOT include plant-based ingredients, water, salt, spices, or tools. "
    "If there are no animal-derived ingredients, return {\"replacements\": []}."
)

REWRITE_STEPS_PROMPT = (
    "Rewrite ONLY the provided recipe steps to use the specified vegan substitutes while preserving "
    "the exact source language, tone, culinary logic, and all existing plant-based ingredients and quantities. "
    "Adjust cooking actions only where required by the vegan substitute (for example, searing tofu/seitan instead of rendering animal fat). "
    "Do NOT translate, summarize, merge, or split steps. "
    "Return strict JSON with key \"steps\": a list of objects with fields:\n"
    "- step_id: exact id from input\n"
    "- text: updated step text in the source language"
)


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            return None, str(exc)
    return None, "invalid_json"


def _call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
    except Exception as exc:
        return None, str(exc)

    content = response.choices[0].message.content if response.choices else ""
    return _parse_json_response(content or "")


def veganize_recipe(
    recipe: dict[str, Any],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    ollama_base_url: str = "http://localhost:11434/v1",
    multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    vegan = copy.deepcopy(recipe)
    warnings: list[dict[str, Any]] = list(vegan.get("warnings", []))
    active_model = model or vegan.get("model") or "qwen2.5vl:7b"
    active_client = client or OpenAI(base_url=ollama_base_url, api_key="ollama")
    active_multipliers = {**VEGAN_MULTIPLIERS, **(multipliers or {})}

    steps_by_id = {s["id"]: s for s in vegan.get("steps", []) if "id" in s}
    ingredients_payload = [
        {
            "item_id": item["id"],
            "step_id": item.get("step_id"),
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "unit": item.get("unit"),
            "preparation": item.get("preparation"),
            "step_context": steps_by_id.get(item.get("step_id", ""), {}).get("text", ""),
        }
        for item in vegan.get("items", [])
        if item.get("type") == "ingredient"
    ]

    if not ingredients_payload:
        vegan["warnings"] = warnings
        return vegan

    id_response, err = _call_llm_json(
        active_client,
        active_model,
        IDENTIFY_ITEMS_PROMPT,
        {"ingredients": ingredients_payload},
    )
    if err or id_response is None:
        warnings.append(
            {
                "code": "vegan_identification_failed",
                "severity": "error",
                "message": f"Failed to identify animal ingredients: {err or 'invalid_json'}",
            }
        )
        vegan["warnings"] = warnings
        return vegan

    raw_replacements = (
        id_response.get("replacements")
        if isinstance(id_response.get("replacements"), list)
        else []
    )
    replacements_by_id: dict[str, dict[str, Any]] = {}
    for rep in raw_replacements:
        if not isinstance(rep, dict):
            continue
        iid = rep.get("item_id")
        sub_name = (rep.get("substitute_name") or "").strip()
        if isinstance(iid, str) and sub_name:
            replacements_by_id[iid] = rep

    applied_substitutions: list[dict[str, Any]] = []
    affected_step_ids: set[str] = set()

    for item in vegan.get("items", []):
        iid = item.get("id")
        if iid not in replacements_by_id:
            continue

        rep = replacements_by_id[iid]
        orig_name = item.get("name", "")
        orig_qty = item.get("quantity")
        orig_prep = item.get("preparation")
        category = rep.get("category")
        if category not in active_multipliers:
            category = "other_animal"
        factor = active_multipliers[category]

        new_qty = orig_qty
        if orig_qty is not None and factor != 1.0:
            num_val, suffix = parse_quantity(orig_qty)
            if num_val is not None:
                new_qty = format_quantity(num_val * factor, suffix)
            else:
                warnings.append(
                    {
                        "code": "vegan_quantity_not_scaled",
                        "severity": "warning",
                        "message": f"Could not apply vegan multiplier to '{orig_qty}' for '{orig_name}'.",
                        "item_id": iid,
                        "step_id": item.get("step_id"),
                    }
                )

        new_name = str(rep["substitute_name"]).strip()
        new_prep = rep.get("substitute_preparation")
        if isinstance(new_prep, str):
            new_prep = new_prep.strip() or None
        else:
            new_prep = orig_prep

        item["vegan_substitution"] = {
            "original_name": orig_name,
            "original_quantity": orig_qty,
            "original_preparation": orig_prep,
            "animal_origin_reason": rep.get("animal_origin_reason"),
            "category": category,
            "multiplier": factor,
        }
        item["name"] = new_name
        item["quantity"] = new_qty
        item["preparation"] = new_prep

        step_id = item.get("step_id")
        if isinstance(step_id, str):
            affected_step_ids.add(step_id)

        applied_substitutions.append(
            {
                "item_id": iid,
                "step_id": step_id,
                "original_name": orig_name,
                "substitute_name": new_name,
                "original_quantity": orig_qty,
                "substitute_quantity": new_qty,
            }
        )

    if not applied_substitutions:
        vegan["warnings"] = warnings
        return vegan

    old_names_lower = [
        sub["original_name"].casefold()
        for sub in applied_substitutions
        if sub["original_name"]
    ]
    for step in vegan.get("steps", []):
        sid = step.get("id")
        step_text_lower = (step.get("text") or "").casefold()
        if any(old_name in step_text_lower for old_name in old_names_lower):
            if isinstance(sid, str):
                affected_step_ids.add(sid)

    steps_to_rewrite = [
        {"step_id": step["id"], "text": step.get("text", "")}
        for step in vegan.get("steps", [])
        if step.get("id") in affected_step_ids
    ]

    if steps_to_rewrite:
        step_response, step_err = _call_llm_json(
            active_client,
            active_model,
            REWRITE_STEPS_PROMPT,
            {
                "substitutions": applied_substitutions,
                "steps": steps_to_rewrite,
            },
        )
        if step_err or step_response is None:
            warnings.append(
                {
                    "code": "vegan_step_rewrite_failed",
                    "severity": "error",
                    "message": f"Failed to rewrite affected steps: {step_err or 'invalid_json'}",
                }
            )
        else:
            raw_steps = (
                step_response.get("steps")
                if isinstance(step_response.get("steps"), list)
                else []
            )
            rewritten_by_id = {
                s["step_id"]: s["text"].strip()
                for s in raw_steps
                if isinstance(s, dict)
                and isinstance(s.get("step_id"), str)
                and isinstance(s.get("text"), str)
                and s["text"].strip()
            }
            for step in vegan.get("steps", []):
                sid = step.get("id")
                if sid in rewritten_by_id:
                    step["original_text"] = step.get("text")
                    step["text"] = rewritten_by_id[sid]

    vegan["warnings"] = warnings
    return vegan