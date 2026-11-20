from __future__ import annotations

import copy
from typing import Any


def verify_veganized_recipe(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    verified = copy.deepcopy(after)
    warnings: list[dict[str, Any]] = list(verified.get("warnings", []))

    before_items = {
        item["id"]: item for item in before.get("items", []) if "id" in item
    }
    after_items = verified.get("items", [])
    after_item_ids = {item.get("id") for item in after_items}

    missing_items = sorted(set(before_items) - after_item_ids)
    if missing_items:
        warnings.append(
            {
                "code": "vegan_missing_items",
                "severity": "error",
                "message": f"Items dropped during veganization: {', '.join(missing_items)}",
            }
        )

    substitutions_by_step: dict[str, list[dict[str, Any]]] = {}
    plant_items_by_step: dict[str, list[dict[str, Any]]] = {}

    for item in after_items:
        iid = item.get("id")
        orig = before_items.get(iid) if isinstance(iid, str) else None
        if orig is None:
            warnings.append(
                {
                    "code": "vegan_hallucinated_item",
                    "severity": "warning",
                    "message": f"Unexpected item '{iid}' added during veganization.",
                    "item_id": iid,
                }
            )
            continue

        sub = item.get("vegan_substitution")
        step_id = item.get("step_id", "")

        if not sub:
            if item.get("name") != orig.get("name") or item.get("quantity") != orig.get("quantity"):
                warnings.append(
                    {
                        "code": "vegan_untouched_item_modified",
                        "severity": "warning",
                        "message": f"Non-animal item '{orig.get('name')}' was unexpectedly modified and has been restored.",
                        "item_id": iid,
                        "step_id": step_id,
                    }
                )
                item["name"] = orig.get("name")
                item["quantity"] = orig.get("quantity")
                item["preparation"] = orig.get("preparation")
            if item.get("type") == "ingredient":
                plant_items_by_step.setdefault(step_id, []).append(item)
        else:
            substitutions_by_step.setdefault(step_id, []).append(
                {
                    "item_id": iid,
                    "original_name": sub.get("original_name", ""),
                    "substitute_name": item.get("name", ""),
                }
            )

    before_steps = {
        step["id"]: step for step in before.get("steps", []) if "id" in step
    }

    for step in verified.get("steps", []):
        sid = step.get("id", "")
        orig_step = before_steps.get(sid)
        if orig_step is None:
            continue

        current_text = step.get("text", "")
        orig_text = orig_step.get("text", "")
        step_subs = substitutions_by_step.get(sid, [])
        step_plants = plant_items_by_step.get(sid, [])

        if current_text == orig_text and not step_subs:
            continue

        current_lower = current_text.casefold()

        for sub_info in step_subs:
            old_name = (sub_info["original_name"] or "").strip()
            new_name = (sub_info["substitute_name"] or "").strip()

            if new_name and new_name.casefold() not in current_lower:
                warnings.append(
                    {
                        "code": "vegan_substitute_missing_in_step",
                        "severity": "warning",
                        "message": f"Substitute '{new_name}' not found in rewritten step text.",
                        "step_id": sid,
                        "item_id": sub_info["item_id"],
                    }
                )

            if (
                old_name
                and old_name.casefold() != new_name.casefold()
                and old_name.casefold() in current_lower
            ):
                warnings.append(
                    {
                        "code": "vegan_animal_term_remaining",
                        "severity": "warning",
                        "message": f"Original animal ingredient '{old_name}' still appears in rewritten step.",
                        "step_id": sid,
                        "item_id": sub_info["item_id"],
                    }
                )

        for plant_item in step_plants:
            p_name = (plant_item.get("name") or "").strip()
            if p_name and p_name.casefold() in orig_text.casefold():
                if p_name.casefold() not in current_lower:
                    warnings.append(
                        {
                            "code": "vegan_plant_ingredient_dropped",
                            "severity": "warning",
                            "message": f"Plant-based ingredient '{p_name}' disappeared from rewritten step.",
                            "step_id": sid,
                            "item_id": plant_item.get("id"),
                        }
                    )

    verified["warnings"] = warnings
    return verified