from __future__ import annotations

import argparse
from pathlib import Path

from src.extract.pipeline import extract
from src.ingest import ingest
from src.manipulate.pipeline import manipulate
from src.output.pipeline import export_recipe

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingest, extract, manipulate, and output recipe pipeline."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input file paths. Defaults to Recipe.pdf in the project root.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "results" / "output.json"),
        help="Path to write structured JSON output.",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        default=str(ROOT / "results" / "recipe.md"),
        help="Path to write formatted Markdown recipe.",
    )
    parser.add_argument(
        "-w",
        "--warnings",
        default=str(ROOT / "results" / "warnings.txt"),
        help="Path to write warnings report.",
    )
    parser.add_argument(
        "--servings",
        type=int,
        default=None,
        help="Target number of servings to scale the recipe to.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Direct scale multiplier (e.g., 2.0 to double the recipe).",
    )
    parser.add_argument(
        "--base-servings",
        type=int,
        default=None,
        help="Assumed original servings when the recipe does not state a serving count.",
    )
    parser.add_argument(
        "--vegan",
        action="store_true",
        help="Convert animal-derived ingredients and affected steps to vegan alternatives.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_paths = [Path(p) for p in args.inputs] if args.inputs else [ROOT / "Recipe.pdf"]
    source = input_paths[0] if len(input_paths) == 1 else input_paths
    source_label = ", ".join(p.name for p in input_paths)

    ingested = ingest(source)
    extracted = extract(ingested)

    processed = manipulate(
        extracted,
        target_servings=args.servings,
        scale_factor=args.scale,
        assumed_base_servings=args.base_servings,
        make_vegan=args.vegan,
    )

    ingest_warnings = [
        {"code": w.code, "page": w.page, "message": w.message}
        for w in ingested.warnings
    ]

    export_recipe(
        processed,
        source_label=source_label,
        page_count=len(ingested.pages),
        ingest_warnings=ingest_warnings,
        json_path=Path(args.output),
        markdown_path=Path(args.markdown),
        warnings_path=Path(args.warnings),
        make_vegan=args.vegan,
    )

    print(
        f"pages={len(ingested.pages)} steps={len(processed.get('steps', []))} "
        f"items={len(processed.get('items', []))}"
    )
    print(
        f"warnings: ingest={len(ingest_warnings)} "
        f"pipeline={len(processed.get('warnings', []))}"
    )


if __name__ == "__main__":
    main()