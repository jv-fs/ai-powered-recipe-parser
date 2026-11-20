from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def merge_crops(top_text: str, bottom_text: str) -> str:
    top_words = top_text.split()
    bottom_words = bottom_text.split()

    if not top_words:
        return " ".join(bottom_words)
    if not bottom_words:
        return " ".join(top_words)

    matcher = SequenceMatcher(None, top_words, bottom_words)
    match = matcher.find_longest_match(0, len(top_words), 0, len(bottom_words))

    if match.size >= 3:
        merged = top_words[: match.a] + bottom_words[match.b :]
        return " ".join(merged)

    return " ".join(top_words + bottom_words)


def verify_transcript(
    full_text: str,
    top_text: str,
    bottom_text: str,
    page_index: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    merged_crops = merge_crops(top_text, bottom_text)

    full_words = full_text.split()
    crop_words = merged_crops.split()

    if not full_words and not crop_words:
        return "", []
    if not full_words:
        return merged_crops, [
            {
                "code": "full_transcript_empty",
                "severity": "warning",
                "message": "Full-page reading was empty; used merged crop readings.",
                "page": page_index,
                "full_reading": "",
                "crop_reading": merged_crops,
            }
        ]
    if not crop_words:
        return full_text.strip(), [
            {
                "code": "crop_transcript_empty",
                "severity": "warning",
                "message": "Crop readings were empty; kept full-page reading.",
                "page": page_index,
                "full_reading": full_text.strip(),
                "crop_reading": "",
            }
        ]

    matcher = SequenceMatcher(None, full_words, crop_words)
    final_words: list[str] = []
    warnings: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        full_span = " ".join(full_words[i1:i2])
        crop_span = " ".join(crop_words[j1:j2])

        if tag == "equal":
            final_words.extend(full_words[i1:i2])

        elif tag == "replace":
            similarity = SequenceMatcher(
                None, full_span.casefold(), crop_span.casefold()
            ).ratio()

            if similarity >= 0.75:
                final_words.extend(crop_words[j1:j2])
                warnings.append(
                    {
                        "code": "crop_refined",
                        "severity": "info",
                        "message": "Minor character discrepancy resolved using zoomed crop.",
                        "page": page_index,
                        "full_reading": full_span,
                        "crop_reading": crop_span,
                    }
                )
            else:
                final_words.extend(full_words[i1:i2])
                warnings.append(
                    {
                        "code": "verification_failed",
                        "severity": "warning",
                        "message": "Significant discrepancy between full-page and crop readings.",
                        "page": page_index,
                        "full_reading": full_span,
                        "crop_reading": crop_span,
                    }
                )

        elif tag == "delete":
            final_words.extend(full_words[i1:i2])
            warnings.append(
                {
                    "code": "missing_in_crop",
                    "severity": "warning",
                    "message": "Segment present in full-page reading but absent in crops.",
                    "page": page_index,
                    "full_reading": full_span,
                    "crop_reading": "",
                }
            )

        elif tag == "insert":
            final_words.extend(crop_words[j1:j2])
            warnings.append(
                {
                    "code": "recovered_by_crop",
                    "severity": "warning",
                    "message": "Segment missed in full-page reading was recovered by crops.",
                    "page": page_index,
                    "full_reading": "",
                    "crop_reading": crop_span,
                }
            )

    return " ".join(final_words), warnings