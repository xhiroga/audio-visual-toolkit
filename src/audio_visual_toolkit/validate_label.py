from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .constants import PHONEMES_JA_JP


def _parse_lines(lines: Iterable[str]) -> list[tuple[str, str, str]]:
    parsed: list[tuple[str, str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            return []
        start, end, phoneme = parts
        if not (start.isdigit() and end.isdigit()):
            return []
        if phoneme not in PHONEMES_JA_JP:
            return []
        parsed.append((start, end, phoneme))
    return parsed


def validate_label(path: str | Path) -> bool:
    """Validate a label file.

    Rules:
    - Each non-empty line has exactly three space-separated fields.
    - First and second fields are numeric (digits only).
    - Third field is a valid Japanese phoneme (PHONEMES_JA_JP).
    - The second number of a line equals the first number of the next line.
    """

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return False

    rows = _parse_lines(text.splitlines())
    if not rows:
        return False

    # adjacency check: end[i] == start[i+1]
    for (start, end, _), (next_start, _, _) in zip(rows, rows[1:]):
        if end != next_start:
            return False

    return True

