import argparse

import logging
from pathlib import Path
from typing import Iterable

from .constants import PHONEMES_JA_JP_OPEN_JTALK


logger = logging.getLogger(__name__)


class _Entry(tuple):
    # (lineno, raw, parts_ok, digits_ok, phoneme_ok, start, end, phoneme)
    __slots__ = ()
    lineno: int
    raw: str
    parts_ok: bool
    digits_ok: bool
    phoneme_ok: bool
    start: str | None
    end: str | None
    phoneme: str | None


def _parse_lines(lines: Iterable[str]) -> tuple[list[_Entry], bool]:
    """Parse non-empty lines.

    Returns (entries, had_error).
    Each entry keeps flags so subsequent validators can still run
    and log multiple errors even if some checks fail.
    """
    entries: list[_Entry] = []
    had_error = False
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        parts_ok = len(parts) == 3
        start = end = phoneme = None
        digits_ok = False
        phoneme_ok = False

        if not parts_ok:
            logger.warning(
                "line=%d content=%r error=%s",
                lineno,
                raw,
                "must have exactly 3 space-separated fields",
            )
            had_error = True
        else:
            start, end, phoneme = parts
            digits_ok = start.isdigit() and end.isdigit()
            if not digits_ok:
                logger.warning(
                    "line=%d content=%r error=%s",
                    lineno,
                    raw,
                    "first and second fields must be digits",
                )
                had_error = True
            phoneme_ok = phoneme in PHONEMES_JA_JP_OPEN_JTALK
            if not phoneme_ok:
                logger.warning(
                    "line=%d content=%r error=%s",
                    lineno,
                    raw,
                    "third field must be a valid Japanese phoneme",
                )
                had_error = True

        entries.append(
            _Entry((lineno, raw, parts_ok, digits_ok, phoneme_ok, start, end, phoneme))
        )

    return entries, had_error


def validate_label_file(path: str | Path) -> bool:
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

    entries, had_error = _parse_lines(text.splitlines())
    if not entries:
        return False

    # numeric checks for lines with valid digits
    nums: list[tuple[int, int, int, str]] = []  # (start, end, lineno, raw)
    for lineno, raw, parts_ok, digits_ok, _phoneme_ok, start, end, _ph in entries:
        if digits_ok and parts_ok and start is not None and end is not None:
            nums.append((int(start), int(end), lineno, raw))

    # per-line strict order: start < end
    for s, e, lineno, raw in nums:
        if not (s < e):
            logger.warning(
                "line=%d content=%r error=%s",
                lineno,
                raw,
                "start must be less than end",
            )
            had_error = True

    # adjacency check and global monotonic increase of end times
    for i, ((s, e, _, _), (ns, ne, next_lineno, next_raw)) in enumerate(
        zip(nums, nums[1:])
    ):
        if e != ns:
            logger.warning(
                "line=%d content=%r error=%s",
                next_lineno,
                next_raw,
                f"start must equal previous end (prev_end={e}, start={ns})",
            )
            had_error = True
        if not (e < ne):
            logger.warning(
                "line=%d content=%r error=%s",
                next_lineno,
                next_raw,
                f"end must be greater than previous end (prev_end={e}, end={ne})",
            )
            had_error = True

    return not had_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate label files (.lab)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label-file", type=str, help="Path to a label file (.lab)")
    group.add_argument(
        "--label-dir",
        type=str,
        help="Path to a directory; recursively validate all .lab files",
    )
    args = parser.parse_args()

    targets: list[Path]
    if args.label_file:
        targets = [Path(args.label_file)]
    else:
        base = Path(args.label_dir)
        targets = sorted(base.rglob("*.lab"))

    if not targets:
        print("No .lab files found")
        raise SystemExit(1)

    any_fail = False
    for p in targets:
        ok = validate_label_file(p)
        print(f"{p}: {'True' if ok else 'False'}")
        if not ok:
            any_fail = True

    raise SystemExit(0 if not any_fail else 1)
