import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from .constants import MFA_V3_TO_PHONEMES_JA_JP_DICTIONARY

# HTK-style .lab units: 10,000,000 ticks per second (100 ns resolution)
HTK_TICKS_PER_SEC = 10_000_000


def _iter_json_files(base: Path) -> Iterable[Path]:
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def _map_phone_to_phonemes(label: str) -> Tuple[str, ...]:
    mapped = MFA_V3_TO_PHONEMES_JA_JP_DICTIONARY.get(label)
    if mapped is None:
        # Unknown phone: fall back to a pause to keep timing contiguous
        return ("pau",)
    if len(mapped) == 0:
        # Explicit empty mapping: also fall back to pause of same duration
        return ("pau",)
    return mapped


def _split_ticks_evenly(start_tick: int, end_tick: int, parts: int) -> List[Tuple[int, int]]:
    total = end_tick - start_tick
    if parts <= 1 or total <= 0:
        return [(start_tick, end_tick)]
    q, r = divmod(total, parts)
    out: List[Tuple[int, int]] = []
    s = start_tick
    for i in range(parts):
        inc = q + (1 if i < r else 0)
        e = s + inc
        out.append((s, e))
        s = e
    return out


def convert_alignment_json_to_lab(input_json: Path, output_lab: Path) -> bool:
    try:
        data = json.loads(input_json.read_text(encoding="utf-8"))
    except Exception:
        return False

    # MFA JSON structure assumptions
    try:
        global_start_s: float = float(data.get("start", 0.0))
        global_end_s: float = float(data.get("end", 0.0))
        tiers = data["tiers"]
        phones = tiers["phones"]
        entries = phones["entries"]
    except Exception:
        return False

    # Sort by start time to be safe
    entries = sorted(entries, key=lambda x: (x[0], x[1]))

    # Build segments as (start_tick, end_tick, phoneme)
    segs: List[Tuple[int, int, str]] = []
    prev_tick = int(round(global_start_s * HTK_TICKS_PER_SEC))
    global_end_tick = int(round(global_end_s * HTK_TICKS_PER_SEC))

    for s_s, e_s, label in entries:
        s_tick = int(round(float(s_s) * HTK_TICKS_PER_SEC))
        e_tick = int(round(float(e_s) * HTK_TICKS_PER_SEC))

        # Fill leading gap with silence
        if s_tick > prev_tick:
            segs.append((prev_tick, s_tick, "sil"))
            prev_tick = s_tick

        # Map phone label to one or more OpenJTalk phonemes
        mapped = _map_phone_to_phonemes(str(label))
        splits = _split_ticks_evenly(prev_tick, e_tick, len(mapped))
        for (ps, pe), ph in zip(splits, mapped):
            if pe > ps:
                segs.append((ps, pe, ph))
                prev_tick = pe

        # If mapping yielded fewer durations (shouldn't), ensure prev is at least e_tick
        if prev_tick < e_tick:
            segs.append((prev_tick, e_tick, "pau"))
            prev_tick = e_tick

    # Trailing silence to global end
    if global_end_tick > prev_tick:
        segs.append((prev_tick, global_end_tick, "sil"))
        prev_tick = global_end_tick

    # Write .lab
    output_lab.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_lab.open("w", encoding="utf-8", newline="") as f:
            for s, e, ph in segs:
                f.write(f"{s} {e} {ph}\n")
    except Exception:
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MFA alignment JSON to .lab")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json-file", type=str, help="Path to an MFA JSON file")
    group.add_argument(
        "--json-dir",
        type=str,
        help="Directory containing MFA JSON files (non-recursive)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write .lab outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.json_file:
        inp = Path(args.json_file)
        if not inp.exists():
            print(f"Not found: {inp}")
            raise SystemExit(1)
        out = out_dir / (inp.stem + ".lab")
        ok = convert_alignment_json_to_lab(inp, out)
        if not ok:
            print(f"Failed: {inp}")
        raise SystemExit(0 if ok else 2)
    else:
        base = Path(args.json_dir)
        if not base.exists() or not base.is_dir():
            print(f"Not found or not a directory: {base}")
            raise SystemExit(1)
        any_fail = False
        for p in sorted(_iter_json_files(base)):
            out = out_dir / (p.stem + ".lab")
            ok = convert_alignment_json_to_lab(p, out)
            print(f"{p} -> {out}: {'OK' if ok else 'FAIL'}")
            if not ok:
                any_fail = True
        raise SystemExit(0 if not any_fail else 2)


if __name__ == "__main__":
    main()

