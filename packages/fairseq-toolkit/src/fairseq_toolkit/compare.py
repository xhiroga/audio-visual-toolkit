from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from fairseq.data import dictionary as fairseq_dictionary


@dataclass
class TensorDiff:
    delta_norm: float
    base_norm: float
    new_norm: float
    relative_delta: float | None
    cosine_similarity: float | None
    max_abs_delta: float
    numel: int
    dtype: str
    shape: tuple[int, ...]

def _resolve_state(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        msg = f"Unsupported checkpoint payload: {type(payload)!r}"
        raise ValueError(msg)

    candidates: Iterable[str] = (
        "model",
        "state_dict",
        "model_state_dict",
        "params",
        "state",
    )
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _load_state(path: Path) -> dict[str, object]:
    torch.serialization.add_safe_globals([fairseq_dictionary.Dictionary])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _resolve_state(payload)


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        msg = f"Expected tensor, got {type(value)!r}"
        raise TypeError(msg)
    if value.device.type != "cpu":
        value = value.cpu()
    if value.dtype.is_floating_point:
        return value.to(dtype=torch.float64)
    return value.to(dtype=torch.float64)


def _compute_diff(base: torch.Tensor, new: torch.Tensor) -> TensorDiff:
    if base.shape != new.shape:
        msg = f"Shape mismatch: {tuple(base.shape)} vs {tuple(new.shape)}"
        raise ValueError(msg)

    base_f = _as_float_tensor(base)
    new_f = _as_float_tensor(new)
    delta = new_f - base_f

    delta_sq = float(torch.sum(delta * delta).item())
    base_sq = float(torch.sum(base_f * base_f).item())
    new_sq = float(torch.sum(new_f * new_f).item())
    delta_norm = math.sqrt(delta_sq)
    base_norm = math.sqrt(base_sq)
    new_norm = math.sqrt(new_sq)

    relative_delta = None if base_norm == 0.0 else delta_norm / base_norm

    cosine = None
    if base_norm > 0.0 and new_norm > 0.0:
        dot = float(torch.sum(base_f * new_f).item())
        cosine = dot / (base_norm * new_norm)
        cosine = max(min(cosine, 1.0), -1.0)

    max_abs_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0

    diff = TensorDiff(
        delta_norm=delta_norm,
        base_norm=base_norm,
        new_norm=new_norm,
        relative_delta=relative_delta,
        cosine_similarity=cosine,
        max_abs_delta=max_abs_delta,
        numel=delta.numel(),
        dtype=str(base.dtype),
        shape=tuple(base.shape),
    )
    return diff


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_keys(
    state: dict[str, object], prefixes: tuple[str, ...], *, role: str
) -> dict[str, tuple[str, object]]:
    normalized: dict[str, tuple[str, object]] = {}
    for original_name, value in state.items():
        canonical = original_name
        for prefix in prefixes:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix) :]
                break
        if canonical in normalized:
            other_name, _ = normalized[canonical]
            msg = (
                f"Key collision after removing prefixes for {role}: "
                f"{original_name!r} and {other_name!r} -> {canonical!r}"
            )
            raise ValueError(msg)
        normalized[canonical] = (original_name, value)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two fairseq checkpoints")
    parser.add_argument(
        "--model",
        nargs=2,
        metavar=("BASE", "UPDATED"),
        required=True,
        help="Paths to the two checkpoints to compare",
    )
    parser.add_argument(
        "--out-dir",
        default="./compare-out",
        help="Directory to write comparison artifacts",
    )
    parser.add_argument(
        "--remove-prefix",
        action="append",
        default=[],
        help="Prefixes to strip from parameter names before comparison",
    )
    args = parser.parse_args()

    base_path = Path(args.model[0]).expanduser().resolve()
    updated_path = Path(args.model[1]).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        raise SystemExit(f"Base checkpoint not found: {base_path}")
    if not updated_path.exists():
        raise SystemExit(f"Updated checkpoint not found: {updated_path}")

    base_state = _load_state(base_path)
    updated_state = _load_state(updated_path)

    remove_prefixes = tuple(args.remove_prefix or [])
    base_entries = _normalize_keys(base_state, remove_prefixes, role="base")
    updated_entries = _normalize_keys(updated_state, remove_prefixes, role="updated")

    all_keys = sorted(set(base_entries.keys()) | set(updated_entries.keys()))

    parameter_rows: list[dict[str, object]] = []

    common_tensor_count = 0
    only_in_base = 0
    only_in_updated = 0
    mismatched_types = 0
    shape_mismatches = 0

    for key in all_keys:
        base_entry = base_entries.get(key)
        updated_entry = updated_entries.get(key)

        if updated_entry is None:
            only_in_base += 1
            assert base_entry is not None
            base_name, base_value = base_entry
            base_shape = None
            base_dtype = None
            if isinstance(base_value, torch.Tensor):
                base_shape = tuple(base_value.shape)
                base_dtype = str(base_value.dtype)
            parameter_rows.append(
                {
                    "name": key,
                    "base_name": base_name,
                    "updated_name": None,
                    "status": "only_in_base",
                    "value_type": type(base_value).__name__ if base_value is not None else None,
                    "dtype": base_dtype,
                    "shape": base_shape,
                }
            )
            continue

        if base_entry is None:
            only_in_updated += 1
            assert updated_entry is not None
            updated_name, updated_value = updated_entry
            updated_shape = None
            updated_dtype = None
            if isinstance(updated_value, torch.Tensor):
                updated_shape = tuple(updated_value.shape)
                updated_dtype = str(updated_value.dtype)
            parameter_rows.append(
                {
                    "name": key,
                    "base_name": None,
                    "updated_name": updated_name,
                    "status": "only_in_updated",
                    "value_type": type(updated_value).__name__ if updated_value is not None else None,
                    "dtype": updated_dtype,
                    "shape": updated_shape,
                }
            )
            continue

        base_name, base_value = base_entry
        updated_name, updated_value = updated_entry

        if not isinstance(base_value, torch.Tensor) or not isinstance(updated_value, torch.Tensor):
            mismatched_types += 1
            parameter_rows.append(
                {
                    "name": key,
                    "base_name": base_name,
                    "updated_name": updated_name,
                    "status": "non_tensor",
                    "base_type": type(base_value).__name__ if base_value is not None else None,
                    "updated_type": type(updated_value).__name__ if updated_value is not None else None,
                    "base_value": repr(base_value) if base_value is not None else None,
                    "updated_value": repr(updated_value) if updated_value is not None else None,
                }
            )
            continue

        try:
            diff = _compute_diff(base_value, updated_value)
        except ValueError as exc:
            shape_mismatches += 1
            parameter_rows.append(
                {
                    "name": key,
                    "base_name": base_name,
                    "updated_name": updated_name,
                    "status": "shape_mismatch",
                    "message": str(exc),
                    "base_shape": tuple(base_value.shape),
                    "updated_shape": tuple(updated_value.shape),
                }
            )
            continue

        common_tensor_count += 1
        parameter_rows.append(
            {
                "name": key,
                "base_name": base_name,
                "updated_name": updated_name,
                "status": "matched",
                "delta_l2": diff.delta_norm,
                "baseline_l2": diff.base_norm,
                "updated_l2": diff.new_norm,
                "relative_delta": diff.relative_delta,
                "cosine_similarity": diff.cosine_similarity,
                "max_abs_delta": diff.max_abs_delta,
                "numel": diff.numel,
                "dtype": diff.dtype,
                "shape": diff.shape,
            }
        )

    parameter_fieldnames = [
        "name",
        "base_name",
        "updated_name",
        "status",
        "delta_l2",
        "baseline_l2",
        "updated_l2",
        "relative_delta",
        "cosine_similarity",
        "max_abs_delta",
        "numel",
        "dtype",
        "shape",
        "value_type",
        "base_type",
        "updated_type",
        "base_value",
        "updated_value",
        "message",
        "base_shape",
        "updated_shape",
    ]

    _write_csv(out_dir / "parameters.csv", parameter_rows, parameter_fieldnames)

    print(f"Compared {common_tensor_count} shared tensor parameters")
    print(f"Only in base: {only_in_base}")
    print(f"Only in updated: {only_in_updated}")
    print(f"Non-tensor entries: {mismatched_types}")
    print(f"Shape mismatches: {shape_mismatches}")
    print(f"Parameter metrics written to {out_dir / 'parameters.csv'}")


if __name__ == "__main__":
    main()
