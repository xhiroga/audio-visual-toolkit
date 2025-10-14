from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
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


def _format_float(value: float | None, precision: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{precision}g}"


def _write_html_report(
    df: pd.DataFrame,
    path: Path,
    *,
    model1_column: str,
    model2_column: str,
) -> None:
    numeric_cols = [
        "delta_l2",
        "baseline_l2",
        "updated_l2",
        "relative_delta",
        "cosine_similarity",
        "max_abs_delta",
        "numel",
    ]

    present_cols = [col for col in numeric_cols if col in df.columns]
    for col in present_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    styler = df.style.format(
        {
            "delta_l2": _format_float,
            "baseline_l2": _format_float,
            "updated_l2": _format_float,
            "relative_delta": _format_float,
            "cosine_similarity": _format_float,
            "max_abs_delta": _format_float,
        }
    )

    if "relative_delta" in df:
        styler = styler.bar(subset=["relative_delta"], color="#ffb347")
    if "cosine_similarity" in df:
        styler = styler.background_gradient(subset=["cosine_similarity"], cmap="RdYlGn")

    html = styler.to_html()
    template = f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Checkpoint Parameter Comparison</title>
    <style>
      body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; margin: 2rem; }}
      h1 {{ margin-bottom: 1rem; }}
      .summary {{ margin-bottom: 1.5rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; }}
      th {{ background: #f8f8f8; position: sticky; top: 0; }}
    </style>
  </head>
  <body>
    <h1>Checkpoint Parameter Comparison</h1>
    <p class=\"summary\">Columns: {model1_column}, {model2_column}</p>
    {html}
  </body>
</html>
"""
    path.write_text(template, encoding="utf-8")


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
        "--model-1",
        required=True,
        help="Checkpoint A (e.g. pretrained)",
    )
    parser.add_argument(
        "--model-2",
        required=True,
        help="Checkpoint B (e.g. finetuned)",
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

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model1_path = Path(args.model_1).expanduser().resolve()
    model2_path = Path(args.model_2).expanduser().resolve()

    if not model1_path.exists():
        raise SystemExit(f"Checkpoint not found: {model1_path}")
    if not model2_path.exists():
        raise SystemExit(f"Checkpoint not found: {model2_path}")

    model1_state = _load_state(model1_path)
    model2_state = _load_state(model2_path)

    model1_column = f"model 1 ({model1_path.name})"
    model2_column = f"model 2 ({model2_path.name})"

    remove_prefixes = tuple(args.remove_prefix or [])
    model1_entries = _normalize_keys(model1_state, remove_prefixes, role="model_1")
    model2_entries = _normalize_keys(model2_state, remove_prefixes, role="model_2")

    all_keys = sorted(set(model1_entries.keys()) | set(model2_entries.keys()))

    parameter_rows: list[dict[str, object]] = []

    common_tensor_count = 0
    only_in_model1 = 0
    only_in_model2 = 0
    mismatched_types = 0
    shape_mismatches = 0

    for key in all_keys:
        model1_entry = model1_entries.get(key)
        model2_entry = model2_entries.get(key)

        if model2_entry is None:
            only_in_model1 += 1
            assert model1_entry is not None
            model1_name, model1_value = model1_entry
            model1_shape = None
            model1_dtype = None
            if isinstance(model1_value, torch.Tensor):
                model1_shape = tuple(model1_value.shape)
                model1_dtype = str(model1_value.dtype)
            parameter_rows.append(
                {
                    "name": key,
                    model1_column: model1_name,
                    model2_column: None,
                    "dtype": model1_dtype,
                    "shape": model1_shape,
                    "message": "missing from model 2",
                }
            )
            continue

        if model1_entry is None:
            only_in_model2 += 1
            assert model2_entry is not None
            model2_name, model2_value = model2_entry
            model2_shape = None
            model2_dtype = None
            if isinstance(model2_value, torch.Tensor):
                model2_shape = tuple(model2_value.shape)
                model2_dtype = str(model2_value.dtype)
            parameter_rows.append(
                {
                    "name": key,
                    model1_column: None,
                    model2_column: model2_name,
                    "dtype": model2_dtype,
                    "shape": model2_shape,
                    "message": "missing from model 1",
                }
            )
            continue

        model1_name, model1_value = model1_entry
        model2_name, model2_value = model2_entry

        if not isinstance(model1_value, torch.Tensor) or not isinstance(model2_value, torch.Tensor):
            mismatched_types += 1
            parameter_rows.append(
                {
                    "name": key,
                    model1_column: model1_name,
                    model2_column: model2_name,
                    "message": (
                        f"non-tensor entry ({type(model1_value).__name__} vs {type(model2_value).__name__})"
                    ),
                }
            )
            continue

        try:
            diff = _compute_diff(model1_value, model2_value)
        except ValueError as exc:
            shape_mismatches += 1
            parameter_rows.append(
                {
                    "name": key,
                    model1_column: model1_name,
                    model2_column: model2_name,
                    "message": str(exc),
                }
            )
            continue

        common_tensor_count += 1
        parameter_rows.append(
            {
                "name": key,
                model1_column: model1_name,
                model2_column: model2_name,
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
        model1_column,
        model2_column,
        "delta_l2",
        "baseline_l2",
        "updated_l2",
        "relative_delta",
        "cosine_similarity",
        "max_abs_delta",
        "numel",
        "dtype",
        "shape",
        "message",
    ]

    parameters_path = out_dir / "parameters.csv"
    html_path = out_dir / "parameters.html"

    _write_csv(parameters_path, parameter_rows, parameter_fieldnames)

    try:
        df = pd.read_csv(parameters_path)
    except Exception as exc:  # pragma: no cover - propagate read errors
        print(f"Failed to rebuild DataFrame for HTML output: {exc}")
    else:
        _write_html_report(
            df,
            html_path,
            model1_column=model1_column,
            model2_column=model2_column,
        )

    print(f"Compared {common_tensor_count} shared tensor parameters")
    print(f"Only in model 1: {only_in_model1}")
    print(f"Only in model 2: {only_in_model2}")
    print(f"Non-tensor entries: {mismatched_types}")
    print(f"Shape mismatches: {shape_mismatches}")
    print(f"Parameter metrics written to {parameters_path}")
    if html_path.exists():
        print(f"HTML report written to {html_path}")


if __name__ == "__main__":
    main()
