from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from fairseq.data import dictionary as fairseq_dictionary  # type: ignore


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


@dataclass
class LoraFactors:
    name: str
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    alpha: float | None = None
    scaling: float | None = None
    down_key: str | None = None
    up_key: str | None = None


_LORA_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_A.default.weight", "down"),
    (".lora_B.default.weight", "up"),
    (".lora_down.default.weight", "down"),
    (".lora_up.default.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_alpha", "alpha"),
    (".alpha", "alpha"),
    (".lora_scaling", "scaling"),
    (".scaling", "scaling"),
)


def _resolve_state(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        candidates: Iterable[str] = (
            "model",
            "state_dict",
            "model_state_dict",
            "params",
            "state",
        )
        for key in candidates:
            value = payload.get(key)  # type: ignore[arg-type]
            if isinstance(value, dict):
                return value
        return payload
    msg = f"Unsupported checkpoint payload: {type(payload)!r}"
    raise ValueError(msg)


def _load_json_state(path: Path) -> dict[str, torch.Tensor]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    maybe_state = data.get("model") if isinstance(data, dict) else None
    if not isinstance(maybe_state, dict):
        msg = f"JSON checkpoint missing 'model' dict: {path}"
        raise ValueError(msg)
    state: dict[str, torch.Tensor] = {}
    for key, value in maybe_state.items():
        if isinstance(value, dict) and value.get("__tensor__"):
            msg = (
                "JSON checkpoint only records tensor metadata without raw values; "
                "convert to a torch or safetensors checkpoint."
            )
            raise ValueError(msg)
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
            state[key] = tensor
    return state


def _load_generic_state(path: Path) -> dict[str, torch.Tensor]:
    torch.serialization.add_safe_globals([fairseq_dictionary.Dictionary])
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, ValueError) as exc:
        if path.suffix.lower() == ".json":
            return _load_json_state(path)
        raise ValueError(f"Failed to load checkpoint {path}: {exc}") from exc
    state = _resolve_state(payload)
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
    return result


def _load_llama_state(identifier: str) -> dict[str, torch.Tensor]:
    path = Path(identifier).expanduser()
    if path.exists():
        return _load_generic_state(path)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "transformers is required to load remote models; install it or provide a local checkpoint"
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(identifier, device_map="cpu")
    try:
        state = {name: param.detach().cpu() for name, param in model.named_parameters()}  # type: ignore[arg-type]
    finally:
        del model
    return state


def _load_lora_state(path: Path) -> dict[str, torch.Tensor]:
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"LoRA checkpoint not found: {path}")
    lower = path.suffix.lower()
    if lower == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "safetensors is required to load .safetensors adapters"
            ) from exc
        return load_file(str(path))
    return _load_generic_state(path)


def _to_scalar(value: object) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return None
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _collect_lora_modules(state: dict[str, torch.Tensor]) -> dict[str, LoraFactors]:
    modules: dict[str, LoraFactors] = {}
    for key, tensor in state.items():
        suffix_kind = None
        suffix = None
        for candidate, kind in _LORA_SUFFIXES:
            if key.endswith(candidate):
                suffix_kind = kind
                suffix = candidate
                break
        if suffix_kind is None or suffix is None:
            continue
        prefix = key[: -len(suffix)]
        entry = modules.setdefault(prefix, LoraFactors(name=prefix))
        if suffix_kind == "down":
            entry.down = tensor.detach().cpu()
            entry.down_key = key
        elif suffix_kind == "up":
            entry.up = tensor.detach().cpu()
            entry.up_key = key
        elif suffix_kind == "alpha":
            entry.alpha = _to_scalar(tensor)
        elif suffix_kind == "scaling":
            entry.scaling = _to_scalar(tensor)
    return modules


def _candidate_base_keys(prefix: str) -> list[str]:
    candidates = [f"{prefix}.weight"]
    stripped = prefix
    for token in (
        "base_model.model.",
        "base_model.",
        "llama.",
        "model.",
        "modules_to_save.",
    ):
        if stripped.startswith(token):
            candidates.append(f"{stripped[len(token):]}.weight")
    return candidates


def _find_base_tensor(
    prefix: str, base_state: dict[str, torch.Tensor]
) -> tuple[str, torch.Tensor] | None:
    candidates = _candidate_base_keys(prefix)
    for cand in candidates:
        tensor = base_state.get(cand)
        if isinstance(tensor, torch.Tensor):
            return cand, tensor
    for cand in candidates:
        for key, tensor in base_state.items():
            if key.endswith(cand) and isinstance(tensor, torch.Tensor):
                return key, tensor
    return None


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
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


def _materialize_delta(entry: LoraFactors) -> tuple[torch.Tensor, dict[str, object]]:
    if entry.down is None or entry.up is None:
        missing = []
        if entry.down is None:
            missing.append("lora_down")
        if entry.up is None:
            missing.append("lora_up")
        msg = ", ".join(missing)
        raise ValueError(f"Missing LoRA components: {msg}")

    down = entry.down.to(dtype=torch.float64)
    up = entry.up.to(dtype=torch.float64)

    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("Only 2D LoRA factors are supported in the current implementation")
    if up.shape[1] != down.shape[0]:
        raise ValueError(
            f"Rank mismatch between LoRA up ({tuple(up.shape)}) and down ({tuple(down.shape)})"
        )

    rank = down.shape[0]
    alpha = entry.alpha if entry.alpha is not None else float(rank)
    scale = entry.scaling if entry.scaling is not None else alpha / max(rank, 1)

    delta = torch.matmul(up, down)
    delta = delta * float(scale)

    metadata = {
        "lora_rank": int(rank),
        "lora_alpha": float(alpha),
        "lora_scale": float(scale),
    }
    return delta, metadata


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
    base_column: str,
    adapter_column: str,
) -> None:
    numeric_cols = [
        "delta_l2",
        "baseline_l2",
        "updated_l2",
        "relative_delta",
        "cosine_similarity",
        "max_abs_delta",
        "delta_l1",
        "delta_density",
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
            "delta_l1": _format_float,
            "delta_density": _format_float,
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
    <title>LLaMA + LoRA Parameter Comparison</title>
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
    <h1>LLaMA + LoRA Parameter Comparison</h1>
    <p class=\"summary\">Baseline: {base_column} | Adapter: {adapter_column}</p>
    {html}
  </body>
</html>
"""
    path.write_text(template, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a base LLaMA checkpoint with a LoRA adapter")
    parser.add_argument("--llama-path", required=True, help="Hugging Face model ID or local checkpoint path")
    parser.add_argument("--lora-path", required=True, help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--out-dir", default="./llama-compare-out", help="Directory to write comparison artifacts")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows in the CSV/HTML report (sorted by delta L2)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_state = _load_llama_state(args.llama_path)
    adapter_state = _load_lora_state(Path(args.lora_path))

    modules = _collect_lora_modules(adapter_state)

    base_column = f"llama ({args.llama_path})"
    adapter_column = f"lora ({Path(args.lora_path).name})"

    rows: list[dict[str, object]] = []

    missing_base = 0
    missing_factors = 0
    compared = 0

    for prefix, factors in modules.items():
        base_lookup = _find_base_tensor(prefix, base_state)
        if base_lookup is None:
            missing_base += 1
            rows.append(
                {
                    "name": prefix,
                    base_column: None,
                    adapter_column: prefix,
                    "message": "base weight not found",
                    "lora_rank": None,
                    "lora_alpha": factors.alpha,
                    "lora_scale": factors.scaling,
                    "lora_up": factors.up_key,
                    "lora_down": factors.down_key,
                }
            )
            continue

        base_key, base_tensor = base_lookup

        try:
            delta, metadata = _materialize_delta(factors)
        except ValueError as exc:
            missing_factors += 1
            rows.append(
                {
                    "name": base_key,
                    base_column: base_key,
                    adapter_column: prefix,
                    "message": str(exc),
                    "lora_rank": None,
                    "lora_alpha": factors.alpha,
                    "lora_scale": factors.scaling,
                    "lora_up": factors.up_key,
                    "lora_down": factors.down_key,
                }
            )
            continue

        base_array = base_tensor.detach().cpu()
        if delta.shape != base_array.shape:
            if delta.numel() == base_array.numel():
                delta = delta.reshape(base_array.shape)
            else:
                missing_factors += 1
                rows.append(
                    {
                        "name": base_key,
                        base_column: base_key,
                        adapter_column: prefix,
                        "message": (
                            f"Delta shape {tuple(delta.shape)} incompatible with base {tuple(base_array.shape)}"
                        ),
                        **metadata,
                        "lora_up": factors.up_key,
                        "lora_down": factors.down_key,
                    }
                )
                continue

        new_tensor = base_array + delta.to(dtype=base_array.dtype)

        diff = _compute_diff(base_array, new_tensor)
        delta_f = _as_float_tensor(delta)

        delta_l1 = float(torch.sum(torch.abs(delta_f)).item())
        non_zero = int(torch.count_nonzero(delta_f).item())
        density = non_zero / delta_f.numel() if delta_f.numel() else 0.0

        row = {
            "name": base_key,
            base_column: base_key,
            adapter_column: prefix,
            "delta_l2": diff.delta_norm,
            "baseline_l2": diff.base_norm,
            "updated_l2": diff.new_norm,
            "relative_delta": diff.relative_delta,
            "cosine_similarity": diff.cosine_similarity,
            "max_abs_delta": diff.max_abs_delta,
            "delta_l1": delta_l1,
            "delta_density": density,
            "numel": diff.numel,
            "dtype": diff.dtype,
            "shape": diff.shape,
            "lora_rank": metadata.get("lora_rank"),
            "lora_alpha": metadata.get("lora_alpha"),
            "lora_scale": metadata.get("lora_scale"),
            "lora_up": factors.up_key,
            "lora_down": factors.down_key,
            "message": None,
        }
        rows.append(row)
        compared += 1

    rows.sort(key=lambda item: (item.get("delta_l2") or 0.0), reverse=True)

    if args.limit is not None:
        rows = rows[: args.limit]

    fieldnames = [
        "name",
        base_column,
        adapter_column,
        "delta_l2",
        "baseline_l2",
        "updated_l2",
        "relative_delta",
        "cosine_similarity",
        "max_abs_delta",
        "delta_l1",
        "delta_density",
        "numel",
        "dtype",
        "shape",
        "lora_rank",
        "lora_alpha",
        "lora_scale",
        "lora_up",
        "lora_down",
        "message",
    ]

    csv_path = out_dir / "parameters.csv"
    html_path = out_dir / "parameters.html"

    _write_csv(csv_path, rows, fieldnames)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - propagate read errors
        print(f"Failed to rebuild DataFrame for HTML output: {exc}")
    else:
        _write_html_report(
            df,
            html_path,
            base_column=base_column,
            adapter_column=adapter_column,
        )

    print(f"Compared {compared} LoRA-injected parameters")
    print(f"Missing base weights: {missing_base}")
    print(f"Skipped due to LoRA configuration issues: {missing_factors}")
    print(f"Parameter metrics written to {csv_path}")
    if html_path.exists():
        print(f"HTML report written to {html_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
