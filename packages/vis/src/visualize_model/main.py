import argparse
from pathlib import Path
from pprint import pformat

import torch
from fairseq.data import dictionary as fairseq_dictionary


def _summarize(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if hasattr(value, "__dict__"):
        return f"{type(value).__name__}(attrs={len(vars(value))})"
    return value


def _normalize(value: object, depth: int | None) -> object:
    if depth is not None and depth <= 0:
        return _summarize(value)

    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        next_depth = None if depth is None else depth - 1
        return {k: _normalize(v, next_depth) for k, v in value.items()}
    if isinstance(value, list):
        next_depth = None if depth is None else depth - 1
        return [_normalize(v, next_depth) for v in value]
    if isinstance(value, tuple):
        next_depth = None if depth is None else depth - 1
        return tuple(_normalize(v, next_depth) for v in value)
    if hasattr(value, "__dict__"):
        return _normalize(vars(value), depth)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List top-level attributes from a torch checkpoint"
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint to inspect")
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Pretty-print depth (use <=0 for unlimited)",
    )
    args = parser.parse_args()

    path = Path(args.model_path).expanduser()
    if not path.exists():
        print(f"Not found: {path}")
        raise SystemExit(1)

    depth_arg = args.depth if args.depth > 0 else None

    try:
        torch.serialization.add_safe_globals([fairseq_dictionary.Dictionary])
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - surface load errors to the user
        print(f"Failed to load checkpoint: {exc}")
        raise SystemExit(2)

    if isinstance(payload, dict):
        normalized = {
            key: _normalize(payload[key], depth_arg) for key in sorted(payload)
        }
        print(pformat(normalized, sort_dicts=False, width=100))
        return

    print(pformat(_normalize(payload, depth_arg)))


if __name__ == "__main__":
    main()
