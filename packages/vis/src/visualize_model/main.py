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


def _normalize(
    value: object,
    depth: int | None,
    *,
    sort_keys: bool = False,
    ignore_keys: set[str] | None = None,
    current_key: str | None = None,
) -> object:
    if ignore_keys and current_key in ignore_keys:
        return _summarize(value)

    if depth is not None and depth <= 0:
        return _summarize(value)

    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    next_depth = None if depth is None else depth - 1

    if isinstance(value, dict):
        items = value.items()
        if sort_keys:
            items = sorted(items)
        return {
            k: _normalize(
                v,
                next_depth,
                sort_keys=sort_keys,
                ignore_keys=ignore_keys,
                current_key=k,
            )
            for k, v in items
        }
    if isinstance(value, list):
        return [
            _normalize(
                v,
                next_depth,
                sort_keys=sort_keys,
                ignore_keys=ignore_keys,
                current_key=current_key,
            )
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _normalize(
                v,
                next_depth,
                sort_keys=sort_keys,
                ignore_keys=ignore_keys,
                current_key=current_key,
            )
            for v in value
        )
    if hasattr(value, "__dict__"):
        return _normalize(
            vars(value),
            next_depth,
            sort_keys=sort_keys,
            ignore_keys=ignore_keys,
            current_key=current_key,
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List top-level attributes from a torch checkpoint"
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint to inspect")
    parser.add_argument(
        "--expand-depth",
        type=int,
        default=3,
        help="Maximum nesting depth to expand (use <=0 for unlimited)",
    )
    parser.add_argument(
        "--expand-ignore",
        nargs="*",
        default=["model"],
        help="Top-level keys to keep summarized regardless of depth",
    )
    args = parser.parse_args()

    path = Path(args.model_path).expanduser()
    if not path.exists():
        print(f"Not found: {path}")
        raise SystemExit(1)

    depth_arg = args.expand_depth if args.expand_depth > 0 else None
    ignore_keys = set(args.expand_ignore or [])

    try:
        torch.serialization.add_safe_globals([fairseq_dictionary.Dictionary])
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - surface load errors to the user
        print(f"Failed to load checkpoint: {exc}")
        raise SystemExit(2)

    normalized = _normalize(
        payload,
        depth_arg,
        sort_keys=isinstance(payload, dict),
        ignore_keys=ignore_keys,
        current_key=None,
    )
    print(pformat(normalized, sort_dicts=False, width=100))


if __name__ == "__main__":
    main()
