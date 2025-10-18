import argparse
import json
from pathlib import Path

import torch

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
    fold_paths: set[tuple[str, ...]] | None = None,
    current_path: tuple[str, ...] = (),
) -> object:
    if fold_paths and current_path in fold_paths:
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
                fold_paths=fold_paths,
                current_path=current_path + (k,),
            )
            for k, v in items
        }
    if isinstance(value, list):
        return [
            _normalize(
                v,
                next_depth,
                sort_keys=sort_keys,
                fold_paths=fold_paths,
                current_path=current_path,
            )
            for v in value
        ]
    if isinstance(value, tuple):
        return [
            _normalize(
                v,
                next_depth,
                sort_keys=sort_keys,
                fold_paths=fold_paths,
                current_path=current_path,
            )
            for v in value
        ]
    if hasattr(value, "__dict__"):
        return _normalize(
            vars(value),
            next_depth,
            sort_keys=sort_keys,
            fold_paths=fold_paths,
            current_path=current_path,
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
        "--fold-model-config",
        action="store_true",
        help="Summarize top-level model config instead of expanding it",
    )
    args = parser.parse_args()

    path = Path(args.model_path).expanduser()
    if not path.exists():
        print(f"Not found: {path}")
        raise SystemExit(1)

    depth_arg = args.expand_depth if args.expand_depth > 0 else None
    fold_paths: set[tuple[str, ...]] = set()
    if args.fold_model_config:
        fold_paths.add(("model",))

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - surface load errors to the user
        print(f"Failed to load checkpoint: {exc}")
        raise SystemExit(2)

    normalized = _normalize(
        payload,
        depth_arg,
        sort_keys=isinstance(payload, dict),
        fold_paths=fold_paths,
        current_path=(),
    )
    print(json.dumps(normalized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
