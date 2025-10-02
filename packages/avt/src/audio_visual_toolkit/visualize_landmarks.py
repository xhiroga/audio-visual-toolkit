import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from pydantic import TypeAdapter, ValidationError, conlist
from tqdm import tqdm


PointType = conlist(float, min_length=2, max_length=2)
FrameType = conlist(PointType, min_length=68, max_length=68)
SegmentsAdapter = TypeAdapter(dict[str, list[FrameType]])


def _render_segment(frames: list[list[list[float]]], out_path: Path) -> None:
    points = np.array([p for frame in frames for p in frame], dtype=np.float32)
    if points.size == 0:
        return
    pad = 16
    min_xy = points.min(axis=0) - pad
    size = points.max(axis=0) - min_xy + pad
    width = int(np.ceil(size[0]))
    height = int(np.ceil(size[1]))
    width = max(width, 32)
    height = max(height, 32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    try:
        for frame in frames:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for x, y in frame:
                ix = int(round(x - min_xy[0]))
                iy = int(round(y - min_xy[1]))
                if 0 <= ix < width and 0 <= iy < height:
                    cv2.circle(canvas, (ix, iy), 2, (0, 255, 0), -1)
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render landmark-only videos")
    parser.add_argument("--pkl-file", required=True, help="Path to landmark pickle")
    parser.add_argument(
        "--out-dir", required=True, help="Directory to write rendered videos"
    )
    args = parser.parse_args()
    pkl_path = Path(args.pkl_file)
    out_dir = Path(args.out_dir)
    if not pkl_path.exists():
        raise SystemExit(f"Not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        segments = pickle.load(f)
    try:
        segments = SegmentsAdapter.validate_python(segments)
    except ValidationError as exc:
        raise SystemExit(
            "Landmark pickle must be dict[str, list[frame]] where each frame has 68"
            " points [x, y].\n"
            f"Validation error: {exc}"
        ) from exc
    json_path = Path(args.out_dir) / f"{pkl_path.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)
    any_written = False
    for seg_id, frames in tqdm(
        segments.items(),
        total=len(segments),
        desc=f"Rendering {pkl_path.stem}",
    ):
        if not frames:
            continue
        out_path = out_dir / f"{seg_id}.mp4"
        _render_segment(frames, out_path)
        any_written = True
    if not any_written:
        raise SystemExit("No frames rendered")


if __name__ == "__main__":
    main()
