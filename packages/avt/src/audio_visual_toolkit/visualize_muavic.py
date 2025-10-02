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


def _render_overlay_segment(
    frames: list[list[list[float]]],
    cap: cv2.VideoCapture,
    fps: float,
    start_sec: float,
    out_path: Path,
) -> None:
    if not frames:
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Failed to read video dimensions")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    start_frame = int(round(start_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for landmarks in frames:
            ok, frame = cap.read()
            if not ok:
                break
            for x, y in landmarks:
                cv2.circle(frame, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def _load_segment_times(segments_file: Path, talk_id: str) -> tuple[list[str], dict[str, tuple[float, float]]]:
    order: list[str] = []
    mapping: dict[str, tuple[float, float]] = {}
    with open(segments_file) as sf:
        for line in sf:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            seg_id, seg_talk, start_sec, end_sec = parts
            if seg_talk != talk_id:
                continue
            if seg_id not in mapping:
                order.append(seg_id)
            mapping[seg_id] = (float(start_sec), float(end_sec))
    return order, mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MuAViC landmarks")
    parser.add_argument("--pkl-file", required=True, help="Path to landmark pickle")
    parser.add_argument("--out-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--video-file", help="Source video for overlay rendering")
    parser.add_argument(
        "--segments-file",
        help="segments file mapping IDs to start/end seconds (use with --video-file)",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{pkl_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    if (args.video_file is None) ^ (args.segments_file is None):
        raise SystemExit("--video-file と --segments-file は同時に指定してください")

    overlay = args.video_file is not None
    seg_ids = list(segments.keys())
    seg_order = seg_ids
    seg_times: dict[str, tuple[float, float]] = {}
    cap: cv2.VideoCapture | None = None
    fps = 25.0

    if overlay:
        video_path = Path(args.video_file)
        segments_file = Path(args.segments_file)
        if not video_path.exists():
            raise SystemExit(f"Not found: {video_path}")
        if not segments_file.exists():
            raise SystemExit(f"Not found: {segments_file}")
        seg_order, seg_times = _load_segment_times(segments_file, pkl_path.stem)
        if not seg_order:
            raise SystemExit("segments ファイルに対象トークの情報がありません")
        missing = [seg_id for seg_id in seg_ids if seg_id not in seg_times]
        if missing:
            preview = ", ".join(sorted(missing[:5]))
            suffix = "..." if len(missing) > 5 else ""
            raise SystemExit(
                "segments ファイルに以下の ID が見つかりませんでした: "
                f"{preview}{suffix}"
            )
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if fps <= 1e-3:
            fps = 25.0
        seg_order = [seg_id for seg_id in seg_order if seg_id in segments]
        seg_ids = seg_order

    any_written = False
    try:
        for seg_id in tqdm(seg_ids, desc=f"Rendering {pkl_path.stem}", total=len(seg_ids)):
            frames = segments[seg_id]
            if not frames:
                continue
            out_path = out_dir / f"{seg_id}.mp4"
            if overlay and cap is not None:
                start_sec, _ = seg_times[seg_id]
                _render_overlay_segment(frames, cap, fps, start_sec, out_path)
            else:
                _render_segment(frames, out_path)
            any_written = True
    finally:
        if cap is not None:
            cap.release()

    if not any_written:
        raise SystemExit("No frames rendered")


if __name__ == "__main__":
    main()
