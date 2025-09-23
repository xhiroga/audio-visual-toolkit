import argparse
import math
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp

MOUTH_OUTER_IDX: tuple[int, ...] = (
    # MediaPipe FaceMesh indices for outer lips region
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
)
MOUTH_INNER_IDX: tuple[int, ...] = (
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
)


def _iter_video_files(base: Path) -> Iterable[Path]:
    """Iterate video files directly under the directory (non-recursive)."""
    exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _detect_rotation(cap: cv2.VideoCapture) -> int:
    """Return clockwise rotation degrees inferred from metadata (0/90/180/270)."""
    rotate_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if rotate_prop is None:
        return 0

    value = cap.get(rotate_prop)
    if value is None:
        return 0
    if isinstance(value, float):
        if math.isnan(value):
            return 0
        value = int(round(value))
    else:
        value = int(value)

    value %= 360
    if value % 90 != 0:
        return 0
    return value


def _apply_rotation(frame, rotation: int):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def crop_mouth_video(
    input_path: Path, output_path: Path, width: int, height: int
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-3:
        fps = 30.0
    rotation = _detect_rotation(cap)
    if rotation:
        print(f"Applying rotation {rotation}° for {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Failed to open writer: {output_path}")
        cap.release()
        return False

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_cx = -1
    last_cy = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _apply_rotation(frame, rotation)
            frame_h, frame_w = frame.shape[:2]

            if last_cx < 0 or last_cy < 0:
                last_cx = frame_w // 2
                last_cy = frame_h // 2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            cx, cy = last_cx, last_cy
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                xs: list[float] = []
                ys: list[float] = []
                for idx in MOUTH_OUTER_IDX + MOUTH_INNER_IDX:
                    pt = lm[idx]
                    xs.append(pt.x * frame_w)
                    ys.append(pt.y * frame_h)
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    cx = int((min_x + max_x) / 2)
                    cy = int((min_y + max_y) / 2)
                    last_cx, last_cy = cx, cy

            # define crop top-left ensuring bounds
            half_w = width // 2
            half_h = height // 2
            x1 = max(0, min(frame_w - width, cx - half_w))
            y1 = max(0, min(frame_h - height, cy - half_h))

            crop = frame[y1 : y1 + height, x1 : x1 + width]

            # If near borders, crop might be smaller; pad to exact size
            ch, cw = crop.shape[:2]
            if ch != height or cw != width:
                pad_top = 0
                pad_bottom = height - ch
                pad_left = 0
                pad_right = width - cw
                crop = cv2.copyMakeBorder(
                    crop,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            out.write(crop)
    finally:
        cap.release()
        out.release()
        face_mesh.close()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop mouth region from videos")
    ing = parser.add_mutually_exclusive_group(required=True)
    ing.add_argument("--video-file", type=str, help="Path to a video file")
    ing.add_argument(
        "--video-dir",
        type=str,
        help="Path to a directory; process files directly under it (non-recursive)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write outputs (required)",
    )
    parser.add_argument("--width", type=int, default=300, help="Crop width (pixels)")
    parser.add_argument("--height", type=int, default=300, help="Crop height (pixels)")
    args = parser.parse_args()

    width: int = max(1, int(args.width))
    height: int = max(1, int(args.height))

    if args.video_file:
        inp = Path(args.video_file)
        if not inp.exists():
            print(f"Not found: {inp}")
            raise SystemExit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / ("LFROI_" + inp.stem + ".mp4")

        ok = crop_mouth_video(inp, out, width, height)
        raise SystemExit(0 if ok else 2)
    else:
        base = Path(args.video_dir)
        if not base.exists() or not base.is_dir():
            print(f"Not found or not a directory: {base}")
            raise SystemExit(1)
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        any_fail = False
        for p in sorted(_iter_video_files(base)):
            out = out_root / ("LFROI_" + p.stem + ".mp4")
            ok = crop_mouth_video(p, out, width, height)
            print(f"{p} -> {out}: {'OK' if ok else 'FAIL'}")
            if not ok:
                any_fail = True

        raise SystemExit(0 if not any_fail else 2)


if __name__ == "__main__":
    main()
