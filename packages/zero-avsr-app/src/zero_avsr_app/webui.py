import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from hydra.experimental import compose, initialize_config_dir

from zero_avsr_app.main import (
    CONFIG_DIR,
    CONFIG_NAME,
    SUPPORTED_LANGUAGE_CODES,
    load_audio_waveform,
    run_inference,
)

MODE_LABELS = {
    "avsr": "Audio + Video (AVSR)",
    "vsr": "Video Only (VSR)",
}

LANGUAGE_NAME_BY_CODE = {
    "ara": "Arabic",
    "deu": "German",
    "ell": "Greek",
    "spa": "Spanish",
    "fra": "French",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "eng": "English",
}


def _maybe_path(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, tuple) and file_obj:
        # Video component may return (video_path, audio_path)
        for item in file_obj:
            candidate = _maybe_path(item)
            if candidate is not None:
                return candidate
        return None
    if isinstance(file_obj, dict) and "name" in file_obj:
        return Path(file_obj["name"])
    path_attr = getattr(file_obj, "path", None)
    if path_attr:
        return Path(path_attr)
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


def _split_media(file_obj: Any) -> Tuple[Optional[Path], Optional[Path]]:
    if isinstance(file_obj, tuple) and len(file_obj) == 2:
        return _maybe_path(file_obj[0]), _maybe_path(file_obj[1])
    path = _maybe_path(file_obj)
    return path, None


def _extract_language_code(label: str) -> str:
    code = label.split(" - ", 1)[0]
    return code if code in LANGUAGE_NAME_BY_CODE else "eng"


def extract_mouth_crops(
    video_path: Path,
    output_size: int = 96,
    margin_ratio: float = 0.25,
) -> Tuple[np.ndarray, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 25.0

    lip_indices = sorted(
        {
            idx
            for connection in mp.solutions.face_mesh.FACEMESH_LIPS
            for idx in connection
        }
    )

    frames_gray: list[np.ndarray] = []
    frames_preview: list[np.ndarray] = []
    last_bbox: Optional[Tuple[int, int, int, int]] = None

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            bbox = None
            if results.multi_face_landmarks:
                landmark = results.multi_face_landmarks[0]
                xs, ys = [], []
                for idx in lip_indices:
                    lm = landmark.landmark[idx]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    width = x_max - x_min
                    height = y_max - y_min
                    expand_x = max(int(width * margin_ratio), 1)
                    expand_y = max(int(height * margin_ratio), 1)
                    x_min = max(x_min - expand_x, 0)
                    x_max = min(x_max + expand_x, w - 1)
                    y_min = max(y_min - expand_y, 0)
                    y_max = min(y_max + expand_y, h - 1)
                    if x_max > x_min and y_max > y_min:
                        bbox = (x_min, y_min, x_max, y_max)
                        last_bbox = bbox

            if bbox is None:
                if last_bbox is not None:
                    bbox = last_bbox
                else:
                    side = min(h, w) // 2
                    cx, cy = w // 2, h // 2
                    half = max(side // 2, 1)
                    bbox = (
                        max(cx - half, 0),
                        max(cy - half, 0),
                        min(cx + half, w - 1),
                        min(cy + half, h - 1),
                    )

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)
            frames_gray.append(resized.astype(np.float32))
            frames_preview.append(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))

    cap.release()

    if not frames_gray:
        raise ValueError("動画からフレームを取得できませんでした")

    preview_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    preview_path = preview_file.name
    preview_file.close()

    writer = cv2.VideoWriter(
        preview_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_size, output_size),
    )
    for frame_bgr in frames_preview:
        writer.write(frame_bgr)
    writer.release()

    frames_array = np.stack(frames_gray, axis=0)
    return frames_array, preview_path


def _create_handlers(
    llm_path: str,
    av_romanizer_path: Path,
    avhubert_path: Path,
    model_path: Path,
    log_level: str,
):
    logger = logging.getLogger("zero_avsr_app.webui")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def preprocess(video_file, language_label, mode_label):
        lang_code = _extract_language_code(language_label)
        if lang_code not in SUPPORTED_LANGUAGE_CODES:
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False), None

        video_path, audio_path = _split_media(video_file)
        if video_path is None:
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False), None

        mode_key = next((k for k, v in MODE_LABELS.items() if v == mode_label), "avsr")

        try:
            frames, preview_path = extract_mouth_crops(video_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("口元抽出に失敗しました: %s", exc)
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False), None

        audio_waveform: Optional[np.ndarray] = None
        audio_rate: Optional[int] = None
        audio_preview = gr.update(value=None, visible=False)

        if mode_key == "avsr":
            audio_source = audio_path if audio_path is not None else video_path
            try:
                audio_waveform, audio_rate = load_audio_waveform(
                    audio_source, sample_rate=None
                )
                audio_preview = gr.update(
                    value=(audio_rate, audio_waveform),
                    visible=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("音声の抽出に失敗しました: %s", exc)

        state: Dict[str, Any] = {
            "frames": frames,
            "audio_waveform": audio_waveform,
            "audio_rate": audio_rate,
            "lang_code": lang_code,
            "mode": mode_key,
        }

        return gr.update(value=preview_path, visible=True), audio_preview, state

    def infer(processed_state):
        if not processed_state:
            return "前処理を実行してください。"

        try:
            outputs = run_inference(
                video_frames=processed_state["frames"],
                audio_waveform=processed_state.get("audio_waveform"),
                audio_rate=processed_state.get("audio_rate"),
                lang=processed_state["lang_code"],
                llm_path=llm_path,
                av_romanizer_path=av_romanizer_path,
                avhubert_path=avhubert_path,
                model_path=model_path,
                logger=logger,
                mode=processed_state["mode"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("推論に失敗しました: %s", exc)
            return f"推論に失敗しました: {exc}"

        if not outputs:
            return "推論結果を取得できませんでした。"
        return outputs[0]

    return preprocess, infer


def build_interface(handlers):
    preprocess_fn, infer_fn = handlers
    lang_choices = [
        (code, f"{code} - {LANGUAGE_NAME_BY_CODE[code]}")
        for code in SUPPORTED_LANGUAGE_CODES
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# Zero-AVSR Web UI")
        with gr.Row():
            with gr.Column():
                mode_radio = gr.Radio(
                    choices=list(MODE_LABELS.values()),
                    value=MODE_LABELS["avsr"],
                    label="モード",
                )
                lang_dropdown = gr.Dropdown(
                    choices=[label for _, label in lang_choices],
                    value=f"eng - {LANGUAGE_NAME_BY_CODE['eng']}",
                    label="言語",
                )
                video_input = gr.Video(
                    label="動画 (MP4)",
                    sources=["upload", "webcam"],
                    format="mp4",
                )
                preprocess_btn = gr.Button("前処理を実行")
            with gr.Column():
                preview_video = gr.Video(
                    label="口元プレビュー",
                    autoplay=True,
                    visible=False,
                )
                audio_preview = gr.Audio(
                    label="音声プレビュー",
                    interactive=False,
                    visible=False,
                )
                infer_btn = gr.Button("推論を実行")
            with gr.Column():
                output_box = gr.Textbox(
                    label="生成結果",
                    lines=6,
                    interactive=False,
                )

        processed_state = gr.State()

        preprocess_btn.click(
            preprocess_fn,
            inputs=[video_input, lang_dropdown, mode_radio],
            outputs=[preview_video, audio_preview, processed_state],
        )

        infer_btn.click(
            infer_fn,
            inputs=[processed_state],
            outputs=output_box,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--av-romanizer-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--avhubert-path", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradio share flag",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    overrides = [
        f"common_eval.path={args.model_path}",
        f"override.llm_path={args.llm_path}",
        f"override.av_romanizer_path={args.av_romanizer_path}",
        f"+model.w2v_path={args.avhubert_path}",
        "override.modalities=[video,audio]",
        "override.use_speech_embs=true",
        f"common.user_dir={Path(__file__).resolve().parent}",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    model_cfg = getattr(cfg, "model", None)
    cfg_w2v = getattr(model_cfg, "w2v_path", None) if model_cfg is not None else None

    preprocess_fn, infer_fn = _create_handlers(
        llm_path=cfg.override.llm_path,
        av_romanizer_path=Path(cfg.override.av_romanizer_path),
        avhubert_path=Path(cfg_w2v) if cfg_w2v else args.avhubert_path,
        model_path=Path(cfg.common_eval.path),
        log_level=args.log_level,
    )

    interface = build_interface((preprocess_fn, infer_fn))
    interface.launch(share=args.share)


if __name__ == "__main__":
    main()
