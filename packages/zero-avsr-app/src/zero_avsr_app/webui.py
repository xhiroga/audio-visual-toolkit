import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from hydra.experimental import compose, initialize_config_dir
from zero_avsr_app.main import (
    ALLOWED_LANGUAGE_CODES,
    CONFIG_DIR,
    CONFIG_NAME,
    LANGUAGE_NAME_BY_CODE,
    run_inference,
)

MODE_LABELS = {
    "avsr": "Audio + Video (AVSR)",
    "vsr": "Video Only (VSR)",
}


def _maybe_path(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, dict) and "name" in file_obj:
        return Path(file_obj["name"])
    path_attr = getattr(file_obj, "path", None)
    if path_attr:
        return Path(path_attr)
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


def _create_predict_fn(
    llm_path: str,
    av_romanizer_path: Path,
    avhubert_path: Path,
    model_path: Path,
    log_level: str,
):
    logger = logging.getLogger("zavsr.webui")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def _predict(
        video_file,
        audio_file,
        language_code: str,
        mode_label: str,
    ) -> str:
        video_path = _maybe_path(video_file)
        if video_path is None:
            return "動画ファイルをアップロードしてください。"

        if language_code not in ALLOWED_LANGUAGE_CODES:
            return "未対応の言語コードです。"

        mode_key = next((k for k, v in MODE_LABELS.items() if v == mode_label), "avsr")
        audio_path: Optional[Path]
        if mode_key == "avsr":
            audio_path = _maybe_path(audio_file)
            if audio_path is None:
                return "AVSRモードでは音声ファイルのアップロードが必要です。"
        else:
            audio_path = None

        outputs = run_inference(
            video_path=video_path,
            audio_path=audio_path,
            lang_code=language_code,
            llm_path=llm_path,
            av_romanizer_path=av_romanizer_path,
            avhubert_path=avhubert_path,
            model_path=model_path,
            logger=logger,
            mode=mode_key,
        )

        if not outputs:
            return "推論結果を取得できませんでした。"

        return outputs[0]

    return _predict


def build_interface(predict_fn):
    lang_choices = [
        (code, f"{code} - {LANGUAGE_NAME_BY_CODE[code]}")
        for code in ALLOWED_LANGUAGE_CODES
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
                video_input = gr.File(
                    label="動画 (MP4)",
                    file_types=["video"],
                )
                audio_input = gr.File(
                    file_types=["audio"],
                    label="音声 (WAV, AVSRモード時必須)",
                )
                submit_btn = gr.Button("推論を実行")
            with gr.Column():
                output_box = gr.Textbox(
                    label="生成結果",
                    lines=6,
                    interactive=False,
                )

        def _convert_language_label(selected_label: str) -> str:
            code = selected_label.split(" - ", 1)[0]
            return code if code in LANGUAGE_NAME_BY_CODE else "eng"

        def _run(video, audio, lang_label, mode_label):
            lang_code = _convert_language_label(lang_label)
            return predict_fn(video, audio, lang_code, mode_label)

        def _toggle_audio(mode_label):
            hide = mode_label == MODE_LABELS["vsr"]
            return gr.update(visible=not hide, interactive=not hide)

        submit_btn.click(
            _run,
            inputs=[video_input, audio_input, lang_dropdown, mode_radio],
            outputs=output_box,
        )

        mode_radio.change(
            _toggle_audio,
            inputs=mode_radio,
            outputs=audio_input,
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
        f"common.user_dir={Path(__file__).resolve().parent}",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    model_cfg = getattr(cfg, "model", None)
    cfg_w2v = getattr(model_cfg, "w2v_path", None) if model_cfg is not None else None

    predict_fn = _create_predict_fn(
        llm_path=cfg.override.llm_path,
        av_romanizer_path=Path(cfg.override.av_romanizer_path),
        avhubert_path=Path(cfg_w2v) if cfg_w2v else args.avhubert_path,
        model_path=Path(cfg.common_eval.path),
        log_level=args.log_level,
    )

    interface = build_interface(predict_fn)
    interface.launch(share=args.share)


if __name__ == "__main__":
    main()
