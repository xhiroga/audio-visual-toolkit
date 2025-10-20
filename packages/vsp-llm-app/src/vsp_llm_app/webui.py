import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from fairseq import checkpoint_utils
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vsp_llm.vsp_llm import VSPLLMConfig, avhubert_llm_seq2seq_cluster_count

from vsp_llm_app.main import (
    NetInput,
    _move_to_device,
    _store_temp_video,
    build_net_input_from_video,
    crop_mouth,
    load_video_file,
)

LOGGER = logging.getLogger("vsp_llm_app.webui")
SNIPPETS_DIR = Path(__file__).resolve().parents[2] / "snippets" / "webui"


def _coerce_path(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        for item in value:
            candidate = _coerce_path(item)
            if candidate:
                return candidate
    if isinstance(value, dict):
        name = value.get("name")
        if name:
            return str(name)
    path_attr = getattr(value, "name", None)
    if path_attr:
        return str(path_attr)
    return None


def _prepare_frames(path: str, *, crop: bool) -> np.ndarray:
    frames = load_video_file(path)
    if crop:
        frames = crop_mouth(frames)
    return frames


def _ensure_video_file(
    frames: np.ndarray,
    *,
    source_path: str,
    temp_dir: Path,
) -> str:
    saved = _store_temp_video(frames=frames, source_path=source_path, temp_dir=temp_dir)
    return str(saved)


def _build_model(
    *,
    w2v_path: str,
    km_path: str,
    model_path: str,
    llm_path: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    video_frames: np.ndarray,
) -> Tuple[avhubert_llm_seq2seq_cluster_count, NetInput]:
    net_input: NetInput = build_net_input_from_video(
        video=video_frames,
        w2v_path=w2v_path,
        km_path=km_path,
        tokenizer=tokenizer,
        device=device,
    )
    net_input = _move_to_device(net_input, device)

    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    base_cfg = state["cfg"]

    structured: VSPLLMConfig = OmegaConf.structured(VSPLLMConfig)
    OmegaConf.set_struct(structured, False)

    label_dir = Path(__file__).with_name("labels")

    model_cfg: VSPLLMConfig = OmegaConf.merge(
        structured,
        base_cfg.model,
        {
            "w2v_path": w2v_path,
            "llm_ckpt_path": llm_path,
            "normalize": True,
            "data": "",
            "w2v_args": {"task": {"labels": ["km"], "label_dir": str(label_dir)}},
        },
    )

    model: avhubert_llm_seq2seq_cluster_count = (
        avhubert_llm_seq2seq_cluster_count.build_model(cfg=model_cfg, task=None)
    )
    LOAD_LORA_WITH_STRICT = False
    model.load_state_dict(state["model"], strict=LOAD_LORA_WITH_STRICT)
    model.eval()
    model = model.to(device)

    if device.type == "cuda":
        model.half()
        net_input["source"]["video"] = net_input["source"]["video"].to(torch.half)

    return model, net_input


def create_interface(
    *,
    w2v_path: str,
    km_path: str,
    model_path: str,
    llm_path: str,
    temp_dir: Path,
    default_crop: bool,
    device: torch.device,
) -> gr.Blocks:
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    def preprocess_handler(raw_video, crop_enabled, state):
        if raw_video is None:
            return (
                gr.update(value=None),
                state,
                "動画が選択されていません",
            )

        video_path = _coerce_path(raw_video)
        if video_path is None:
            return (
                gr.update(value=None),
                state,
                "動画パスを解決できませんでした",
            )

        try:
            frames = _prepare_frames(video_path, crop=crop_enabled)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("前処理に失敗しました")
            return (gr.update(value=None), state, f"前処理に失敗しました: {exc}")

        try:
            processed_path = _ensure_video_file(
                frames=frames,
                source_path=video_path,
                temp_dir=temp_dir,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("前処理結果の保存に失敗しました")
            return (gr.update(value=None), state, f"保存に失敗しました: {exc}")

        new_state = {
            "frames": frames,
            "processed_path": processed_path,
            "crop": crop_enabled,
        }

        return processed_path, new_state, "前処理が完了しました"

    def inference_handler(processed_video, state):
        video_path = _coerce_path(processed_video)
        frames: Optional[np.ndarray] = None

        if video_path:
            try:
                frames = load_video_file(video_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("前処理済み動画の読み込みに失敗しました")
                return "前処理済み動画の読み込みに失敗しました: %s" % exc, state
        elif state:
            frames = state.get("frames")
            if frames is None:
                return (
                    "前処理済みのフレームがありません。前処理を実行してください。",
                    state,
                )
        else:
            return "前処理済みの入力がありません。", state

        try:
            model, net_input = _build_model(
                w2v_path=w2v_path,
                km_path=km_path,
                model_path=model_path,
                llm_path=llm_path,
                tokenizer=tokenizer,
                device=device,
                video_frames=frames,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("モデルの構築に失敗しました")
            return f"モデル読み込みに失敗しました: {exc}", state

        with torch.no_grad():
            best_hypo = model.generate(**net_input)
        if device.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()
        decoded = tokenizer.batch_decode(
            best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if decoded:
            return decoded[0], state
        return "", state

    with gr.Blocks() as demo:
        gr.Markdown("# VSP-LLM Web UI")

        state = gr.State({})

        with gr.Row():
            with gr.Column():
                raw_video = gr.Video(
                    label="入力動画 (アップロード/撮影)",
                    sources=["upload", "webcam"],
                )
                crop_checkbox = gr.Checkbox(
                    label="口領域をクロップする",
                    value=default_crop,
                )
                preprocess_btn = gr.Button("前処理")
                preprocess_status = gr.Textbox(
                    label="前処理ステータス",
                    interactive=False,
                )

            with gr.Column():
                processed_video = gr.Video(
                    label="前処理済み動画 (アップロードまたはプレビュー)",
                    sources=["upload"],
                )
                infer_btn = gr.Button("推論")

            with gr.Column():
                result_box = gr.Textbox(
                    label="推論結果",
                    interactive=False,
                    lines=8,
                )

        preprocess_btn.click(
            fn=preprocess_handler,
            inputs=[raw_video, crop_checkbox, state],
            outputs=[processed_video, state, preprocess_status],
        )

        infer_btn.click(
            fn=inference_handler,
            inputs=[processed_video, state],
            outputs=[result_box, state],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSP-LLM Web UI")
    parser.add_argument("--w2v-path", required=True)
    parser.add_argument("--km-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=str(SNIPPETS_DIR),
        help="前処理済み動画を保存するディレクトリ",
    )
    parser.add_argument(
        "--default-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="前処理時に口領域クロップを有効化するか",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    demo = create_interface(
        w2v_path=args.w2v_path,
        km_path=args.km_path,
        model_path=args.model_path,
        llm_path=args.llm_path,
        temp_dir=temp_dir,
        default_crop=args.default_crop,
        device=device,
    )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
