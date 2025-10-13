# References:
# - zero-avsr/scripts/stage2/eval.sh
# - zero-avsr/stage2/eval.py
# - avhubert/hubert_pretraining.py

import argparse
import logging
from pathlib import Path
from typing import List, Optional, TypedDict, Union

import cv2
import librosa
import numpy as np
import stage2  # noqa: F401  # タスク登録の副作用が必要
import torch
from fairseq import checkpoint_utils, utils
from hydra.experimental import compose, initialize_config_dir
from python_speech_features import logfbank
from transformers import AutoTokenizer


class SourceInput(TypedDict):
    audio: Optional[torch.Tensor]
    video: Optional[torch.Tensor]
    instruction: List[torch.Tensor]
    roman_sources: List[str]
    langs: List[str]
    zero_shot_samples: torch.Tensor


class NetInput(TypedDict):
    source: SourceInput
    padding_mask: torch.Tensor


def build_net_input_from_video(
    video_frames: np.ndarray,
    instruction: torch.Tensor,
    roman_source: str,
    lang_id: Union[int, str],
    zero_shot: bool,
    audio_features: Optional[np.ndarray] = None,
) -> NetInput:
    """Convert raw features into the minimal net_input structure expected by zero-AVSR."""

    if video_frames.ndim != 3:
        raise ValueError("video_frames must be a (time, height, width) array")
    if instruction.dim() != 1:
        raise ValueError("instruction must be a 1D tensor of token ids")

    video_tensor = torch.from_numpy(video_frames).float().unsqueeze(0)
    video_tensor = video_tensor.unsqueeze(0)  # -> [1, 1, T, H, W]
    time_length = video_tensor.shape[2]

    audio_tensor: Optional[torch.Tensor]
    if audio_features is None:
        audio_tensor = None
    else:
        if audio_features.ndim != 2:
            raise ValueError("audio_features must be a (time, feature) array")
        audio_tensor = (
            torch.from_numpy(audio_features).float().transpose(0, 1).unsqueeze(0)
        )
        audio_time = audio_tensor.shape[-1]
        if audio_time < time_length:
            pad = torch.zeros(
                (1, audio_tensor.shape[1], time_length - audio_time),
                dtype=audio_tensor.dtype,
            )
            audio_tensor = torch.cat([audio_tensor, pad], dim=-1)
        elif audio_time > time_length:
            audio_tensor = audio_tensor[..., :time_length]
        time_length = audio_tensor.shape[-1]

    padding_mask = torch.zeros((1, time_length), dtype=torch.bool)

    source: SourceInput = {
        "audio": audio_tensor,
        "video": video_tensor,
        "instruction": [instruction],
        "roman_sources": [roman_source],
        "langs": [str(lang_id)],
        "zero_shot_samples": torch.tensor([int(zero_shot)], dtype=torch.long),
    }

    return {"source": source, "padding_mask": padding_mask}


def stack_audio_features(feats: np.ndarray, stack_order: int) -> np.ndarray:
    if stack_order <= 1:
        return feats

    feat_dim = feats.shape[1]
    remainder = len(feats) % stack_order
    if remainder != 0:
        pad = np.zeros((stack_order - remainder, feat_dim), dtype=feats.dtype)
        feats = np.vstack([feats, pad])
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
        -1, stack_order * feat_dim
    )
    return feats


def load_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"動画を読み込めませんでした: {video_path}")

    return np.stack(frames).astype(np.float32)


def preprocess_video(
    frames: np.ndarray, crop_size: int, mean: float, std: float
) -> np.ndarray:
    frames = (frames - 0.0) / 255.0
    t, h, w = frames.shape
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    frames = frames[:, top : top + crop_size, left : left + crop_size]
    frames = (frames - mean) / std
    return frames.astype(np.float32)


def load_audio_logfbank(
    audio_path: Path, sample_rate: int, stack_order: int
) -> np.ndarray:
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"音声トラックを取得できませんでした: {audio_path}")

    feats = logfbank(audio, samplerate=sample_rate).astype(np.float32)
    feats = stack_audio_features(feats, stack_order)
    return feats


def main(
    video_path: Path,
    audio_path: Optional[Path],
    lang: Optional[str],
    llm_path: str,
    av_romanizer_path: Path,
    avhubert_path: Path,
    model_path: Path,
    log_level: str,
):
    """
    Recognize the video file given as an argument with Zero-AVSR and return the result.
    """

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger("zavsr")

    logger.info("Loading tokenizer from %s", llm_path)

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.sep_token = tokenizer.unk_token

    model_override_cfg = {
        "model": {
            "llm_path": llm_path,
            "av_romanizer_path": str(av_romanizer_path),
            "w2v_path": str(avhubert_path),
        }
    }

    logger.info("Loading Zero-AVSR model from %s", model_path)
    models, saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path)], arg_overrides=model_override_cfg, strict=False
    )
    logger.info("Loaded %d model(s)", len(models))
    if not models:
        raise RuntimeError("モデルのロードに失敗しました")

    model = models[0]
    model.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        model.half()

    sample_rate = getattr(saved_cfg.task, "sample_rate", 16000)
    stack_order = getattr(saved_cfg.task, "stack_order_audio", 1)
    crop_size = getattr(saved_cfg.task, "image_crop_size", 88)
    image_mean = getattr(saved_cfg.task, "image_mean", 0.421)
    image_std = getattr(saved_cfg.task, "image_std", 0.165)

    logger.debug(
        "Preprocess params: sample_rate=%s stack_order=%s crop_size=%s image_mean=%s image_std=%s",
        sample_rate,
        stack_order,
        crop_size,
        image_mean,
        image_std,
    )

    logger.info("Loading video frames from %s", video_path)
    raw_frames = load_video_frames(video_path)
    video_frames = preprocess_video(
        raw_frames, crop_size=crop_size, mean=image_mean, std=image_std
    )

    audio_features: Optional[np.ndarray] = None
    if audio_path:
        try:
            logger.info("Extracting audio features from %s", audio_path)
            audio_features = load_audio_logfbank(
                audio_path, sample_rate=sample_rate, stack_order=stack_order
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("音声特徴量の抽出に失敗しました: %s", exc)
            audio_features = None
    else:
        logger.info("Audio path not provided; skipping audio features")

    lang_name = lang or "English"
    instruction_ids = tokenizer(
        f"Given romanized transcriptions extracted from audio-visual materials, back-transliterate them into the original script of {lang_name}. Input:",
        return_tensors="pt",
    ).input_ids[0]

    net_input = build_net_input_from_video(
        video_frames=video_frames,
        instruction=instruction_ids,
        roman_source="",
        lang_id=lang_name,
        zero_shot=False,
        audio_features=audio_features,
    )

    logger.debug(
        "Prepared net input: video_shape=%s audio_shape=%s",
        net_input["source"]["video"].shape
        if net_input["source"]["video"] is not None
        else None,
        net_input["source"]["audio"].shape
        if net_input["source"]["audio"] is not None
        else None,
    )

    if use_cuda:
        net_input = utils.move_to_cuda(net_input)
        if net_input["source"]["video"] is not None:
            net_input["source"]["video"] = net_input["source"]["video"].half()
        if net_input["source"]["audio"] is not None:
            net_input["source"]["audio"] = net_input["source"]["audio"].half()

    with torch.no_grad():
        logger.info("Running model.generate")
        hypotheses = model.generate(
            num_beams=2,
            temperature=0.3,
            use_speech_embs=True,
            use_roman_toks=False,
            **net_input,
        )

    decoded = tokenizer.batch_decode(
        hypotheses, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for idx, text in enumerate(decoded):
        logger.info("Hypothesis[%d]: %s", idx, text)


CONFIG_DIR = Path(__file__).resolve().parent / "conf"
CONFIG_NAME = "s2s_decode"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--audio-path", type=Path)
    parser.add_argument("--lang")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--av-romanizer-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--avhubert-path", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    overrides = [
        f"common_eval.path={args.model_path}",
        f"override.llm_path={args.llm_path}",
        f"override.av_romanizer_path={args.av_romanizer_path}",
        f"+model.w2v_path={args.avhubert_path}",
        ("override.modalities=[video,audio]" if args.audio_path else "override.modalities=[video]"),
        f"common.user_dir={Path(__file__).resolve().parent}",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    model_cfg = getattr(cfg, "model", None)
    cfg_w2v = getattr(model_cfg, "w2v_path", None) if model_cfg is not None else None

    main(
        video_path=args.video_path,
        audio_path=args.audio_path,
        lang=args.lang,
        llm_path=cfg.override.llm_path,
        av_romanizer_path=Path(cfg.override.av_romanizer_path),
        avhubert_path=Path(cfg_w2v) if cfg_w2v else args.avhubert_path,
        model_path=Path(cfg.common_eval.path),
        log_level=args.log_level,
    )
