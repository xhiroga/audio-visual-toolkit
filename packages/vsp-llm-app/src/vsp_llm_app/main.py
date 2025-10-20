import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, TypedDict

import cv2
import mediapipe as mp
import numpy as np
import torch
from clustering.dump_hubert_feature import HubertFeatureReader
from clustering.dump_km_label import ApplyKmeans
from fairseq import checkpoint_utils
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vsp_llm import utils_vsp_llm
from vsp_llm.vsp_llm import VSPLLMConfig, avhubert_llm_seq2seq_cluster_count


# clustering/dump_hubert_feature.py#L98
class AVHubertModelForwardSource(TypedDict):
    audio: torch.Tensor
    video: torch.Tensor


# vsp_llm/vsp_llm_dataset.py#L372
class SourceInput(TypedDict):
    audio: torch.Tensor | None
    video: torch.Tensor
    cluster_counts: list[torch.Tensor]
    text: torch.Tensor


class NetInput(TypedDict):
    source: SourceInput
    padding_mask: torch.Tensor
    text_attn_mask: torch.Tensor


def _move_to_device(sample, device: torch.device):
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    if isinstance(sample, dict):
        return {key: _move_to_device(value, device) for key, value in sample.items()}
    if isinstance(sample, list):
        return [_move_to_device(value, device) for value in sample]
    return sample


def load_video_file(video_path: str) -> np.ndarray:
    """Load a video file as grayscale frames."""

    frames = utils_vsp_llm.load_video(video_path)
    if frames.size == 0:
        raise ValueError(f"動画フレームを取得できませんでした: {video_path}")
    return frames.astype(np.float32)


def crop_mouth(video: np.ndarray, crop_size: Optional[int] = None) -> np.ndarray:
    """Mediapipe を用いて顔下半分を中心としたクロップを行う。"""

    if video.ndim != 3:
        raise ValueError("crop_mouth expects (frames, height, width) input")

    num_frames, height, width = video.shape
    if num_frames == 0:
        return video

    mp_face_mesh = mp.solutions.face_mesh
    lip_indices = sorted({idx for c in mp_face_mesh.FACEMESH_LIPS for idx in c})
    oval_indices = sorted({idx for c in mp_face_mesh.FACEMESH_FACE_OVAL for idx in c})
    NOSE_BRIDGE_INDEX = 1

    centers: list[tuple[float, float]] = []
    target_size: Optional[float] = float(crop_size) if crop_size is not None else None
    last_center = (width / 2.0, height / 2.0)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for frame in video:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)
            results = face_mesh.process(frame_rgb)

            center = last_center
            suggested_size: Optional[float] = None

            if results.multi_face_landmarks:
                landmark = results.multi_face_landmarks[0].landmark

                xs = [landmark[idx].x * width for idx in oval_indices]
                ys = [landmark[idx].y * height for idx in oval_indices]

                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    face_width = max(x_max - x_min, 1.0)
                    face_height = max(y_max - y_min, 1.0)

                    mouth_ys = [landmark[idx].y * height for idx in lip_indices]
                    mouth_center_y = (
                        float(np.mean(mouth_ys))
                        if mouth_ys
                        else y_min + face_height * 0.65
                    )

                    nose_y = (
                        landmark[NOSE_BRIDGE_INDEX].y * height
                        if NOSE_BRIDGE_INDEX < len(landmark)
                        else y_min + face_height * 0.45
                    )

                    lower_top = max(
                        min(nose_y, mouth_center_y) - face_height * 0.05, y_min
                    )
                    lower_bottom = min(y_max + face_height * 0.05, height)
                    lower_height = max(lower_bottom - lower_top, face_height * 0.5)

                    center_x = (x_min + x_max) / 2.0
                    center_y = lower_top + lower_height * 0.6

                    center = (center_x, center_y)
                    suggested_size = max(lower_height * 1.05, face_width * 1.05)

            if suggested_size is not None:
                if target_size is None:
                    target_size = suggested_size
                else:
                    target_size = 0.8 * target_size + 0.2 * suggested_size

            centers.append(center)
            last_center = center

    if target_size is None:
        if crop_size is None:
            _, height, width = video.shape
            crop_size = min(height, width)
        cropper = utils_vsp_llm.CenterCrop((int(crop_size), int(crop_size)))
        return cropper(video)

    target_size = float(np.clip(target_size, 1.0, float(min(height, width))))
    size_int = max(1, int(round(target_size)))

    cropped = np.zeros((num_frames, size_int, size_int), dtype=video.dtype)

    half = size_int / 2.0
    for idx, frame in enumerate(video):
        cx, cy = centers[idx]
        cx = (
            float(np.clip(cx, half, width - half)) if width >= size_int else width / 2.0
        )
        cy = (
            float(np.clip(cy, half, height - half))
            if height >= size_int
            else height / 2.0
        )

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x1 = max(0, min(width - size_int, x1))
        y1 = max(0, min(height - size_int, y1))
        x2 = x1 + size_int
        y2 = y1 + size_int

        crop = frame[y1:y2, x1:x2]
        if crop.shape != (size_int, size_int):
            padded = np.zeros((size_int, size_int), dtype=frame.dtype)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded

        cropped[idx] = crop

    return cropped


def _resolve_fps(video_path: str, default: float = 25.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    if fps is None or fps <= 1e-3 or np.isnan(fps):
        return default
    return float(fps)


def _store_temp_video(
    *,
    frames: np.ndarray,
    source_path: str,
    temp_dir: Path,
) -> Path:
    """Persist processed frames to a temporary MP4 under the provided directory."""

    temp_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(source_path).stem
    output_path = temp_dir / f"{basename}_cropped.mp4"

    frames_uint8 = np.clip(frames, 0, 255).astype(np.uint8)
    if frames_uint8.ndim != 3:
        raise ValueError(
            "_store_temp_video expects frames shaped (frames, height, width)"
        )

    frame_count, height, width = frames_uint8.shape
    fps = _resolve_fps(source_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"一時動画ファイルを書き出せませんでした: {output_path}")

    try:
        for idx in range(frame_count):
            gray_frame = frames_uint8[idx]
            bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()

    logging.info("Stored temporary video at %s", output_path)
    return output_path


def build_net_input_from_video(
    video: np.ndarray,
    w2v_path: str,
    km_path: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> NetInput:
    reader = HubertFeatureReader(
        ckpt_path=w2v_path,
        layer=12,
        custom_utils=utils_vsp_llm,
    )

    if video.size == 0:
        raise ValueError("動画フレームを取得できませんでした。入力が空です。")

    transformed = reader.transform(video)
    video_frames = np.expand_dims(transformed, axis=-1)

    num_frames = video_frames.shape[0]
    print(f"{reader.task=}")
    stack_order_audio = getattr(reader.task.cfg, "stack_order_audio", 4)
    audio_feat_dim = 26 * stack_order_audio
    audio_feats = np.zeros((num_frames, audio_feat_dim), dtype=np.float32)

    audio_feats = torch.from_numpy(audio_feats).to(device)
    audio_feats = audio_feats.unsqueeze(0).transpose(1, 2)

    video_tensor_for_feat = torch.from_numpy(video_frames.astype(np.float32)).to(device)
    video_tensor_for_feat = (
        video_tensor_for_feat.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
    )

    av_source: AVHubertModelForwardSource = {
        "audio": audio_feats,
        "video": video_tensor_for_feat,
    }

    with torch.no_grad():
        feat, _ = reader.model.extract_features(
            source=av_source,
            padding_mask=None,
            mask=False,
            output_layer=12,
            ret_conv=False,
        )
    feat_np = feat.squeeze(0).cpu().numpy()

    apply_kmeans = ApplyKmeans(km_path)
    labels = apply_kmeans(feat_np)
    if labels.size == 0:
        raise ValueError("クラスタの推定に失敗しました。入力フレームが不足しています。")

    def _run_length_encode(ids: np.ndarray) -> List[int]:
        counts: List[int] = []
        current = ids[0]
        length = 1
        for value in ids[1:]:
            if value == current:
                length += 1
            else:
                counts.append(length)
                current = value
                length = 1
        counts.append(length)
        return counts

    cluster_counts_list = _run_length_encode(labels)
    cluster_counts_tensor = torch.tensor(cluster_counts_list, dtype=torch.long)
    if cluster_counts_tensor.sum().item() != num_frames:
        raise AssertionError("cluster_counts の合計とフレーム数が一致しません。")

    video_tensor = (
        torch.from_numpy(video_frames.astype(np.float32))
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .contiguous()
    )
    padding_mask = torch.zeros((1, num_frames), dtype=torch.bool)

    instruction = "Recognize this speech in English. Input : "
    instruction_ids = tokenizer(instruction, return_tensors="pt").input_ids
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    text_attn_mask = instruction_ids[0].ne(pad_token_id)

    source: SourceInput = {
        "audio": None,
        "video": video_tensor,
        "cluster_counts": [cluster_counts_tensor],
        "text": instruction_ids,
    }

    return {
        "source": source,
        "padding_mask": padding_mask,
        "text_attn_mask": text_attn_mask,
    }


def main(
    video_path: str,
    w2v_path: str,
    km_path: str,
    model_path: str,
    llm_path: str,
    crop: bool,
    temp_dir: str | None,
):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    video_frames = load_video_file(video_path)
    if crop:
        video_frames = crop_mouth(video_frames)
    if temp_dir is not None:
        _store_temp_video(
            frames=video_frames,
            source_path=video_path,
            temp_dir=Path(temp_dir),
        )

    net_input: NetInput = build_net_input_from_video(
        video=video_frames,
        w2v_path=w2v_path,
        km_path=km_path,
        tokenizer=tokenizer,
        device=device,
    )
    net_input = _move_to_device(net_input, device)

    # fairseq モデルをビルドするための cfg の構築には、次の通り複雑な手順が必要。
    # なおこの手順は、ModelとConfigは最新・Checkpotinは古いという前提に基づくため、結局は場合による。
    # 1. Dataclass から OmegaConf の設定を生成する。checkpointのcfgは最新版のキーを持たないことがある。
    # 怠った場合、`omegaconf.errors.ConfigAttributeError: Key 'dropout_input' is not in struct`のようなエラーが発生
    # 2. checkpoint の cfg をコンテナ(dict)に変換し、Modelのdataclassに存在するキーのみを抽出する。
    # 怠った場合、`OmegaConf.merge`の際に`omegaconf.errors.ConfigKeyError: Key 'decoder_layers' not in 'VSPLLMConfig'
    # 3. CLIから渡される値を設定する。
    # 4. checkpoint からは復元できないにも関わらず、VSPLLMConfig では設定ツリー上の他の項目に依存するキーがある。解決されないまま使うとエラーなので手動で設定する。
    # 怠った場合、`omegaconf.errors.ConfigKeyError: str interpolation key 'task.normalize' not found`
    # 5. vsp-llm についてはモデルのみを構築するが、その内側で使う AVHuBERT は タスクごとビルドするので、関連設定かつローカルパスに依存する箇所を上書き。
    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    base_cfg = state["cfg"]

    structured: VSPLLMConfig = OmegaConf.structured(VSPLLMConfig)
    OmegaConf.set_struct(structured, False)

    # 辞書サイズによってモデルサイズが変わるため、スタブでもファイルの指定が必要
    # なお、AV-HuBERT は辞書に 4 つの特別トークン（<pad>, <s>, </s>, <unk>）を自動で付け足すことに留意
    label_dir = os.path.join(os.path.dirname(__file__), "labels")

    model_cfg: VSPLLMConfig = OmegaConf.merge(
        structured,
        base_cfg.model,
        {
            "w2v_path": w2v_path,
            "llm_ckpt_path": llm_path,
            "normalize": True,
            "data": "",
            "w2v_args": {"task": {"labels": ["km"], "label_dir": label_dir}},
        },
    )
    model: avhubert_llm_seq2seq_cluster_count = (
        avhubert_llm_seq2seq_cluster_count.build_model(cfg=model_cfg, task=None)
    )
    LOAD_LORA_WITH_STRICT = False
    model.load_state_dict(state["model"], strict=LOAD_LORA_WITH_STRICT)
    model.eval()
    model: avhubert_llm_seq2seq_cluster_count = model.to(device)
    if device.type == "cuda":
        model.half()
        net_input["source"]["video"] = net_input["source"]["video"].to(torch.half)

    best_hypo = model.generate(**net_input)
    best_hypo = tokenizer.batch_decode(
        best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"{best_hypo=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--w2v-path", required=True)
    parser.add_argument("--km-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Enable mouth cropping prior to feature extraction.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="If set, save the processed grayscale video to this directory.",
    )
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        w2v_path=args.w2v_path,
        km_path=args.km_path,
        model_path=args.model_path,
        llm_path=args.llm_path,
        crop=args.crop,
        temp_dir=args.temp_dir,
    )
