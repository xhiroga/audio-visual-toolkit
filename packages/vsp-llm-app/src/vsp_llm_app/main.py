import argparse
import logging
import os
from typing import List, TypedDict

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


def preprocess_video(
    video_path: str,
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

    video_frames = reader.load_image(video_path)
    if video_frames.size == 0:
        raise ValueError(f"動画フレームを取得できませんでした: {video_path}")

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


def main(video_path: str, w2v_path: str, km_path: str, model_path: str, llm_path: str):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    net_input: NetInput = preprocess_video(
        video_path=video_path,
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
    print(f"{best_hypo=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--w2v-path", required=True)
    parser.add_argument("--km-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llm-path", required=True)
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        w2v_path=args.w2v_path,
        km_path=args.km_path,
        model_path=args.model_path,
        llm_path=args.llm_path,
    )
