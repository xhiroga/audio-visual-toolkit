import argparse
import logging
import os

from fairseq import checkpoint_utils
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vsp_llm.vsp_llm import VSPLLMConfig, avhubert_llm_seq2seq_cluster_count


def main(video_path: str, model_path: str, w2v_path: str, llm_path: str):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--w2v-path", required=True)
    parser.add_argument("--llm-path", required=True)
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        model_path=args.model_path,
        llm_path=args.llm_path,
        w2v_path=args.w2v_path,
    )
