import argparse
import logging
import os
from transformers import AutoTokenizer

from vsp_llm.vsp_llm import avhubert_llm_seq2seq_cluster_count, VSPLLMConfig

def main(video_path: str, w2v_path: str, llm_path: str):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper()
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    # Default values: https://github.com/Sally-SH/VSP-LLM/blob/91c4f3d4418ac9cd0f6328f043de5e48e1d091b2/src/vsp_llm.py#L34
    # Checkpoint values: uvx --python 3.10 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/fairseq-toolkit' dump --model-path $REPOS/VSP-LLM/checkpoints/checkpoint_finetune.pt
    # Train values: https://github.com/Sally-SH/VSP-LLM/blob/91c4f3d4418ac9cd0f6328f043de5e48e1d091b2/src/conf/vsp-llm-433h-finetune.yaml#L73
    cfg = VSPLLMConfig(
        w2v_path=w2v_path,
        llm_ckpt_path=llm_path,
        apply_mask =False,
        mask_selection= "static",
        mask_length= 10,
        mask_other= 0,
        mask_prob= 0.75,
        mask_channel_selection= "static",
        mask_channel_length= 64,
        mask_channel_other= 0,
        mask_channel_prob= 0.5,
        layerdrop= 0.1,
        dropout= 0.0,
        activation_dropout= 0.1,
        attention_dropout= 0.0,
        feature_grad_mult= 1.0,
        encoder_embed_dim= 1024,
        decoder_embed_dim= 4096,
        freeze_finetune_updates= 18000,
        # Should be same with w2v_args.task.normalize.
        normalize=True,
    )
    model = avhubert_llm_seq2seq_cluster_count.build_model(cfg=cfg, task=None)

    print(f"{model}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--w2v-path", required=True)
    parser.add_argument("--llm-path", required=True)
    args = parser.parse_args()
    main(video_path=args.video_path, llm_path=args.llm_path, w2v_path=args.w2v_path)
