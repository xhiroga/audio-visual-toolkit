# References:
# - zero-avsr/scripts/stage2/eval.sh
# - zero-avsr/stage2/eval.py

import argparse
from pathlib import Path

from fairseq import checkpoint_utils
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def main(movie_path: Path, llm_path: str, av_romanizer_path: Path, model_path: Path):
    """
    Recognize the video file given as an argument with Zero-AVSR and return the result.
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model_override_cfg = {
        "model": {
            "llm_path": llm_path,
            "av_romanizer_path": str(av_romanizer_path),
        }
    }
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path)], model_override_cfg, strict=False
    )

    print("Hello from zavsr!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-path", type=Path, required=True)
    parser.add_argument("--llm-path")
    parser.add_argument("--av-romanizer-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("conf"))
    parser.add_argument("--config-name", default="s2s_decode")
    parser.add_argument("--modalities", nargs="*", default=None)
    args = parser.parse_args()

    config_path = args.config_dir / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    if args.llm_path:
        OmegaConf.update(cfg, "override.llm_path", args.llm_path, merge=False)
    if args.av_romanizer_path:
        OmegaConf.update(
            cfg, "override.av_romanizer_path", str(args.av_romanizer_path), merge=False
        )
    if args.model_path:
        OmegaConf.update(cfg, "common_eval.path", str(args.model_path), merge=False)
    if args.modalities is not None:
        OmegaConf.update(cfg, "override.modalities", list(args.modalities), merge=False)
    elif OmegaConf.is_missing(cfg, "override.modalities") or OmegaConf.select(
        cfg, "override.modalities"
    ) is None:
        OmegaConf.update(cfg, "override.modalities", ["video", "audio"], merge=False)

    if OmegaConf.is_missing(cfg, "common.user_dir") or OmegaConf.select(
        cfg, "common.user_dir"
    ) is None:
        OmegaConf.update(
            cfg,
            "common.user_dir",
            str(Path(__file__).resolve().parent),
            merge=False,
        )

    def required(path: str, hint: str, flag_name: str) -> str:
        if OmegaConf.is_missing(cfg, path):
            parser.error(f"{hint} を --{flag_name} で指定してください")
        value = OmegaConf.select(cfg, path)
        if value is None:
            parser.error(f"{hint} を設定してください")
        return value

    llm_path = args.llm_path or required("override.llm_path", "LLM のパス", "llm-path")
    av_romanizer_value = (
        str(args.av_romanizer_path)
        if args.av_romanizer_path
        else required("override.av_romanizer_path", "av-romanizer のパス", "av-romanizer-path")
    )
    model_path_value = (
        str(args.model_path)
        if args.model_path
        else required("common_eval.path", "モデルチェックポイントのパス", "model-path")
    )

    main(
        movie_path=args.movie_path,
        llm_path=llm_path,
        av_romanizer_path=Path(av_romanizer_value),
        model_path=Path(model_path_value),
    )
