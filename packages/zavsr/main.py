# References:
# - zero-avsr/scripts/stage2/eval.sh
# - zero-avsr/stage2/eval.py

import argparse
from pathlib import Path

from fairseq import checkpoint_utils
from hydra.experimental import compose, initialize_config_dir
from transformers import AutoTokenizer


def _extract_model_state(checkpoint_state: dict) -> dict:
    """Load the model state_dict from a fairseq checkpoint dictionary."""

    for key in ("model", "model_state", "model_state_dict"):
        if key in checkpoint_state:
            return checkpoint_state[key]

    if "models" in checkpoint_state and checkpoint_state["models"]:
        return checkpoint_state["models"][0]

    raise KeyError("Expected model weights in checkpoint but none were found.")


def main(movie_path: Path, llm_path: str, av_romanizer_path: Path, model_path: Path):
    """
    Recognize the video file given as an argument with Zero-AVSR and return the result.
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    checkpoint_state = checkpoint_utils.load_checkpoint_to_cpu(str(model_path))
    model = _extract_model_state(checkpoint_state)

    print(f"{tokenizer=}, {model=}")


CONFIG_DIR = Path(__file__).resolve().parent / "conf"
CONFIG_NAME = "s2s_decode"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-path", type=Path, required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--av-romanizer-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    overrides = [
        f"common_eval.path={args.model_path}",
        f"override.llm_path={args.llm_path}",
        f"override.av_romanizer_path={args.av_romanizer_path}",
        "override.modalities=[video,audio]",
        f"common.user_dir={Path(__file__).resolve().parent}",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    main(
        movie_path=args.movie_path,
        llm_path=cfg.override.llm_path,
        av_romanizer_path=Path(cfg.override.av_romanizer_path),
        model_path=Path(cfg.common_eval.path),
    )
