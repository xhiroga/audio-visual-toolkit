include .env
export

run:
	uv run main.py \
		--config-dir conf \
		--config-name s2s_decode \
		--movie-path $$MUAVIC_DIR/mtedx/video/de/train/_Hk4MOw9gsA.mp4 \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt \
		--modalities video audio
