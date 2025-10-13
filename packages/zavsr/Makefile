include .env
export

run:
	uv run main.py \
		--movie-path $$MUAVIC_DIR/mtedx/video/de/train/_Hk4MOw9gsA.mp4 \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--avhubert-path $$PRETRAINED_MODELS_DIR/avhubert/large_vox_iter5.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt
