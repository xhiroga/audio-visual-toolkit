include .env
export

MEMRAY_DIR := memray

debug:
	mkdir -p $(MEMRAY_DIR)
	uv run memray run -o $(MEMRAY_DIR)/memray-`date +%Y%m%d-%H%M%S`.bin -- \
		main.py \
		--video-path $$MUAVIC_DIR/muavic/de/video/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0020.mp4 \
		--audio-path $$MUAVIC_DIR/muavic/de/audio/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0020.wav \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--avhubert-path $$PRETRAINED_MODELS_DIR/avhubert/large_vox_iter5.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt \
		--log-level INFO

flamegraph:
	FILE=$$(find $(MEMRAY_DIR) -name '*.bin' | fzf) && \
		[ -n "$$FILE" ] && uv run memray flamegraph "$$FILE"
