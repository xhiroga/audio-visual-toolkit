include .env
export

MEMRAY_DIR := memray

webui:
	uv run --extra webui src/zero_avsr_app/webui.py \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--avhubert-path $$PRETRAINED_MODELS_DIR/avhubert/large_vox_iter5.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt

avsr:
	mkdir -p $(MEMRAY_DIR)
	uv run memray run -o $(MEMRAY_DIR)/memray-`date +%Y%m%d-%H%M%S`.bin -- \
		src/zero_avsr_app/main.py \
		--video-path $$MUAVIC_DIR/muavic/de/video/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0020.mp4 \
		--audio-path $$MUAVIC_DIR/muavic/de/audio/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0020.wav \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--avhubert-path $$PRETRAINED_MODELS_DIR/avhubert/large_vox_iter5.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt \
		--mode avsr \
		--lang deu \
		--log-level INFO

vsr:
	mkdir -p $(MEMRAY_DIR)
	uv run memray run -o $(MEMRAY_DIR)/memray-`date +%Y%m%d-%H%M%S`.bin -- \
		src/zero_avsr_app/main.py \
		--video-path $$MUAVIC_DIR/muavic/de/video/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0020.mp4 \
		--llm-path meta-llama/Llama-3.2-3B \
		--av-romanizer-path $$PRETRAINED_MODELS_DIR/av-romanizer/all/checkpoint_best.pt \
		--avhubert-path $$PRETRAINED_MODELS_DIR/avhubert/large_vox_iter5.pt \
		--model-path $$PRETRAINED_MODELS_DIR/zero-avsr/all/checkpoint_best.pt \
		--mode vsr \
		--lang deu \
		--log-level INFO

flamegraph:
	FILE=$$(find $(MEMRAY_DIR) -name '*.bin' | fzf) && \
		[ -n "$$FILE" ] && uv run memray flamegraph "$$FILE"
