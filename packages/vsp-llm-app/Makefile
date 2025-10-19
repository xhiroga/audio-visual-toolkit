include .env
export

split ?= test
# 処理対象データの分割数
nshard ?= 1
# 処理対象データの分割番号 (0 から nshard-1 まで)
rank ?= 0
feat_dir ?= av_feat
# 公開されている checkpoint を利用する場合
cluster ?= 10
feat_path ?= ${feat_dir}/${split}_${rank}_${nshard}.npy
leng_path ?= ${feat_dir}/${split}_${rank}_${nshard}.len
km_path ?= ${feat_dir}/new/${cluster}.km
# データのうち k-means 学習に使用する割合。今回はtestデータを用いるため、元々の件数が少ない。よって1.0とした。
# debug 0.05
percent ?= 0.05

${feat_path} ${leng_path}:
	uv run -m clustering.dump_hubert_feature ${TSV_DIR} ${split} ${W2V_PATH} 12 ${nshard} ${rank} ${feat_dir}

${km_path}: ${feat_path} ${leng_path}
	uv run -m clustering.learn_kmeans ${feat_dir} ${split} ${nshard} ${km_path} ${cluster} --percent ${percent}

main:
	uv run src/vsp_llm_app/main.py \
		--video-path ${VIDEO_PATH} \
		--model-path ${MODEL_PATH} \
		--w2v-path ${W2V_PATH} \
		--llm-path ${LLM_PATH}
