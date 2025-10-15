# fairseq-toolkit

## Run

### Compare Fairseq Models

```console
% uv run compare --model1 $ROOT/pretrained_models/avhubert/large_vox_iter5.pt --model2 $ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt --remove-prefix "av_romanizer.w2v_model." --out-dir ./out
Parameter metrics written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.csv
HTML report written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.html
Markdown report written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.md
```

### Compare Llama with/without LoRA

```console
% uv run compare-lora --llama-path meta-llama/Llama-3.2-3B --lora-path $ROOT/pretrained_models/zero-avsr/all/checkpoint_best.pt --out-dir ./out
Parameter metrics written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.csv
HTML report written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.html
Markdown report written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.md
```
