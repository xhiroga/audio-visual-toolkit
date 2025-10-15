# audio-visual-toolkit

## How to Run

### Crop Mouth

```sh
uvx --python 3.12 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/avt' crop-mouth --video-file in.mp4 --out-dir out
# Extract the mouth area using MediaPipe FaceMesh and export LFROI_<stem>.mp4
```

### Validate Label

```sh
uvx --python 3.12 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/avt' validate-label -h
```

### Visualize [MuAViC](https://github.com/facebookresearch/muavic) Face Metadata

```sh
uvx --python 3.12 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/avt' visualize-muavic \
  --pkl-file $ROOT/metadata/de/train/_Hk4MOw9gsA.pkl \
  --out-dir ./out

# Overlapping
uvx --python 3.12 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/avt' visualize-muavic \
  --pkl-file $ROOT/metadata/de/train/_Hk4MOw9gsA.pkl \
  --out-dir ./out \
  --video-file $ROOT/mtedx/video/de/train/_Hk4MOw9gsA.mp4 \
  --segments-file $ROOT/mtedx/de-de/data/train/txt/segments
```

### Dump Fairseq Model

```sh
uvx --python 3.10 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/fairseq-toolkit' dump --model-path $ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt
```

### Compare Fairseq Models

```console
% uvx --python 3.10 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/fairseq-toolkit' compare --model1 $ROOT/pretrained_models/avhubert/large_vox_iter5.pt --model2 $ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt --remove-prefix "av_romanizer.w2v_model." --out-dir ./out
Parameter metrics written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.csv
HTML report written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.html
Markdown report written to ./out/large_vox_iter5_pt-vs-checkpoint_best_pt.md
```

### Compare Llama with/without LoRA

```console
% uvx --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/fairseq-toolkit' compare-lora --llama-path meta-llama/Llama-3.2-3B --lora-path $ROOT/pretrained_models/zero-avsr/all/checkpoint_best.pt --out-dir ./out
Parameter metrics written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.csv
HTML report written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.html
Markdown report written to ./out/Llama-3_2-3B-vs-checkpoint_best_pt.md
```

### Run Zero-AVSR with mp4 file

WIP...
