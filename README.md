# audio-visual-toolkit

## How to Run

## Crop Mouth

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
uvx --python 3.12 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/vis' visualize-muavic --pkl-file $ROOT/muavic/metadata/de/train/_Hk4MOw9gsA.pkl --out-dir ./out
```

### Visualize Model

```sh
uvx --python 3.10 --from 'git+https://github.com/xhiroga/audio-visual-toolkit#subdirectory=packages/vis' visualize-model --model-path $ROOT/pretrained_models/av-romanizer/all/checkpoint_best.pt
```
