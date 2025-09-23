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
