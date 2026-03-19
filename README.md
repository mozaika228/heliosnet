# HeliosNet

Distributed real-time computer vision on the edge.

## What It Is
HeliosNet is a multi-stream, offline-first edge CV cluster for large territories.

## Quick Start (Scaffold)
1. Create a venv and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\pip install -r requirements.txt`
2. Run a local pipeline:
   - `.\scripts\run_local.ps1 -Config .\configs\dev.yaml`

## Docs
- `ARCHITECTURE.md` for system overview

## Source Formats
- `file:./path/to/video.mp4`
- `rtsp://user:pass@host/stream`
- `webcam:0`
- `image:./path/to/image.jpg`
- `dir:./path/to/images`

## Detection Backend
- `detect.backend: "stub"` (default)
- `detect.backend: "onnxruntime"` (requires `onnxruntime` + a valid ONNX model)
  - Current parser expects NMS-like output `[N,6]` or `[1,N,6]` (xyxy, conf, cls)

## Event Rules
Examples in `configs/dev.yaml`:
- `count_threshold` emits when object count >= threshold
- `zone_entry` emits when object center enters a rectangle
