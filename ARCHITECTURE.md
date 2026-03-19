# HeliosNet Architecture (MVP)

## Goals
- Multi-stream real-time ingest (4-16 sources)
- Edge inference and tracking with low latency
- Event extraction and local decisioning
- Offline-first operation with delayed sync

## High-Level Data Flow
1. Ingest: RTSP/WebRTC/Files -> frame normalization
2. Detect: YOLO (ONNX/TensorRT) -> detections
3. Track: ByteTrack/BoT-SORT -> tracks
4. Events: rules/metrics -> events/anomalies
5. Edge bus: local routing + buffered sync

## Core Services (Python MVP)
- ingest: multi-stream reader, sync, backpressure
- detect: model runner, batching, adaptive res
- track: tracker integration, ReID hook
- events: metrics, anomaly rules
- edge_bus: local event store + sync

## Runtime
- scheduler: energy-aware prioritization
- health: liveness/readiness, watchdog
- observability: Prometheus metrics

## Future (post-MVP)
- distributed coordination (Rust: gossip/Raft)
- federated/online learning
- C++/CUDA optimization for Jetson
