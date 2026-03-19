# HeliosNet Architecture

## Goals
- Multi-stream real-time ingest
- Edge inference and tracking with low latency
- Event extraction and local decisioning
- Offline-first operation with delayed sync

## High-Level Flow
1. Ingest -> frames
2. Inference -> detections
3. Tracker -> tracks
4. Events -> event records
5. Store + Sync

## Modules
- ingest/manager.py
- inference/engine.py
- tracker/coordinator.py
- events/processor.py
- events/store.py
- sync/engine.py
- distributed/gossip.py
- energy/scheduler.py
- core/node.py
