# HeliosNet
Distributed Real-Time Computer Vision for Edge Networks

```
Multi-source video -> Edge inference -> Object tracking -> Events -> Sync to central
```

Autonomous, solar-powered edge clusters for monitoring large areas.
Works offline. Syncs when connected. Self-coordinates via gossip.

---

## Architecture

```
heliosnet/
|-- config/
|   |-- settings.py          # All config dataclasses + energy profiles
|   `-- config.example.yaml  # Sample deployment config
|
|-- energy/
|   `-- scheduler.py         # Battery monitoring + energy-aware profile switching
|                             # EnergyLevel: FULL / NORMAL / LOW / CRITICAL
|
|-- ingest/
|   `-- manager.py           # Multi-stream reader (RTSP/USB/file)
|                             # Adaptive FPS + resolution per energy profile
|
|-- inference/
|   `-- engine.py            # YOLO inference
|                             # Backends: Ultralytics | ONNX | GroundingDINO
|                             # Dynamic batching ready
|
|-- tracker/
|   `-- coordinator.py       # ByteTrack (via supervision) or IoU tracker
|                             # Track lifecycle: NEW -> ACTIVE -> LOST -> REMOVED
|
|-- events/
|   |-- store.py             # JSONL event store (offline-first)
|   `-- processor.py         # Anomaly detection, count snapshots, event emission
|
|-- distributed/
|   |-- gossip.py            # Cluster membership + peer state snapshots
|   `-- raft.py              # Raft-like control-plane state and config log
|                             # Leader state, terms, command proposals
|
|-- sync/
|   `-- engine.py            # Offline-first sync queue + idempotent delivery
|                             # Modes: offline | local
|
|-- observability/
|   `-- metrics.py           # Prometheus metrics (fps, p95 latency, objs/sec)
|
|-- core/
|   `-- node.py              # Main orchestrator (wires everything together)
|
`-- tests/
    `-- test_core.py         # Async tests, no GPU/camera required
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download a model (Ultralytics)
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### 3. Run with default demo config (CPU)
```bash
python -m core.node
```

### 4. Run with your own config
```bash
cp config/config.example.yaml config/config.yaml
# Edit streams, model path, node_id ...
python -m core.node config/config.yaml
```

### 5. Run tests (no GPU needed)
```bash
pytest tests/ -v
```

### 6. Batch inference (optional)
In `config/config.yaml`:
```
inference:
  batch_size: 2
  batch_timeout_ms: 20
```

### 7. Multi-stream sync (optional)
In `config/config.yaml`:
```
ingest:
  sync: true
  sync_timeout_ms: 200
```

### 8. Track IDs and labels
Enable tracker preview to see `label#track_id` per object:
```
tracker:
  preview: true
  preview_window: "HeliosNet"
```

### 9. Event store options
In `config/config.yaml`:
```
events:
  store_path: "./data/events.jsonl"
  per_source_files: true
  csv_path: "./data/events.csv"
```

### 10. Offline sync queue and control plane
In `config/config.yaml`:
```
sync:
  mode: "local"
  interval_sec: 60
  queue_path: "./data/sync_queue.jsonl"
  sent_path: "./data/sent_events.jsonl"
  acked_path: "./data/sync_acked.txt"

distributed:
  enabled: true
  cluster_state_path: "./data/cluster_state.json"
  raft_state_path: "./data/raft_state.json"
  raft_log_path: "./data/raft_log.jsonl"
```

---

## Energy Profiles

| Level    | Threshold | Resolution | FPS | Batch | Sync interval |
|----------|-----------|------------|-----|-------|---------------|
| FULL     | > 80%     | 1280x720   | 30  | 8     | 30s           |
| NORMAL   | 40-80%    | 640x480    | 20  | 4     | 60s           |
| LOW      | 15-40%    | 416x416    | 10  | 2     | 5 min         |
| CRITICAL | < 15%     | 320x240    | 3   | 1     | 15 min        |

### Energy-aware mode
HeliosNet adjusts FPS and resolution based on battery level.
In `config/config.yaml`:
```
energy:
  simulated_percent: 55
```

---

## Metrics (Prometheus)

Available at `http://<node>:9090/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `heliosnet_frames_total` | Counter | Frames processed per stream |
| `heliosnet_detections_total` | Counter | Detections per class/stream |
| `heliosnet_inference_latency_seconds` | Histogram | Inference latency distribution |
| `heliosnet_objects_per_frame` | Gauge | Objects per frame (instant) |
| `heliosnet_confidence_mean` | Gauge | Mean confidence per stream (EMA) |
| `heliosnet_tracks_total` | Counter | Tracks updated |
| `heliosnet_events_built_total` | Counter | Events built |
| `heliosnet_events_written_total` | Counter | Events written |

### Dashboard (Grafana)
Start Prometheus + Grafana:
```bash
docker-compose up -d
```
Grafana: `http://localhost:3000` (admin/admin)

Prometheus is mapped to `http://localhost:9091`.
If `host.docker.internal` does not resolve, update `docker/prometheus/prometheus.yml`.

---

## Distributed Gossip

Nodes discover each other automatically via UDP gossip on port 7946.
Bootstrap by pointing at any known peer:
```python
await node.gossip.join([("192.168.1.2", 7946)])
```

Events propagated:
- PUSH_STATE
- MODEL_UPDATE
- EVENT_NOTIFY
- PING / PING_ACK

---

## Phase Roadmap

```
Phase 1  Single-node, single-stream (current)
Phase 2  TensorRT export pipeline for Jetson
Phase 3  Federated learning (Flower integration)
Phase 4  Multi-node Raft for model consensus
Phase 5  Rust rewrite of gossip + sync layer
Phase 6  C++/CUDA custom kernels for Jetson Orin
```

---

## Hardware Targets

| Device         | Inference backend | Expected FPS (YOLOv8m) |
|----------------|-------------------|------------------------|
| Jetson Orin NX | TensorRT INT8     | 60-120 FPS             |
| Jetson Nano    | TensorRT FP16     | 15-25 FPS              |
| RPi 5          | ONNX CPU          | 3-7 FPS                |
| x86 + RTX 3060 | ONNX CUDA         | 80-200 FPS             |

Export to TensorRT:
```python
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
model.export(format="engine", device=0, half=True)
```
