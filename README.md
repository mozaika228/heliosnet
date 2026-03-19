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
??? config/
?   ??? settings.py          # All config dataclasses + energy profiles
?   ??? config.example.yaml  # Sample deployment config
?
??? energy/
?   ??? scheduler.py         # Battery monitoring + energy-aware profile switching
?                              EnergyLevel: FULL / NORMAL / LOW / CRITICAL
?
??? ingest/
?   ??? manager.py           # Multi-stream reader (RTSP/USB/file)
?                              Adaptive FPS + resolution per energy profile
?
??? inference/
?   ??? engine.py            # YOLO inference
?                              Backends: Ultralytics | ONNX | GroundingDINO
?                              Dynamic batching ready
?
??? tracker/
?   ??? coordinator.py       # Simple IoU tracker
?                              Track lifecycle: NEW -> ACTIVE -> LOST -> REMOVED
?
??? events/
?   ??? store.py             # JSONL event store (offline-first)
?   ??? processor.py         # Anomaly detection, count snapshots, event emission
?
??? distributed/
?   ??? gossip.py            # UDP gossip protocol (stub)
?                              Node discovery, model propagation, cluster state
?
??? sync/
?   ??? engine.py            # Offline-first sync to central (stub)
?                              States: OFFLINE -> ONLINE -> BACKOFF
?
??? observability/
?   ??? metrics.py           # Prometheus metrics (stub)
?
??? core/
?   ??? node.py              # Main orchestrator (wires everything together)
?
??? tests/
    ??? test_core.py         # Async tests, no GPU/camera required
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

---

## Energy Profiles

| Level    | Threshold | Resolution | FPS | Batch | Sync interval |
|----------|-----------|------------|-----|-------|---------------|
| FULL     | > 80%     | 1280x720   | 30  | 8     | 30s           |
| NORMAL   | 40-80%    | 640x480    | 20  | 4     | 60s           |
| LOW      | 15-40%    | 416x416    | 10  | 2     | 5 min         |
| CRITICAL | < 15%     | 320x240    | 3   | 1     | 15 min        |

---

## Metrics (Prometheus)

Available at `http://<node>:9090/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `heliosnet_frames_total` | Counter | Frames processed per stream |
| `heliosnet_detections_total` | Counter | Detections per class/stream |
| `heliosnet_inference_latency_ms` | Histogram | Inference time distribution |
| `heliosnet_active_tracks` | Gauge | Live object tracks |
| `heliosnet_battery_percent` | Gauge | Battery level |
| `heliosnet_events_unsynced` | Gauge | Events pending sync |
| `heliosnet_anomalies_total` | Counter | Anomaly events per rule |

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
