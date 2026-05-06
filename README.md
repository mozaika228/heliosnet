# HeliosNet
[![CI](https://github.com/mozaika228/heliosnet/actions/workflows/ci.yml/badge.svg)](https://github.com/mozaika228/heliosnet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mozaika228/heliosnet/branch/main/graph/badge.svg)](https://codecov.io/gh/mozaika228/heliosnet)
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
|   `-- model_registry.py    # Canary/rollback model lifecycle state
|   `-- policy.py            # Policy engine (who can run what command)
|   `-- config_consensus.py  # Config apply log + consensus epoch state
|   `-- security.py          # HMAC signatures for control/messages
|   `-- zero_trust.py        # mTLS readiness + artifact hash verification
|   `-- zero_trust_service.py# Key rotation + artifact verification loop
|
|-- fusion/
|   `-- coordinator.py       # Multi-sensor sync/fusion (RGB/thermal baseline)
|   `-- geo.py               # Pixel-to-world approximation helpers
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
|   `-- live_state.py        # Real-time state cache for UI/events
|   `-- command_center.py    # Operator command loop
|
|-- ui/
|   `-- server.py            # Real-time digital twin web UI + SSE + command API
|
|-- notifier/
|   `-- telegram.py          # Telegram alert sink
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

### 9.1 Phone model guess (offline)
HeliosNet can estimate phone model for `cell phone` detections using a local catalog.
```
inference:
  phone_id_enabled: true
  phone_catalog_dir: "./data/phone_catalog"
  phone_min_score: 0.45
  phone_top_k: 3
  phone_embed_backend: "resnet18"   # resnet18 | histogram
  phone_embed_device: "cpu"         # cpu | cuda
  phone_cache_path: "./data/phone_catalog_index.npz"
```
Catalog format:
- put reference images in `./data/phone_catalog`
- filename pattern: `ModelName__anything.jpg`
- example: `iPhone_13__front.jpg`, `Samsung_A54__angle2.png`
- for `resnet18` backend install optional package: `pip install torchvision`

Detection output fields:
- `phone_model_guess`
- `phone_candidates` (`top_k` with similarity scores)

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
  bind_host: "0.0.0.0"
  bind_port: 7946
  cluster_state_path: "./data/cluster_state.json"
  raft_state_path: "./data/raft_state.json"
  raft_log_path: "./data/raft_log.jsonl"
  model_registry_path: "./data/model_registry.json"
  model_commands_path: "./data/model_commands.jsonl"
  canary_duration_sec: 120
  rollback_flag_path: "./data/model_rollback.flag"
```

### 11. Model lifecycle commands
Append JSON lines to `./data/model_commands.jsonl`:
```json
{"action":"register","version":"v2","path":"C:/models/yolo11s.pt","backend":"ultralytics"}
{"action":"canary","version":"v2"}
{"action":"promote"}
{"action":"rollback"}
{"action":"pin","version":"v2"}
{"action":"unpin"}
{"action":"shadow_on","version":"v2"}
{"action":"shadow_off"}
```

### 12. Enterprise security and audit
In `config/config.yaml`:
```
distributed:
  shared_secret: "change-me"
  require_signed_control: true
  require_signed_commands: true
  audit_log_path: "./data/audit_log.jsonl"
```

With `require_signed_commands: true`, each command line must contain `signature`.
Generate signed command:
```bash
python scripts/sign_model_command.py --secret "change-me" --action register --version v2 --path C:/models/yolo11s.pt --backend ultralytics
```

To force rollback during canary, create `./data/model_rollback.flag`.

### 12.1 Safety-Critical core
`config/config.yaml`:
```yaml
safety:
  slo:
    max_e2e_latency_ms_p95: 400
    min_uptime_percent: 99.0
    max_miss_rate: 0.25
  watchdog:
    enabled: true
    stale_heartbeat_sec: 5
  replay:
    enabled: true
    incidents_path: "./data/incidents"
```

Deterministic replay check:
```bash
python -m core.replay --incident <incident_id>
```

### 12.2 Edge MLOps
`config/config.yaml`:
```yaml
inference:
  shadow_enabled: true
  shadow_sample_rate: 10

mlops:
  drift:
    enabled: true
    data_threshold: 0.18
    concept_threshold: 0.15
  evaluation:
    enabled: true
    benchmark_path: "./data/edge_benchmark.jsonl"
    interval_sec: 600
```

Benchmark row format (`edge_benchmark.jsonl`):
```json
{"image":"./data/bench/frame_001.jpg","expected_count":1,"expected_class":0}
```

### 12.4 Pose state events
With pose model enabled, add rule:
```yaml
events:
  rules:
    - type: "pose_state"
      name: "pose_monitor"
      classes: [0]
      min_score: 0.55
```

Generated events include `pose_state` (`standing`, `sitting`, `hands_up`, `falling`, `unknown`).
`FALL_ALERT` is emitted automatically when `pose_state=falling`.

Gesture events from pose keypoints:
```yaml
events:
  rules:
    - type: "gesture_state"
      name: "gesture_monitor"
      classes: [0]
      min_score: 0.6
```
Detected gestures: `hands_up`, `left_hand_up`, `right_hand_up`, `arms_crossed`, `t_pose`.
Each detected gesture emits `GESTURE_ALERT`.

### 12.5 Mission AI planner
The planner decides what node should do next based on risk, battery and link quality.
```yaml
mission:
  enabled: true
  retask_every_sec: 5
  top_k_sources: 1
  alert_weight: 1.0
  slo_weight: 2.0
  fall_weight: 2.5
  drift_weight: 1.5
```

Planner outputs explainable `MISSION_ACTION` events with reasons and selected target sources.

### 12.3 Fusion + Distributed control plane
`config/config.yaml`:
```yaml
fusion:
  enabled: true
  strategy: "late"
  sync_window_ms: 120
  sources:
    src-00: {sensor: "rgb"}
    src-01: {sensor: "thermal"}
    src-02: {sensor: "depth"}
    src-03: {sensor: "radar"}

control_plane:
  policy_enabled: true
  policy_path: "./data/policy.json"
  config_commands_path: "./data/config_commands.jsonl"
  mtls_enabled: false
  key_rotation_interval_sec: 86400
```

Operator command with policy + consensus:
```json
{"actor":"local_operator","action":"config_apply","patch":{"inference":{"conf":0.55}}}
```

### 13. Custom classes (your dataset)
Create dataset skeleton:
```bash
python scripts/create_yolo_dataset.py --name my_objects --classes "pen,glasses,phone"
```

Annotate images in:
- `data/datasets/my_objects/images/{train,val,test}`
- `data/datasets/my_objects/labels/{train,val,test}`

Train:
```bash
python scripts/train_yolo.py --data data/datasets/my_objects/data.yaml --model yolo11n.pt --epochs 50 --device cpu
```

Use trained weights in config:
```yaml
inference:
  backend: "ultralytics"
  model: "runs/heliosnet/custom_train/weights/best.pt"
```

### 14. Human pose detection
Download pose model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

Run HeliosNet in pose mode:
```bash
powershell -ExecutionPolicy Bypass -File .\scripts\run_local.ps1 -Config .\config\pose.yaml
```

Notes:
- pose keypoints are attached to detections as `keypoints: [[x, y, conf], ...]`
- use `inference.preview: true` and `tracker.preview: false` to see skeletons

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

### Live Web UI (Digital Twin)
Enabled by default:
```
observability:
  web_ui: true
  web_ui_port: 8080
```
Open `http://localhost:8080` for:
- real-time source map and track/object counters
- SSE event feed
- operator command panel (battery/model actions)

### Telegram alerts
```
observability:
  telegram_enabled: true
  telegram_bot_token: "..."
  telegram_chat_id: "..."
```

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
