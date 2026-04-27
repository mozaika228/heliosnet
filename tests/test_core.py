from pathlib import Path

from core.node import Node


def _write_test_config(path: Path) -> None:
    path.write_text(
        """node_id: "ci-node"
ingest:
  sources: []
  max_fps: 1
  loop_files: false
  idle_sleep_ms: 1
  sync: false
  sync_timeout_ms: 100
inference:
  backend: "stub"
  model: ""
  device: "cpu"
  conf: 0.5
  classes: []
  preview: false
  batch_size: 1
tracker:
  backend: "iou"
  iou_thr: 0.3
  max_age: 5
  preview: false
events:
  rules: []
  store_path: "./data/test_events.jsonl"
  per_source_files: false
  csv_path: ""
sync:
  mode: "offline"
  interval_sec: 60
  batch_size: 10
  queue_path: "./data/test_sync_queue.jsonl"
  sent_path: "./data/test_sent.jsonl"
  acked_path: "./data/test_acked.txt"
distributed:
  enabled: false
  bind_host: "127.0.0.1"
  bind_port: 7946
  shared_secret: ""
  require_signed_control: false
  require_signed_commands: false
  gossip_interval_sec: 2
  peer_timeout_sec: 10
  bootstrap_peers: []
  cluster_state_path: "./data/test_cluster_state.json"
  raft_state_path: "./data/test_raft_state.json"
  raft_log_path: "./data/test_raft_log.jsonl"
  raft_heartbeat_sec: 1.0
  raft_min_cluster_size: 1
  model_registry_path: "./data/test_model_registry.json"
  model_commands_path: "./data/test_model_commands.jsonl"
  operator_commands_path: "./data/test_operator_commands.jsonl"
  canary_duration_sec: 120
  rollback_flag_path: "./data/test_rollback.flag"
  audit_log_path: "./data/test_audit_log.jsonl"
observability:
  metrics: false
  metrics_port: 9090
  web_ui: false
  web_ui_host: "127.0.0.1"
  web_ui_port: 18080
  telegram_enabled: false
  telegram_bot_token: ""
  telegram_chat_id: ""
  telegram_min_interval_sec: 1
safety:
  slo:
    max_e2e_latency_ms_p95: 400
    min_uptime_percent: 99.0
    max_miss_rate: 0.5
    window_size: 50
    breach_cooldown_sec: 1
  watchdog:
    enabled: false
  replay:
    enabled: false
mlops:
  drift:
    enabled: false
  evaluation:
    enabled: false
energy:
  simulated_percent: 50
  profiles:
    - level: "NORMAL"
      threshold: 0
      resolution: "640x480"
      fps: 5
      batch: 1
      sync_interval_sec: 60
runtime:
  tick_ms: 10
""",
        encoding="utf-8",
    )


def test_node_init():
    tmp_dir = Path("./tests/.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg = tmp_dir / "config.ci.yaml"
    _write_test_config(cfg)
    node = Node(str(cfg))
    assert node.config.node_id
