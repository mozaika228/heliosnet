from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from core.safety import IncidentRecorder, SLOMonitor


class _Gauge:
    def labels(self, **kwargs):
        return self

    def set(self, value):
        return None


class _Metrics:
    def __init__(self):
        self.miss_rate = _Gauge()
        self.breaches = 0

    def observe_e2e(self, seconds: float) -> None:
        return None

    def inc_slo_breach(self, name: str) -> None:
        self.breaches += 1


def _tmp_dir(tag: str) -> Path:
    p = Path("./tests/.tmp") / f"{tag}_{int(time.time() * 1000)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        node_id="test-node",
        inference={"model": "m.pt", "backend": "stub"},
        safety={
            "slo": {
                "max_e2e_latency_ms_p95": 1000,
                "max_miss_rate": 0.2,
                "window_size": 20,
                "breach_cooldown_sec": 0,
            },
            "replay": {"enabled": True, "incidents_path": str(tmp_path / "incidents")},
        },
    )


def test_slo_monitor_emits_breach():
    tmp_path = _tmp_dir("slo")
    cfg = _cfg(tmp_path)
    m = _Metrics()
    svc = SLOMonitor(cfg, m, audit=None)
    out = []
    svc.set_next(SimpleNamespace(handle=lambda item: out.append(item)))
    for _ in range(15):
        svc.handle({"ts": 0.0, "source_id": "src-01", "detections": []})
    assert out
    assert any(evt.get("name") == "SLO_BREACH" for evt in out[-1].get("events", []))
    assert m.breaches >= 1


def test_incident_replay_hash():
    tmp_path = _tmp_dir("replay")
    cfg = _cfg(tmp_path)
    m = _Metrics()
    svc = IncidentRecorder(cfg, m, audit=None)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    captured = []
    svc.set_next(SimpleNamespace(handle=lambda item: captured.append(item)))
    svc.handle(
        {
            "ts": 100.0,
            "source_id": "src-01",
            "frame": frame,
            "detections": [{"cls": 0, "conf": 0.9, "bbox": [1, 1, 10, 10]}],
            "tracks": [],
            "events": [{"name": "x", "ts": 100.0, "payload": {}}],
        }
    )
    assert captured
    inc_id = captured[0]["events"][0]["payload"]["incident_id"]
    cmd = [
        sys.executable,
        "-m",
        "core.replay",
        "--incident",
        inc_id,
        "--root",
        str(tmp_path / "incidents"),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
