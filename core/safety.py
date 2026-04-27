from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
import time

import cv2

from core.service import BaseService


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


class SLOMonitor(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("slo_monitor")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        safety_cfg = getattr(config, "safety", {}) or {}
        slo_cfg = safety_cfg.get("slo", {}) or {}
        self._max_p95_ms = float(slo_cfg.get("max_e2e_latency_ms_p95", 400.0))
        self._min_uptime = float(slo_cfg.get("min_uptime_percent", 99.0))
        self._max_miss = float(slo_cfg.get("max_miss_rate", 0.25))
        self._window = int(slo_cfg.get("window_size", 200))
        self._cooldown_sec = float(slo_cfg.get("breach_cooldown_sec", 15.0))
        self._lat_hist: deque[float] = deque(maxlen=self._window)
        self._miss_hist: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=self._window))
        self._last_breach: dict[str, float] = {}

    def handle(self, item) -> None:
        now = time.time()
        ts = float(item.get("ts", now))
        e2e_s = max(0.0, now - ts)
        self.metrics.observe_e2e(e2e_s)
        self._lat_hist.append(e2e_s * 1000.0)

        source_id = str(item.get("source_id", "source"))
        miss = 1 if not item.get("detections") else 0
        self._miss_hist[source_id].append(miss)
        self.metrics.miss_rate.labels(source_id=source_id).set(
            sum(self._miss_hist[source_id]) / max(1, len(self._miss_hist[source_id]))
        )

        breaches = []
        p95 = _percentile(list(self._lat_hist), 0.95)
        if len(self._lat_hist) >= 10 and p95 > self._max_p95_ms:
            breaches.append(("latency_p95", {"p95_ms": round(p95, 2), "limit_ms": self._max_p95_ms}))
        miss_rate = sum(self._miss_hist[source_id]) / max(1, len(self._miss_hist[source_id]))
        if len(self._miss_hist[source_id]) >= 10 and miss_rate > self._max_miss:
            breaches.append(
                (
                    "miss_rate",
                    {
                        "source_id": source_id,
                        "miss_rate": round(miss_rate, 4),
                        "limit": self._max_miss,
                    },
                )
            )

        for name, payload in breaches:
            if not self._emit_allowed(name, now):
                continue
            evt = {
                "name": "SLO_BREACH",
                "ts": now,
                "payload": {"slo": name, **payload},
                "source_id": source_id,
                "group_id": item.get("group_id"),
            }
            item.setdefault("events", []).append(evt)
            self.metrics.inc_slo_breach(name)
            if self.audit is not None:
                self.audit.write("slo_breach", evt["payload"])
        self.push(item)

    def tick(self) -> None:
        return

    def _emit_allowed(self, key: str, now: float) -> bool:
        prev = self._last_breach.get(key, 0.0)
        if now - prev < self._cooldown_sec:
            return False
        self._last_breach[key] = now
        return True


class IncidentRecorder(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("incident_recorder")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        safety_cfg = getattr(config, "safety", {}) or {}
        replay_cfg = safety_cfg.get("replay", {}) or {}
        self._enabled = bool(replay_cfg.get("enabled", True))
        self._root = Path(replay_cfg.get("incidents_path", "./data/incidents"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._image_quality = int(replay_cfg.get("image_quality", 85))

    def handle(self, item) -> None:
        if self._enabled and item.get("events"):
            incident_id = self._write_incident(item)
            if incident_id:
                for evt in item.get("events", []):
                    payload = evt.setdefault("payload", {})
                    payload.setdefault("incident_id", incident_id)
        self.push(item)

    def _write_incident(self, item: dict) -> str | None:
        ts = float(item.get("ts", time.time()))
        source = str(item.get("source_id", "source"))
        decision = {
            "events": item.get("events", []),
            "detections_count": len(item.get("detections", []) or []),
            "tracks_count": len(item.get("tracks", []) or []),
        }
        manifest = {
            "node_id": getattr(self.config, "node_id", "edge"),
            "source_id": source,
            "ts": ts,
            "group_id": item.get("group_id"),
            "model": getattr(self.config, "inference", {}).get("model", ""),
            "backend": getattr(self.config, "inference", {}).get("backend", ""),
            "decision": decision,
        }
        decision_blob = json.dumps(decision, sort_keys=True, ensure_ascii=True)
        manifest["decision_hash"] = hashlib.sha1(decision_blob.encode("utf-8")).hexdigest()
        incident_id = f"{int(ts * 1000)}_{source}"
        out_dir = self._root / incident_id
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = item.get("frame")
        if frame is not None:
            img_path = out_dir / "frame.jpg"
            cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._image_quality])
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        (out_dir / "item.json").write_text(
            json.dumps(_strip_frame(item), ensure_ascii=True, indent=2), encoding="utf-8"
        )
        if self.audit is not None:
            self.audit.write("incident_recorded", {"incident_id": incident_id, "source_id": source})
        return incident_id


def _strip_frame(item: dict) -> dict:
    out = dict(item)
    out.pop("frame", None)
    return out
