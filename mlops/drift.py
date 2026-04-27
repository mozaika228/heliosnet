from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from core.service import BaseService


class DriftMonitorService(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("drift_monitor")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        mlops_cfg = getattr(config, "mlops", {}) or {}
        drift_cfg = mlops_cfg.get("drift", {}) or {}
        self._enabled = bool(drift_cfg.get("enabled", True))
        self._alpha = float(drift_cfg.get("ema_alpha", 0.03))
        self._data_thr = float(drift_cfg.get("data_threshold", 0.18))
        self._concept_thr = float(drift_cfg.get("concept_threshold", 0.15))
        self._state_path = Path(drift_cfg.get("state_path", "./data/drift_state.json"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._cooldown_sec = float(drift_cfg.get("event_cooldown_sec", 30.0))
        self._last_evt = 0.0

    def handle(self, item) -> None:
        if not self._enabled:
            self.push(item)
            return
        source_id = str(item.get("source_id", "source"))
        frame = item.get("frame")
        dets = item.get("detections", []) or []
        if frame is None:
            self.push(item)
            return
        try:
            brightness = float(np.mean(frame) / 255.0)
        except Exception:
            brightness = 0.0
        conf = 0.0
        if dets:
            conf = float(np.mean([float(d.get("conf", 0.0)) for d in dets]))

        s = self._state.setdefault(source_id, {"b_ema": brightness, "c_ema": conf})
        prev_b = float(s.get("b_ema", brightness))
        prev_c = float(s.get("c_ema", conf))
        s["b_ema"] = (1.0 - self._alpha) * prev_b + self._alpha * brightness
        s["c_ema"] = (1.0 - self._alpha) * prev_c + self._alpha * conf
        data_score = abs(brightness - s["b_ema"])
        concept_score = abs(conf - s["c_ema"])
        self.metrics.set_drift_score(source_id, "data", data_score)
        self.metrics.set_drift_score(source_id, "concept", concept_score)
        now = time.time()
        if (data_score > self._data_thr or concept_score > self._concept_thr) and now - self._last_evt > self._cooldown_sec:
            evt = {
                "name": "DRIFT_ALERT",
                "ts": now,
                "source_id": source_id,
                "group_id": item.get("group_id"),
                "payload": {
                    "data_score": round(data_score, 5),
                    "concept_score": round(concept_score, 5),
                    "data_threshold": self._data_thr,
                    "concept_threshold": self._concept_thr,
                },
            }
            item.setdefault("events", []).append(evt)
            self._last_evt = now
            if self.audit is not None:
                self.audit.write("drift_alert", evt["payload"])
        self._persist_state()
        self.push(item)

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _persist_state(self) -> None:
        self._state_path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")
