from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService


class SyntheticIncidentGenerator(BaseService):
    def __init__(self, config, metrics, live_state=None, audit=None):
        super().__init__("synthetic_incidents")
        self.config = config
        self.metrics = metrics
        self.live_state = live_state
        self.audit = audit
        sim_cfg = getattr(config, "simulation", {}) or {}
        inc_cfg = sim_cfg.get("incidents", {}) or {}
        self._enabled = bool(inc_cfg.get("enabled", False))
        self._path = Path(inc_cfg.get("scenario_path", "./data/sim_incidents.jsonl"))
        self._interval = float(inc_cfg.get("interval_sec", 10))
        self._last = 0.0
        self._rows = self._load_rows()
        self._idx = 0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled or not self._rows:
            return
        now = time.time()
        if now - self._last < self._interval:
            return
        self._last = now
        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1
        evt = {
            "name": str(row.get("name", "SYNTHETIC_ALERT")),
            "ts": now,
            "source_id": str(row.get("source_id", "sim-cam-00")),
            "payload": row.get("payload", {}),
        }
        if self.live_state is not None:
            self.live_state.push_events([evt])
        if self.audit is not None:
            self.audit.write("synthetic_incident", evt)

    def _load_rows(self) -> list[dict]:
        if not self._path.exists():
            return []
        out = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

