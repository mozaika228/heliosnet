from __future__ import annotations

import time

from core.service import BaseService


class SyncEngine(BaseService):
    def __init__(self, config, metrics, energy=None):
        super().__init__("sync")
        self.config = config
        self.metrics = metrics
        self.energy = energy
        sync_cfg = getattr(config, "sync", {})
        self._interval = int(sync_cfg.get("interval_sec", 60))
        self._last = 0.0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        now = time.time()
        if self.energy is not None:
            profile = self.energy.current_profile()
            if profile is not None:
                self._interval = int(profile.sync_interval_sec)
        if now - self._last >= self._interval:
            self._last = now
