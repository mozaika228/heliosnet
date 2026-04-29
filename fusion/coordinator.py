from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import cv2
import numpy as np

from core.service import BaseService


@dataclass
class Observation:
    ts: float
    source_id: str
    sensor: str
    frame: np.ndarray | None
    meta: dict


class FusionCoordinator(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("fusion")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        fusion_cfg = getattr(config, "fusion", {}) or {}
        self._enabled = bool(fusion_cfg.get("enabled", False))
        self._window_ms = int(fusion_cfg.get("sync_window_ms", 120))
        self._strategy = str(fusion_cfg.get("strategy", "late")).lower()
        self._sources = fusion_cfg.get("sources", {}) or {}
        self._pending: dict[str, dict[str, Observation]] = defaultdict(dict)

    def handle(self, item) -> None:
        if not self._enabled:
            self.push(item)
            return
        source_id = str(item.get("source_id", "source"))
        sensor = self._sensor_for_source(source_id)
        obs = Observation(
            ts=float(item.get("ts", time.time())),
            source_id=source_id,
            sensor=sensor,
            frame=item.get("frame"),
            meta=item.get("meta", {}) or {},
        )
        group_key = str(item.get("group_id", source_id))
        self._pending[group_key][sensor] = obs
        fused = self._fuse_if_ready(group_key)
        if fused is None:
            return
        item["frame"] = fused
        item["fusion"] = {
            "strategy": self._strategy,
            "sensors": sorted(list(self._pending[group_key].keys())),
            "group_key": group_key,
        }
        self._pending.pop(group_key, None)
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        ttl = self._window_ms / 1000.0
        for k in list(self._pending.keys()):
            bucket = self._pending[k]
            if not bucket:
                self._pending.pop(k, None)
                continue
            newest = max(o.ts for o in bucket.values())
            if now - newest > ttl:
                self._pending.pop(k, None)

    def _sensor_for_source(self, source_id: str) -> str:
        row = self._sources.get(source_id, {}) or {}
        return str(row.get("sensor", "rgb")).lower()

    def _fuse_if_ready(self, key: str) -> np.ndarray | None:
        bucket = self._pending.get(key, {})
        if not bucket:
            return None
        rgb = bucket.get("rgb")
        if rgb is None or rgb.frame is None:
            return None
        if self._strategy == "late":
            return rgb.frame
        thermal = bucket.get("thermal")
        if thermal is None or thermal.frame is None:
            return rgb.frame
        return _overlay_thermal(rgb.frame, thermal.frame)


def _overlay_thermal(rgb: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    if rgb is None or thermal is None:
        return rgb
    th = thermal
    if len(th.shape) == 3:
        th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
    th = cv2.resize(th, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    th_color = cv2.applyColorMap(th, cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb, 0.75, th_color, 0.25, 0)

