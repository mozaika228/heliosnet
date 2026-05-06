from __future__ import annotations

import random
import time

import cv2
import numpy as np

from core.service import BaseService


class SensorSimulator(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("sensor_simulator")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        sim_cfg = getattr(config, "simulation", {}) or {}
        self._enabled = bool(sim_cfg.get("enabled", False))
        self._fps = int(sim_cfg.get("fps", 5))
        self._size = tuple(sim_cfg.get("frame_size", [640, 480]))
        self._weather = str(sim_cfg.get("weather", "clear")).lower()
        self._noise = float(sim_cfg.get("noise", 0.02))
        self._interference = float(sim_cfg.get("interference", 0.0))
        self._sources = sim_cfg.get("sources", ["sim-cam-00"])
        self._last = 0.0
        self._idx = 0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last < (1.0 / max(1, self._fps)):
            return
        self._last = now
        src = self._sources[self._idx % len(self._sources)]
        self._idx += 1
        frame = self._build_frame(now)
        item = {"ts": now, "source_id": src, "frame": frame, "meta": {"simulated": True}}
        self.metrics.inc_frame(src)
        self.push(item)

    def _build_frame(self, now: float) -> np.ndarray:
        w, h = int(self._size[0]), int(self._size[1])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (25, 35, 45)
        # draw moving synthetic target
        cx = int((w * 0.1) + ((now * 60) % (w * 0.8)))
        cy = int(h * 0.5 + np.sin(now) * h * 0.2)
        cv2.rectangle(img, (cx - 40, cy - 70), (cx + 40, cy + 70), (80, 220, 80), 2)
        cv2.putText(img, "SIM_TARGET", (max(0, cx - 55), max(20, cy - 80)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)

        if self._weather == "rain":
            for _ in range(350):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                cv2.line(img, (x, y), (x + 2, y + 8), (180, 180, 180), 1)
        elif self._weather == "fog":
            fog = np.full_like(img, 180)
            img = cv2.addWeighted(img, 0.55, fog, 0.45, 0)

        if self._noise > 0:
            noise = np.random.normal(0.0, self._noise * 255.0, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if self._interference > 0 and random.random() < self._interference:
            x = random.randint(0, w - 20)
            cv2.rectangle(img, (x, 0), (min(w - 1, x + 20), h - 1), (255, 255, 255), -1)
            img = cv2.addWeighted(img, 0.8, np.zeros_like(img), 0.2, 0)
        return img

