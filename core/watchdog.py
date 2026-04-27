from __future__ import annotations

import time

from core.service import BaseService


class WatchdogService(BaseService):
    def __init__(self, config, metrics, services: dict[str, BaseService], audit=None):
        super().__init__("watchdog")
        self.config = config
        self.metrics = metrics
        self.services = services
        self.audit = audit
        safety_cfg = getattr(config, "safety", {}) or {}
        wd_cfg = safety_cfg.get("watchdog", {}) or {}
        self._enabled = bool(wd_cfg.get("enabled", True))
        self._stale_sec = float(wd_cfg.get("stale_heartbeat_sec", 5.0))
        self._mode = "normal"
        self._last_action_ts = 0.0
        self._action_cooldown = float(wd_cfg.get("action_cooldown_sec", 5.0))
        self._degraded_fps_cap = int(wd_cfg.get("degraded_fps_cap", 8))
        self._safe_fps_cap = int(wd_cfg.get("safe_fps_cap", 4))
        self._degraded_resolution = _parse_hw(str(wd_cfg.get("degraded_resolution", "640x480")))
        self._safe_resolution = _parse_hw(str(wd_cfg.get("safe_resolution", "416x416")))
        self._critical = set(
            wd_cfg.get(
                "critical_services",
                ["ingest", "inference", "drift_monitor", "tracker", "events", "slo_monitor", "incident_recorder", "event_store"],
            )
        )

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            self.metrics.set_runtime_mode("normal")
            return
        now = time.time()
        stale = []
        total = 0
        alive = 0
        for name, svc in self.services.items():
            if name == self.name:
                continue
            if self._critical and name not in self._critical:
                continue
            total += 1
            hb = float(svc.last_heartbeat())
            if hb <= 0 or now - hb > self._stale_sec:
                stale.append(name)
            else:
                alive += 1

        uptime_ratio = alive / max(1, total)
        self.metrics.set_uptime_ratio(uptime_ratio)

        if not stale:
            if self._mode != "normal":
                self._set_mode("normal")
            return

        if now - self._last_action_ts < self._action_cooldown:
            return
        self._last_action_ts = now
        if self._mode == "normal":
            self._set_mode("degraded", stale)
        else:
            self._set_mode("safe", stale)

    def _set_mode(self, mode: str, stale: list[str] | None = None) -> None:
        self._mode = mode
        ingest = self.services.get("ingest")
        inference = self.services.get("inference")
        if mode == "normal":
            if hasattr(ingest, "set_runtime_overrides"):
                ingest.set_runtime_overrides(0, None)
            if hasattr(inference, "set_runtime_mode"):
                inference.set_runtime_mode("normal")
            self.metrics.watchdog_action("recover_normal")
            self.metrics.set_runtime_mode("normal")
            if self.audit is not None:
                self.audit.write("watchdog_recover", {})
            return
        if mode == "degraded":
            if hasattr(ingest, "set_runtime_overrides"):
                ingest.set_runtime_overrides(self._degraded_fps_cap, self._degraded_resolution)
            if hasattr(inference, "set_runtime_mode"):
                inference.set_runtime_mode("degraded")
            self.metrics.watchdog_action("degrade")
            self.metrics.set_runtime_mode("degraded")
            if self.audit is not None:
                self.audit.write("watchdog_degrade", {"stale_services": stale or []})
            return
        if hasattr(ingest, "set_runtime_overrides"):
            ingest.set_runtime_overrides(self._safe_fps_cap, self._safe_resolution)
        if hasattr(inference, "set_runtime_mode"):
            inference.set_runtime_mode("safe")
        self.metrics.watchdog_action("safe_mode")
        self.metrics.set_runtime_mode("safe")
        if self.audit is not None:
            self.audit.write("watchdog_safe_mode", {"stale_services": stale or []})


def _parse_hw(value: str) -> tuple[int, int] | None:
    if "x" not in value:
        return None
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except Exception:
        return None
