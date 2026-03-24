from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService


class GossipNode(BaseService):
    def __init__(self, config, metrics):
        super().__init__("gossip")
        self.config = config
        self.metrics = metrics
        dist_cfg = getattr(config, "distributed", {})
        self._enabled = bool(dist_cfg.get("enabled", True))
        self._interval = int(dist_cfg.get("gossip_interval_sec", 2))
        self._max_silence = int(dist_cfg.get("peer_timeout_sec", 10))
        self._peers = {
            str(p): {"last_seen": 0.0, "state": {}}
            for p in dist_cfg.get("bootstrap_peers", [])
        }
        self._last_tick = 0.0
        self._state_path = Path(dist_cfg.get("cluster_state_path", "./data/cluster_state.json"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_tick < self._interval:
            return
        self._last_tick = now
        self._cleanup(now)
        self._write_local_snapshot(now)

    def _cleanup(self, now: float) -> None:
        for peer, rec in self._peers.items():
            last_seen = float(rec.get("last_seen", 0.0))
            rec["alive"] = (now - last_seen) <= self._max_silence

    def _write_local_snapshot(self, now: float) -> None:
        payload = {
            "node_id": self.config.node_id,
            "ts": now,
            "peers": self._peers,
            "battery_percent": getattr(getattr(self.config, "energy", None), "simulated_percent", None),
        }
        self._state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    async def join(self, peers: list[tuple[str, int]]) -> None:
        for host, port in peers:
            key = f"{host}:{port}"
            if key not in self._peers:
                self._peers[key] = {"last_seen": time.time(), "alive": True, "state": {}}
