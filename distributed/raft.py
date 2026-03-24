from __future__ import annotations

import json
from pathlib import Path
import random
import time

from core.service import BaseService


class RaftController(BaseService):
    def __init__(self, config, metrics):
        super().__init__("raft")
        self.config = config
        self.metrics = metrics
        dist_cfg = getattr(config, "distributed", {})
        self._enabled = bool(dist_cfg.get("enabled", True))
        self._state_path = Path(dist_cfg.get("raft_state_path", "./data/raft_state.json"))
        self._log_path = Path(dist_cfg.get("raft_log_path", "./data/raft_log.jsonl"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._term = 0
        self._role = "follower"
        self._leader_id = None
        self._last_heartbeat = time.time()
        self._election_timeout = self._new_timeout()
        self._heartbeat_interval = float(dist_cfg.get("raft_heartbeat_sec", 1.0))
        self._last_tick = 0.0
        self._load_state()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if self._role == "leader":
            if now - self._last_tick >= self._heartbeat_interval:
                self._last_tick = now
                self._write_state(now)
            return
        if now - self._last_heartbeat >= self._election_timeout:
            self._start_election(now)

    def propose(self, command: dict) -> bool:
        if self._role != "leader":
            return False
        entry = {
            "ts": time.time(),
            "term": self._term,
            "leader_id": self.config.node_id,
            "command": command,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        return True

    def _start_election(self, now: float) -> None:
        self._term += 1
        # Single-node default: become leader immediately.
        self._role = "leader"
        self._leader_id = self.config.node_id
        self._last_heartbeat = now
        self._election_timeout = self._new_timeout()
        self._write_state(now)

    def _new_timeout(self) -> float:
        return random.uniform(1.5, 3.0)

    def _write_state(self, now: float) -> None:
        payload = {
            "node_id": self.config.node_id,
            "term": self._term,
            "role": self._role,
            "leader_id": self._leader_id,
            "ts": now,
        }
        self._state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._term = int(payload.get("term", 0))
        self._role = str(payload.get("role", "follower"))
        self._leader_id = payload.get("leader_id")
