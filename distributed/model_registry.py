from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService


class ModelRegistry(BaseService):
    def __init__(self, config, metrics, raft=None):
        super().__init__("model_registry")
        self.config = config
        self.metrics = metrics
        self.raft = raft
        dist_cfg = getattr(config, "distributed", {})
        self._path = Path(dist_cfg.get("model_registry_path", "./data/model_registry.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._canary_sec = int(dist_cfg.get("canary_duration_sec", 120))
        self._rollback_flag = Path(dist_cfg.get("rollback_flag_path", "./data/model_rollback.flag"))
        self._state = self._load()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        now = time.time()
        state = self._state
        if state.get("status") == "canary":
            started = float(state.get("canary_started_at", 0.0))
            if self._rollback_flag.exists():
                self.rollback("rollback_flag")
                return
            if now - started >= self._canary_sec:
                self.promote_canary()

    def propose_candidate(self, version: str, model_path: str) -> bool:
        if self.raft is not None and not self.raft.is_leader:
            return False
        cmd = {"type": "model_canary", "version": version, "path": model_path}
        if self.raft is not None:
            ok = self.raft.propose(cmd)
            if not ok:
                return False
        self._state["canary"] = {"version": version, "path": model_path}
        self._state["status"] = "canary"
        self._state["canary_started_at"] = time.time()
        self._persist()
        return True

    def promote_canary(self) -> None:
        canary = self._state.get("canary")
        if not canary:
            return
        self._state["current"] = canary
        self._state["status"] = "stable"
        self._state["canary_started_at"] = 0.0
        self._persist()

    def rollback(self, reason: str) -> None:
        self._state["canary"] = None
        self._state["status"] = "rolled_back"
        self._state["rollback_reason"] = reason
        self._state["canary_started_at"] = 0.0
        self._persist()
        if self._rollback_flag.exists():
            try:
                self._rollback_flag.unlink()
            except Exception:
                pass

    def _load(self) -> dict:
        if not self._path.exists():
            initial = {
                "current": {"version": "initial", "path": self.config.inference.get("model", "")},
                "canary": None,
                "status": "stable",
                "canary_started_at": 0.0,
            }
            self._path.write_text(json.dumps(initial, ensure_ascii=True, indent=2), encoding="utf-8")
            return initial
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "current": {"version": "initial", "path": self.config.inference.get("model", "")},
                "canary": None,
                "status": "stable",
                "canary_started_at": 0.0,
            }

    def _persist(self) -> None:
        self._path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")
