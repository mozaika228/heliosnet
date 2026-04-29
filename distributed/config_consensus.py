from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService


class ConfigConsensusService(BaseService):
    def __init__(self, config, metrics, raft=None, audit=None):
        super().__init__("config_consensus")
        self.config = config
        self.metrics = metrics
        self.raft = raft
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._state_path = Path(cp_cfg.get("config_state_path", "./data/config_state.json"))
        self._commands_path = Path(cp_cfg.get("config_commands_path", "./data/config_commands.jsonl"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._commands_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_size = 0
        self._state = self._load_state()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._commands_path.exists():
            return
        size = self._commands_path.stat().st_size
        if size == self._last_size:
            return
        self._last_size = size
        with self._commands_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue
                self.apply_command(cmd)

    def apply_command(self, cmd: dict) -> bool:
        action = str(cmd.get("action", "")).lower()
        if action != "apply":
            return False
        patch = cmd.get("patch", {}) or {}
        if self.raft is not None and self.raft.is_leader:
            self.raft.propose({"type": "config_apply", "patch": patch, "ts": time.time()})
        self._state["epoch"] = int(self._state.get("epoch", 0)) + 1
        self._state["last_patch"] = patch
        self._state["last_ts"] = time.time()
        self._persist()
        if self.audit is not None:
            self.audit.write("config_apply", {"epoch": self._state["epoch"]})
        return True

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            state = {"epoch": 0, "last_patch": {}, "last_ts": 0.0}
            self._state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
            return state
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"epoch": 0, "last_patch": {}, "last_ts": 0.0}

    def _persist(self) -> None:
        self._state_path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")

