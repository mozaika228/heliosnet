from __future__ import annotations

import json
from pathlib import Path

from core.service import BaseService


class CommandCenter(BaseService):
    def __init__(self, config, metrics, energy, model_registry):
        super().__init__("command_center")
        self.config = config
        self.metrics = metrics
        self.energy = energy
        self.model_registry = model_registry
        dist_cfg = getattr(config, "distributed", {})
        self._cmd_path = Path(dist_cfg.get("operator_commands_path", "./data/operator_commands.jsonl"))
        self._cmd_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_size = 0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._cmd_path.exists():
            return
        size = self._cmd_path.stat().st_size
        if size == self._last_size:
            return
        self._last_size = size
        with self._cmd_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue
                self._apply(cmd)

    def _apply(self, cmd: dict) -> None:
        action = str(cmd.get("action", "")).lower()
        if action == "set_battery":
            percent = int(cmd.get("percent", 50))
            self.energy.update_battery(percent)
        elif action == "model_pin":
            version = str(cmd.get("version", ""))
            if version:
                self.model_registry.pin(version)
        elif action == "model_unpin":
            self.model_registry.unpin()
        elif action == "model_canary":
            version = str(cmd.get("version", ""))
            versions = self.model_registry._state.get("versions", {})
            row = versions.get(version)
            if version and row:
                self.model_registry.propose_candidate(version, str(row.get("path", "")))
        elif action == "model_promote":
            self.model_registry.promote_canary()
        elif action == "model_rollback":
            self.model_registry.rollback("operator_command")
        elif action == "model_shadow_on":
            version = str(cmd.get("version", ""))
            if version:
                self.model_registry.set_shadow(version)
        elif action == "model_shadow_off":
            self.model_registry.clear_shadow()
