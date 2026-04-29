from __future__ import annotations

import json
from pathlib import Path

from core.service import BaseService


class PolicyEngine(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("policy")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._enabled = bool(cp_cfg.get("policy_enabled", True))
        self._path = Path(cp_cfg.get("policy_path", "./data/policy.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._policy = self._load_policy()

    def handle(self, item) -> None:
        self.push(item)

    def allow(self, actor: str, action: str) -> bool:
        if not self._enabled:
            return True
        actors = self._policy.get("actors", {})
        allowed = set(actors.get(actor, []))
        if action in allowed or "*" in allowed:
            return True
        if self.audit is not None:
            self.audit.write("policy_deny", {"actor": actor, "action": action})
        return False

    def _load_policy(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        default = {
            "actors": {
                "local_operator": [
                    "set_battery",
                    "model_pin",
                    "model_unpin",
                    "model_canary",
                    "model_promote",
                    "model_rollback",
                    "model_shadow_on",
                    "model_shadow_off",
                    "config_apply",
                ]
            }
        }
        self._path.write_text(json.dumps(default, ensure_ascii=True, indent=2), encoding="utf-8")
        return default

