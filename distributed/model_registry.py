from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService
from distributed.security import MessageSecurity


class ModelRegistry(BaseService):
    def __init__(self, config, metrics, raft=None, audit=None):
        super().__init__("model_registry")
        self.config = config
        self.metrics = metrics
        self.raft = raft
        self.audit = audit
        dist_cfg = getattr(config, "distributed", {})
        self._path = Path(dist_cfg.get("model_registry_path", "./data/model_registry.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._canary_sec = int(dist_cfg.get("canary_duration_sec", 120))
        self._rollback_flag = Path(dist_cfg.get("rollback_flag_path", "./data/model_rollback.flag"))
        self._commands_path = Path(dist_cfg.get("model_commands_path", "./data/model_commands.jsonl"))
        self._commands_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_cmd_size = 0
        self._security = MessageSecurity(str(dist_cfg.get("shared_secret", "")))
        self._require_signed = bool(dist_cfg.get("require_signed_commands", False))
        self._state = self._load()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        self._consume_commands()
        now = time.time()
        state = self._state
        if state.get("status") == "canary":
            started = float(state.get("canary_started_at", 0.0))
            if self._rollback_flag.exists():
                self.rollback("rollback_flag")
                return
            if now - started >= self._canary_sec:
                self.promote_canary()

    def current_target(self) -> tuple[str, str, str]:
        versions = self._state.get("versions", {})
        pinned = self._state.get("pinned_version")
        current = self._state.get("current_version", "initial")
        canary = self._state.get("canary_version")
        status = self._state.get("status", "stable")
        if pinned and pinned in versions:
            version = pinned
        elif status == "canary" and canary and canary in versions:
            version = canary
        else:
            version = current
        row = versions.get(version, {})
        backend = str(row.get("backend", self.config.inference.get("backend", "stub")))
        model_path = str(row.get("path", self.config.inference.get("model", "")))
        return version, backend, model_path

    def shadow_target(self) -> tuple[str, str, str] | None:
        versions = self._state.get("versions", {})
        shadow = self._state.get("shadow_version")
        if not shadow or shadow not in versions:
            return None
        row = versions.get(shadow, {})
        backend = str(row.get("backend", self.config.inference.get("backend", "stub")))
        model_path = str(row.get("path", self.config.inference.get("model", "")))
        return shadow, backend, model_path

    def propose_candidate(self, version: str, model_path: str) -> bool:
        if self.raft is not None and not self.raft.is_leader:
            return False
        cmd = {"type": "model_canary", "version": version, "path": model_path}
        if self.raft is not None:
            ok = self.raft.propose(cmd)
            if not ok:
                return False
        versions = self._state.setdefault("versions", {})
        versions[version] = {
            "path": model_path,
            "backend": self.config.inference.get("backend", "stub"),
            "created_at": time.time(),
        }
        self._state["canary_version"] = version
        self._state["status"] = "canary"
        self._state["canary_started_at"] = time.time()
        self._persist()
        if self.audit is not None:
            self.audit.write("model_canary_start", {"version": version, "path": model_path})
        return True

    def promote_canary(self) -> None:
        canary = self._state.get("canary_version")
        if not canary:
            return
        self._state["current_version"] = canary
        self._state["canary_version"] = None
        self._state["status"] = "stable"
        self._state["canary_started_at"] = 0.0
        self._persist()
        if self.audit is not None:
            self.audit.write("model_promote", {"version": canary})

    def rollback(self, reason: str) -> None:
        self._state["canary_version"] = None
        self._state["status"] = "rolled_back"
        self._state["rollback_reason"] = reason
        self._state["canary_started_at"] = 0.0
        self._persist()
        if self.audit is not None:
            self.audit.write("model_rollback", {"reason": reason})
        if self._rollback_flag.exists():
            try:
                self._rollback_flag.unlink()
            except Exception:
                pass

    def pin(self, version: str) -> bool:
        versions = self._state.get("versions", {})
        if version not in versions:
            return False
        self._state["pinned_version"] = version
        self._persist()
        if self.audit is not None:
            self.audit.write("model_pin", {"version": version})
        return True

    def unpin(self) -> None:
        self._state["pinned_version"] = None
        self._persist()
        if self.audit is not None:
            self.audit.write("model_unpin", {})

    def set_shadow(self, version: str) -> bool:
        versions = self._state.get("versions", {})
        if version not in versions:
            return False
        self._state["shadow_version"] = version
        self._persist()
        if self.audit is not None:
            self.audit.write("model_shadow_on", {"version": version})
        return True

    def clear_shadow(self) -> None:
        self._state["shadow_version"] = None
        self._persist()
        if self.audit is not None:
            self.audit.write("model_shadow_off", {})

    def _consume_commands(self) -> None:
        if not self._commands_path.exists():
            return
        size = self._commands_path.stat().st_size
        if size == self._last_cmd_size:
            return
        self._last_cmd_size = size
        with self._commands_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue
                if not self._verify_command(cmd):
                    if self.audit is not None:
                        self.audit.write("model_command_rejected", {"reason": "invalid_signature"})
                    continue
                self._apply_command(cmd)

    def _apply_command(self, cmd: dict) -> None:
        action = str(cmd.get("action", "")).lower()
        if action == "register":
            version = str(cmd.get("version", ""))
            path = str(cmd.get("path", ""))
            backend = str(cmd.get("backend", self.config.inference.get("backend", "stub")))
            if version and path:
                versions = self._state.setdefault("versions", {})
                versions[version] = {"path": path, "backend": backend, "created_at": time.time()}
                self._persist()
        elif action == "canary":
            version = str(cmd.get("version", ""))
            versions = self._state.get("versions", {})
            row = versions.get(version)
            if version and row:
                self.propose_candidate(version, str(row.get("path", "")))
        elif action == "promote":
            self.promote_canary()
        elif action == "rollback":
            self.rollback("manual_command")
        elif action == "pin":
            version = str(cmd.get("version", ""))
            self.pin(version)
        elif action == "unpin":
            self.unpin()
        elif action == "shadow_on":
            version = str(cmd.get("version", ""))
            self.set_shadow(version)
        elif action == "shadow_off":
            self.clear_shadow()

    def _verify_command(self, cmd: dict) -> bool:
        if not self._require_signed:
            return True
        signature = cmd.get("signature")
        if not isinstance(signature, str):
            return False
        payload = dict(cmd)
        payload.pop("signature", None)
        return self._security.verify(payload, signature)

    def _load(self) -> dict:
        if not self._path.exists():
            backend = self.config.inference.get("backend", "stub")
            model = self.config.inference.get("model", "")
            initial = {
                "versions": {"initial": {"path": model, "backend": backend, "created_at": time.time()}},
                "current_version": "initial",
                "canary_version": None,
                "pinned_version": None,
                "shadow_version": None,
                "status": "stable",
                "canary_started_at": 0.0,
            }
            self._path.write_text(json.dumps(initial, ensure_ascii=True, indent=2), encoding="utf-8")
            return initial
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            if "shadow_version" not in state:
                state["shadow_version"] = None
            return state
        except Exception:
            backend = self.config.inference.get("backend", "stub")
            model = self.config.inference.get("model", "")
            return {
                "versions": {"initial": {"path": model, "backend": backend, "created_at": time.time()}},
                "current_version": "initial",
                "canary_version": None,
                "pinned_version": None,
                "shadow_version": None,
                "status": "stable",
                "canary_started_at": 0.0,
            }

    def _persist(self) -> None:
        self._path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")
