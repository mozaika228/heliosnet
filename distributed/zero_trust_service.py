from __future__ import annotations

import json
from pathlib import Path
import secrets
import time

from core.service import BaseService
from distributed.zero_trust import mtls_config_ready, verify_artifact_sha256


class ZeroTrustService(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("zero_trust")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._mtls_enabled = bool(cp_cfg.get("mtls_enabled", False))
        self._cert = str(cp_cfg.get("tls_cert_path", "./secrets/node.crt"))
        self._key = str(cp_cfg.get("tls_key_path", "./secrets/node.key"))
        self._ca = str(cp_cfg.get("tls_ca_path", "./secrets/ca.crt"))
        self._manifest = Path(cp_cfg.get("artifact_manifest_path", "./data/artifacts.json"))
        self._rotation_state = Path(cp_cfg.get("rotation_state_path", "./data/key_rotation.json"))
        self._rotate_sec = int(cp_cfg.get("key_rotation_interval_sec", 86400))
        self._last = 0.0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        now = time.time()
        if self._mtls_enabled:
            ready = mtls_config_ready(self._cert, self._key, self._ca)
            if not ready and self.audit is not None:
                self.audit.write("mtls_not_ready", {"cert": self._cert, "key": self._key, "ca": self._ca})
        self._verify_artifacts()
        if now - self._last >= self._rotate_sec:
            self._last = now
            self._rotate_keys(now)

    def _verify_artifacts(self) -> None:
        if not self._manifest.exists():
            return
        try:
            data = json.loads(self._manifest.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in data.get("artifacts", []):
            path = str(row.get("path", ""))
            sha = str(row.get("sha256", ""))
            if not path or not sha:
                continue
            ok = verify_artifact_sha256(path, sha)
            if not ok and self.audit is not None:
                self.audit.write("artifact_verify_failed", {"path": path})

    def _rotate_keys(self, now: float) -> None:
        state = {"ts": now, "key_id": secrets.token_hex(8)}
        self._rotation_state.parent.mkdir(parents=True, exist_ok=True)
        self._rotation_state.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
        if self.audit is not None:
            self.audit.write("key_rotated", {"key_id": state["key_id"]})

