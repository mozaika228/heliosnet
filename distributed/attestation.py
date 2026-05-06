from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

from core.service import BaseService


class RemoteAttestationService(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("attestation")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._enabled = bool(cp_cfg.get("attestation_enabled", True))
        self._nonce_path = Path(cp_cfg.get("attestation_nonce_path", "./data/attestation_nonce.txt"))
        self._report_path = Path(cp_cfg.get("attestation_report_path", "./data/attestation_report.json"))
        self._boot_chain = Path(cp_cfg.get("boot_chain_path", "./data/boot_chain.json"))
        self._interval = float(cp_cfg.get("attestation_interval_sec", 60))
        self._last = 0.0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last < self._interval:
            return
        self._last = now
        nonce = self._read_nonce()
        report = self._build_report(nonce)
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        if self.audit is not None:
            self.audit.write("attestation_report", {"nonce": nonce, "platform": report.get("platform")})

    def _read_nonce(self) -> str:
        if self._nonce_path.exists():
            return self._nonce_path.read_text(encoding="utf-8").strip()
        return "no-nonce"

    def _build_report(self, nonce: str) -> dict:
        boot_hash = ""
        if self._boot_chain.exists():
            raw = self._boot_chain.read_bytes()
            boot_hash = hashlib.sha256(raw).hexdigest()
        payload = {
            "node_id": getattr(self.config, "node_id", "edge"),
            "ts": time.time(),
            "nonce": nonce,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "boot_chain_sha256": boot_hash,
        }
        payload["quote"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return payload

