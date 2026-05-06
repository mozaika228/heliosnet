from __future__ import annotations

import json
from pathlib import Path
import time

from core.service import BaseService
from distributed.security import MessageSecurity
from distributed.zero_trust import verify_artifact_sha256


class ProvenanceVerifierService(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("provenance")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._enabled = bool(cp_cfg.get("provenance_enabled", True))
        self._bundle = Path(cp_cfg.get("bundle_manifest_path", "./data/bundles.json"))
        self._interval = float(cp_cfg.get("provenance_interval_sec", 60))
        self._last = 0.0
        dist_cfg = getattr(config, "distributed", {}) or {}
        self._security = MessageSecurity(str(dist_cfg.get("shared_secret", "")))

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last < self._interval:
            return
        self._last = now
        if not self._bundle.exists():
            return
        try:
            data = json.loads(self._bundle.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in data.get("bundles", []):
            self._verify_bundle(row)

    def _verify_bundle(self, row: dict) -> None:
        artifact = str(row.get("artifact", ""))
        sha = str(row.get("sha256", ""))
        signature = str(row.get("signature", ""))
        meta = row.get("meta", {}) or {}
        ok_hash = verify_artifact_sha256(artifact, sha) if artifact and sha else False
        payload = {"artifact": artifact, "sha256": sha, "meta": meta}
        ok_sig = self._security.verify(payload, signature) if signature else False
        if self.audit is not None:
            self.audit.write(
                "provenance_verify",
                {"artifact": artifact, "ok_hash": ok_hash, "ok_signature": ok_sig},
            )

