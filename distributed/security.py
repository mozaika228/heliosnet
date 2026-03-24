from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any


class MessageSecurity:
    def __init__(self, shared_secret: str | None = None):
        self._secret = shared_secret or os.getenv("HELIOSNET_SHARED_SECRET", "")

    @property
    def enabled(self) -> bool:
        return bool(self._secret)

    def sign(self, payload: dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        mac = hmac.new(self._secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256)
        return mac.hexdigest()

    def verify(self, payload: dict[str, Any], signature: str | None) -> bool:
        if not self.enabled:
            return True
        if not signature:
            return False
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)
