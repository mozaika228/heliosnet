from __future__ import annotations

import hashlib
from pathlib import Path


def verify_artifact_sha256(path: str, expected_sha256: str) -> bool:
    p = Path(path)
    if not p.exists() or not expected_sha256:
        return False
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.lower()


def mtls_config_ready(cert_path: str, key_path: str, ca_path: str) -> bool:
    return Path(cert_path).exists() and Path(key_path).exists() and Path(ca_path).exists()

