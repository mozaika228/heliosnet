from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


class AuditLog:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash()

    def write(self, event_type: str, details: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event_type": event_type,
            "details": details,
            "prev_hash": self._last_hash,
        }
        body = json.dumps(row, ensure_ascii=True, sort_keys=True)
        row_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        row["hash"] = row_hash
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._last_hash = row_hash

    def _load_last_hash(self) -> str:
        if not self._path.exists():
            return ""
        last = ""
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                last = str(row.get("hash", last))
        return last
