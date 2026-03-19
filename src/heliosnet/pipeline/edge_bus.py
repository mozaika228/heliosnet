from __future__ import annotations

from __future__ import annotations

import json
from pathlib import Path
import time

from heliosnet.runtime.scheduler import BaseService


class LocalStore:
    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "events.jsonl"

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class EdgeBusService(BaseService):
    def __init__(self, config, metrics):
        super().__init__("edge_bus")
        self.config = config
        self.metrics = metrics
        edge_cfg = getattr(config, "edge_bus", {})
        self._store_all = bool(edge_cfg.get("store_all", False))
        self._store = LocalStore(edge_cfg.get("local_store", "./data/edge_bus"))

    def handle(self, item) -> None:
        events = item.get("events", [])
        if events:
            for evt in events:
                record = {
                    "ts": time.time(),
                    "source_id": item.get("source_id"),
                    "event": evt,
                    "tracks": item.get("tracks", []),
                    "detections": item.get("detections", []),
                }
                self._store.append(record)
                self.metrics.inc("events_written")
        elif self._store_all:
            record = {
                "ts": time.time(),
                "source_id": item.get("source_id"),
                "events": [],
                "tracks": item.get("tracks", []),
                "detections": item.get("detections", []),
            }
            self._store.append(record)
            self.metrics.inc("items_out")
