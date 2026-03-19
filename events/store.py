from __future__ import annotations

import json
from pathlib import Path

from core.service import BaseService


class EventStore(BaseService):
    def __init__(self, config, metrics):
        super().__init__("event_store")
        self.config = config
        self.metrics = metrics
        events_cfg = getattr(config, "events", {})
        self._path = Path(events_cfg.get("store_path", "./data/events.jsonl"))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, item) -> None:
        events = item.get("events", [])
        if events:
            with self._path.open("a", encoding="utf-8") as f:
                for evt in events:
                    f.write(json.dumps(evt, ensure_ascii=True) + "\n")
            self.metrics.inc("events_written")
        self.push(item)
