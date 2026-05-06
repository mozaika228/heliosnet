from __future__ import annotations

import csv
import json
from pathlib import Path

from core.service import BaseService


class EventStore(BaseService):
    def __init__(self, config, metrics, live_state=None):
        super().__init__("event_store")
        self.config = config
        self.metrics = metrics
        self.live_state = live_state
        events_cfg = getattr(config, "events", {})
        self._path = Path(events_cfg.get("store_path", "./data/events.jsonl"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._per_source = bool(events_cfg.get("per_source_files", False))
        self._csv_path = events_cfg.get("csv_path", "")

    def handle(self, item) -> None:
        events = item.get("events", [])
        if events:
            if self.live_state is not None:
                self.live_state.push_events(events)
            if self._per_source and item.get("source_id"):
                src = item.get("source_id")
                path = self._path.parent / f"events_{src}.jsonl"
                with path.open("a", encoding="utf-8") as f:
                    for evt in events:
                        f.write(json.dumps(evt, ensure_ascii=True) + "\n")
            else:
                with self._path.open("a", encoding="utf-8") as f:
                    for evt in events:
                        f.write(json.dumps(evt, ensure_ascii=True) + "\n")
            if self._csv_path:
                self._append_csv(events)
            self.metrics.inc("events_written")
        self.push(item)

    def _append_csv(self, events: list[dict]) -> None:
        path = Path(self._csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fieldnames = [
            "name",
            "ts",
            "source_id",
            "group_id",
            "payload_json",
        ]
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            for evt in events:
                flat = {
                    "name": evt.get("name"),
                    "ts": evt.get("ts"),
                    "source_id": evt.get("source_id"),
                    "group_id": evt.get("group_id"),
                    "payload_json": json.dumps(evt.get("payload", {}) or {}, ensure_ascii=True),
                }
                writer.writerow(flat)
