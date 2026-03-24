from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from core.service import BaseService


class SyncEngine(BaseService):
    def __init__(self, config, metrics, energy=None):
        super().__init__("sync")
        self.config = config
        self.metrics = metrics
        self.energy = energy
        sync_cfg = getattr(config, "sync", {})
        self._mode = str(sync_cfg.get("mode", "offline")).lower()
        self._interval = int(sync_cfg.get("interval_sec", 60))
        self._last = 0.0
        self._batch_size = int(sync_cfg.get("batch_size", 100))
        self._queue_path = Path(sync_cfg.get("queue_path", "./data/sync_queue.jsonl"))
        self._sent_path = Path(sync_cfg.get("sent_path", "./data/sent_events.jsonl"))
        self._acked_path = Path(sync_cfg.get("acked_path", "./data/sync_acked.txt"))
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._sent_path.parent.mkdir(parents=True, exist_ok=True)
        self._acked_path.parent.mkdir(parents=True, exist_ok=True)
        self._acked = self._load_acked()

    def handle(self, item) -> None:
        events = item.get("events", [])
        if events:
            self._enqueue(events)
        self.push(item)

    def tick(self) -> None:
        now = time.time()
        if self.energy is not None:
            profile = self.energy.current_profile()
            if profile is not None:
                self._interval = int(profile.sync_interval_sec)
        if now - self._last < self._interval:
            return
        self._last = now
        self._flush_once()

    def _enqueue(self, events: list[dict]) -> None:
        with self._queue_path.open("a", encoding="utf-8") as f:
            for evt in events:
                event_id = self._event_id(evt)
                if event_id in self._acked:
                    continue
                row = {"event_id": event_id, "event": evt, "ts": time.time()}
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _flush_once(self) -> None:
        if self._mode == "offline":
            return
        rows = self._read_queue(self._batch_size)
        if not rows:
            return
        sent_count = 0
        with self._sent_path.open("a", encoding="utf-8") as sent_file:
            for row in rows:
                event_id = row.get("event_id")
                if not event_id or event_id in self._acked:
                    continue
                sent_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                self._acked.add(event_id)
                sent_count += 1
        self._persist_acked()
        self._rewrite_queue(exclude_ids=self._acked)
        if sent_count > 0:
            self.metrics.inc("events_written", sent_count)

    def _read_queue(self, limit: int) -> list[dict]:
        if not self._queue_path.exists():
            return []
        out = []
        with self._queue_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
        return out

    def _rewrite_queue(self, exclude_ids: set[str]) -> None:
        if not self._queue_path.exists():
            return
        temp = self._queue_path.with_suffix(".tmp")
        with self._queue_path.open("r", encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event_id = row.get("event_id")
                if event_id in exclude_ids:
                    continue
                dst.write(json.dumps(row, ensure_ascii=True) + "\n")
        temp.replace(self._queue_path)

    def _event_id(self, evt: dict) -> str:
        base = json.dumps(evt, ensure_ascii=True, sort_keys=True)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _load_acked(self) -> set[str]:
        if not self._acked_path.exists():
            return set()
        out = set()
        for line in self._acked_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.add(line)
        return out

    def _persist_acked(self) -> None:
        self._acked_path.write_text("\n".join(sorted(self._acked)) + "\n", encoding="utf-8")
