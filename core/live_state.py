from __future__ import annotations

from collections import deque
from threading import Lock
import time


class LiveState:
    def __init__(self, max_events: int = 500):
        self._lock = Lock()
        self._sources = {}
        self._events = deque(maxlen=max_events)
        self._seq = 0
        self._stats = {
            "frames": 0,
            "objects": 0,
            "last_event_ts": 0.0,
        }

    def update_frame(self, source_id: str, tracks: list[dict], detections: list[dict]) -> None:
        with self._lock:
            self._sources[source_id] = {
                "ts": time.time(),
                "track_count": len(tracks),
                "object_count": len(detections),
                "tracks": tracks[:100],
                "detections": detections[:100],
            }
            self._stats["frames"] += 1
            self._stats["objects"] += len(detections)

    def push_events(self, events: list[dict]) -> None:
        if not events:
            return
        with self._lock:
            for evt in events:
                self._seq += 1
                row = dict(evt)
                row["seq"] = self._seq
                self._events.append(row)
                self._stats["last_event_ts"] = float(row.get("ts", time.time()))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sources": dict(self._sources),
                "stats": dict(self._stats),
                "events_tail": list(self._events)[-50:],
                "seq": self._seq,
            }

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return [e for e in self._events if int(e.get("seq", 0)) > seq]
