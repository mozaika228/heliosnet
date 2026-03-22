from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from core.service import BaseService


@dataclass
class SourceConfig:
    uri: str
    source_id: str
    max_fps: int
    loop: bool


class BaseSource:
    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._last_emit = 0.0

    def _rate_ok(self) -> bool:
        if self.cfg.max_fps <= 0:
            return True
        now = time.time()
        if now - self._last_emit < 1.0 / self.cfg.max_fps:
            return False
        self._last_emit = now
        return True

    def read(self) -> dict | None:
        raise NotImplementedError


class VideoSource(BaseSource):
    def __init__(self, cfg: SourceConfig, device_index: int | None = None) -> None:
        super().__init__(cfg)
        self._device_index = device_index
        self._cap = None

    def _open(self) -> None:
        if self._device_index is None:
            self._cap = cv2.VideoCapture(self.cfg.uri)
        else:
            self._cap = cv2.VideoCapture(self._device_index)

    def read(self) -> dict | None:
        if not self._rate_ok():
            return None
        if self._cap is None or not self._cap.isOpened():
            self._open()
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok:
            if self.cfg.loop and self._device_index is None:
                self._cap.release()
                self._cap = None
                self._open()
                ok, frame = self._cap.read()
            if not ok:
                return None
        return {"ts": time.time(), "source_id": self.cfg.source_id, "frame": frame}


class ImageSource(BaseSource):
    def __init__(self, cfg: SourceConfig, images: list[Path]) -> None:
        super().__init__(cfg)
        self._images = images
        self._idx = 0

    def read(self) -> dict | None:
        if not self._rate_ok():
            return None
        if not self._images:
            return None
        if self._idx >= len(self._images):
            if self.cfg.loop:
                self._idx = 0
            else:
                return None
        path = self._images[self._idx]
        self._idx += 1
        frame = cv2.imread(str(path))
        if frame is None:
            return None
        return {
            "ts": time.time(),
            "source_id": self.cfg.source_id,
            "frame": frame,
            "meta": {"path": str(path)},
        }


def build_sources(entries: list[str], max_fps: int, loop: bool) -> list[BaseSource]:
    sources: list[BaseSource] = []
    for idx, entry in enumerate(entries):
        entry = entry.strip()
        source_id = f"src-{idx:02d}"
        cfg = SourceConfig(uri=entry, source_id=source_id, max_fps=max_fps, loop=loop)

        if entry.startswith("webcam:"):
            device = int(entry.split(":", 1)[1])
            sources.append(VideoSource(cfg, device_index=device))
            continue
        if entry.startswith("image:"):
            path = Path(entry.split(":", 1)[1])
            sources.append(ImageSource(cfg, [path]))
            continue
        if entry.startswith("dir:"):
            path = Path(entry.split(":", 1)[1])
            images = sorted([p for p in path.glob("*") if p.is_file()])
            sources.append(ImageSource(cfg, images))
            continue
        if entry.startswith("file:"):
            uri = entry.split(":", 1)[1]
            cfg.uri = uri
            sources.append(VideoSource(cfg))
            continue
        if "://" in entry:
            sources.append(VideoSource(cfg))
            continue

        sources.append(VideoSource(cfg))

    return sources


class IngestManager(BaseService):
    def __init__(self, config, metrics):
        super().__init__("ingest")
        self.config = config
        self.metrics = metrics
        ingest_cfg = getattr(config, "ingest", {})
        sources = ingest_cfg.get("sources", [])
        max_fps = int(ingest_cfg.get("max_fps", 0) or 0)
        self._idle_sleep_ms = int(ingest_cfg.get("idle_sleep_ms", 5))
        loop_files = bool(ingest_cfg.get("loop_files", True))
        self._sync = bool(ingest_cfg.get("sync", False))
        self._sync_timeout_ms = int(ingest_cfg.get("sync_timeout_ms", 200))
        self._sources = build_sources(sources, max_fps=max_fps, loop=loop_files)
        self._rr = 0
        self._pending = {}
        self._group_id = 0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._sources:
            time.sleep(self._idle_sleep_ms / 1000.0)
            return
        if self._sync:
            self._tick_sync()
            return
        tried = 0
        while tried < len(self._sources):
            src = self._sources[self._rr]
            self._rr = (self._rr + 1) % len(self._sources)
            tried += 1
            item = src.read()
            if item is None:
                continue
            self.metrics.inc_frame(item.get("source_id", "source"))
            self.push(item)
            return
        time.sleep(self._idle_sleep_ms / 1000.0)

    def _tick_sync(self) -> None:
        # Try to read one frame per source and emit a synchronized group
        for src in self._sources:
            item = src.read()
            if item is not None:
                self._pending[src.cfg.source_id] = item

        if len(self._pending) < len(self._sources):
            time.sleep(self._idle_sleep_ms / 1000.0)
            return

        items = list(self._pending.values())
        ts_values = [it.get("ts", 0) for it in items]
        if ts_values and (max(ts_values) - min(ts_values)) * 1000.0 > self._sync_timeout_ms:
            # drop oldest
            oldest_id = None
            oldest_ts = None
            for sid, it in self._pending.items():
                t = it.get("ts", 0)
                if oldest_ts is None or t < oldest_ts:
                    oldest_ts = t
                    oldest_id = sid
            if oldest_id is not None:
                self._pending.pop(oldest_id, None)
            time.sleep(self._idle_sleep_ms / 1000.0)
            return

        self._group_id += 1
        for it in items:
            it["group_id"] = self._group_id
            self.metrics.inc_frame(it.get("source_id", "source"))
            self.push(it)
        self._pending = {}
