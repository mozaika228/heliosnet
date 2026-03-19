from __future__ import annotations

import time

from heliosnet.runtime.scheduler import BaseService
from heliosnet.pipeline.sources import build_sources


class IngestService(BaseService):
    def __init__(self, config, metrics):
        super().__init__("ingest")
        self.config = config
        self.metrics = metrics
        ingest_cfg = getattr(config, "ingest", {})
        sources = ingest_cfg.get("sources", [])
        max_fps = int(ingest_cfg.get("max_fps", 0) or 0)
        self._idle_sleep_ms = int(ingest_cfg.get("idle_sleep_ms", 5))
        loop_files = bool(ingest_cfg.get("loop_files", True))
        self._sources = build_sources(sources, max_fps=max_fps, loop=loop_files)
        self._rr = 0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._sources:
            time.sleep(self._idle_sleep_ms / 1000.0)
            return
        tried = 0
        while tried < len(self._sources):
            src = self._sources[self._rr]
            self._rr = (self._rr + 1) % len(self._sources)
            tried += 1
            item = src.read()
            if item is None:
                continue
            self.metrics.inc("frames_in")
            self.push(item)
            return
        time.sleep(self._idle_sleep_ms / 1000.0)
