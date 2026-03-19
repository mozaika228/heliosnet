from __future__ import annotations

from heliosnet.runtime.scheduler import BaseService


class TrackService(BaseService):
    def __init__(self, config, metrics):
        super().__init__("track")
        self.config = config
        self.metrics = metrics

    def handle(self, item) -> None:
        # Placeholder for tracker output
        item["tracks"] = []
        self.metrics.inc("frames_tracked")
        self.push(item)
