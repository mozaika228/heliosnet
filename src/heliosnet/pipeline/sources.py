from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import cv2


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
        return {
            "ts": time.time(),
            "source_id": self.cfg.source_id,
            "frame": frame,
        }


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


def build_sources(entries: Iterable[str], max_fps: int, loop: bool) -> list[BaseSource]:
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

        # default to file path (including Windows drive paths)
        sources.append(VideoSource(cfg))

    return sources
