from __future__ import annotations

from dataclasses import dataclass
import time

from core.service import BaseService


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    cls: int
    conf: float
    last_ts: float
    hits: int = 1
    age: int = 0


class IoUTracker:
    def __init__(self, iou_thr: float, max_age: int) -> None:
        self._iou_thr = iou_thr
        self._max_age = max_age
        self._next_id = 1
        self._tracks: list[Track] = []

    def update(self, dets: list[dict], ts: float) -> list[dict]:
        for tr in self._tracks:
            tr.age += 1

        assigned = set()
        for tr in self._tracks:
            best_iou = 0.0
            best_j = -1
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                if int(d.get("cls", -1)) != tr.cls:
                    continue
                iou = _iou(tr.bbox, d.get("bbox", tr.bbox))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self._iou_thr:
                d = dets[best_j]
                tr.bbox = d["bbox"]
                tr.conf = float(d.get("conf", tr.conf))
                tr.last_ts = ts
                tr.hits += 1
                tr.age = 0
                assigned.add(best_j)

        for j, d in enumerate(dets):
            if j in assigned:
                continue
            tr = Track(
                track_id=self._next_id,
                bbox=d["bbox"],
                cls=int(d.get("cls", -1)),
                conf=float(d.get("conf", 0.0)),
                last_ts=ts,
            )
            self._next_id += 1
            self._tracks.append(tr)

        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

        return [
            {
                "track_id": t.track_id,
                "bbox": t.bbox,
                "cls": t.cls,
                "conf": t.conf,
                "age": t.age,
                "hits": t.hits,
            }
            for t in self._tracks
        ]


class TrackCoordinator(BaseService):
    def __init__(self, config, metrics):
        super().__init__("tracker")
        self.config = config
        self.metrics = metrics
        track_cfg = getattr(config, "tracker", {})
        self._tracker = IoUTracker(
            float(track_cfg.get("iou_thr", 0.3)),
            int(track_cfg.get("max_age", 15)),
        )

    def handle(self, item) -> None:
        ts = float(item.get("ts", time.time()))
        dets = item.get("detections", [])
        item["tracks"] = self._tracker.update(dets, ts)
        self.metrics.inc("frames_tracked")
        self.push(item)
