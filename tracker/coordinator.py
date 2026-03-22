from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import cv2

from core.service import BaseService

try:
    import supervision as sv
except Exception:  # pragma: no cover - optional dependency
    sv = None


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


class ByteTrackWrapper:
    def __init__(self, cfg: dict) -> None:
        if sv is None:
            raise RuntimeError("supervision not available")
        params = {}
        for key in (
            "track_activation_threshold",
            "lost_track_buffer",
            "minimum_matching_threshold",
            "frame_rate",
            "minimum_consecutive_frames",
        ):
            if key in cfg:
                params[key] = cfg[key]
        self._tracker = sv.ByteTrack(**params)

    def update(self, dets: list[dict]) -> list[dict]:
        if not dets:
            return []
        xyxy = np.array([d["bbox"] for d in dets], dtype=np.float32)
        conf = np.array([d.get("conf", 0.0) for d in dets], dtype=np.float32)
        cls = np.array([int(d.get("cls", -1)) for d in dets], dtype=int)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls,
        )
        tracked = self._tracker.update_with_detections(detections)
        if tracked is None or tracked.tracker_id is None:
            return []
        out = []
        for i in range(len(tracked.xyxy)):
            out.append(
                {
                    "track_id": int(tracked.tracker_id[i]),
                    "bbox": tracked.xyxy[i].tolist(),
                    "cls": int(tracked.class_id[i]) if tracked.class_id is not None else -1,
                    "conf": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                }
            )
        return out


class TrackCoordinator(BaseService):
    def __init__(self, config, metrics):
        super().__init__("tracker")
        self.config = config
        self.metrics = metrics
        track_cfg = getattr(config, "tracker", {})
        backend = str(track_cfg.get("backend", "iou")).lower()
        self._tracker = None
        self._preview = bool(track_cfg.get("preview", False))
        self._preview_window = str(track_cfg.get("preview_window", "HeliosNet"))
        if backend == "bytetrack":
            try:
                self._tracker = ByteTrackWrapper(track_cfg)
                print("[tracker] ByteTrack enabled", flush=True)
            except Exception as e:
                print(f"[tracker] ByteTrack unavailable, fallback to IoU: {e}", flush=True)
        if self._tracker is None:
            self._tracker = IoUTracker(
                float(track_cfg.get("iou_thr", 0.3)),
                int(track_cfg.get("max_age", 15)),
            )

    def handle(self, item) -> None:
        ts = float(item.get("ts", time.time()))
        dets = item.get("detections", [])
        label_map = {}
        for d in dets:
            if "label" in d and d["label"]:
                label_map[int(d.get("cls", -1))] = d["label"]
        if isinstance(self._tracker, IoUTracker):
            tracks = self._tracker.update(dets, ts)
        else:
            tracks = self._tracker.update(dets)
        for t in tracks:
            cls = int(t.get("cls", -1))
            if cls in label_map:
                t["label"] = label_map[cls]
        item["tracks"] = tracks
        if self._preview and isinstance(item.get("frame"), np.ndarray):
            self._show_preview(item.get("frame"), tracks, item.get("source_id", "source"))
        self.metrics.inc("frames_tracked")
        self.push(item)

    def _show_preview(self, frame: np.ndarray, tracks: list[dict], source_id: str) -> None:
        vis = frame.copy()
        window = f"{self._preview_window}::{source_id}"
        for t in tracks:
            bbox = t.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            tid = t.get("track_id", -1)
            name = t.get("label") or str(t.get("cls", -1))
            label = f"{name}#{tid}"
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit
