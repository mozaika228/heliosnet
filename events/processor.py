from __future__ import annotations

from dataclasses import dataclass
import time

from core.service import BaseService


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _in_rect(cx: float, cy: float, rect: list[float]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _in_poly(cx: float, cy: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > cy) != (yj > cy)) and (
            cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _iter_objects(item: dict, prefer_tracks: bool):
    if prefer_tracks and item.get("tracks"):
        return item["tracks"]
    return item.get("detections", [])


def _pose_state_from_keypoints(kps: list[list[float]], bbox: list[float]) -> tuple[str, float] | None:
    # COCO keypoints indices used by YOLO pose:
    # 0 nose, 5/6 shoulders, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
    if not kps or len(kps) < 17 or not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    def _ok(i: int) -> bool:
        return i < len(kps) and len(kps[i]) >= 3 and float(kps[i][2]) >= 0.35

    if not (_ok(5) and _ok(6) and _ok(11) and _ok(12)):
        return None

    sh_y = (float(kps[5][1]) + float(kps[6][1])) / 2.0
    hip_y = (float(kps[11][1]) + float(kps[12][1])) / 2.0
    torso = max(1.0, hip_y - sh_y)
    aspect = w / h

    wrists_up = False
    if _ok(9) and _ok(10):
        wrists_up = float(kps[9][1]) < sh_y and float(kps[10][1]) < sh_y

    knees_ok = _ok(13) and _ok(14)
    ankles_ok = _ok(15) and _ok(16)
    standing = False
    sitting = False
    if knees_ok and ankles_ok:
        knee_y = (float(kps[13][1]) + float(kps[14][1])) / 2.0
        ankle_y = (float(kps[15][1]) + float(kps[16][1])) / 2.0
        standing = sh_y < hip_y < knee_y < ankle_y and (ankle_y - hip_y) > 0.8 * torso
        sitting = sh_y < hip_y < knee_y and (knee_y - hip_y) < 0.6 * torso

    # simple fall heuristic: horizontal body + compressed torso + wide bbox
    fall = aspect > 1.2 and torso < 0.45 * h

    if fall:
        return "falling", min(1.0, aspect / 2.0)
    if wrists_up:
        return "hands_up", 0.75
    if standing:
        return "standing", 0.8
    if sitting:
        return "sitting", 0.7
    return "unknown", 0.5


@dataclass
class Event:
    name: str
    ts: float
    payload: dict


class CountThresholdRule:
    def __init__(self, name: str, threshold: int, prefer_tracks: bool, classes: list[int]):
        self.name = name
        self.threshold = threshold
        self.prefer_tracks = prefer_tracks
        self.classes = classes

    def apply(self, item: dict) -> list[Event]:
        objs = _iter_objects(item, self.prefer_tracks)
        if self.classes:
            objs = [o for o in objs if int(o.get("cls", -1)) in self.classes]
        count = len(objs)
        if count >= self.threshold:
            track_ids = [o.get("track_id") for o in objs if "track_id" in o]
            class_ids = [int(o.get("cls", -1)) for o in objs if "cls" in o]
            labels = [o.get("label") for o in objs if o.get("label")]
            payload = {"count": count}
            if track_ids:
                payload["track_ids"] = track_ids
            if class_ids:
                payload["class_ids"] = class_ids
            if labels:
                payload["labels"] = labels
            return [Event(self.name, time.time(), payload)]
        return []


class ZoneEntryRule:
    def __init__(
        self,
        name: str,
        rect: list[float] | None,
        poly: list[list[float]] | None,
        prefer_tracks: bool,
        classes: list[int],
    ):
        self.name = name
        self.rect = rect
        self.poly = poly
        self.prefer_tracks = prefer_tracks
        self.classes = classes

    def apply(self, item: dict) -> list[Event]:
        events: list[Event] = []
        objs = _iter_objects(item, self.prefer_tracks)
        for o in objs:
            if self.classes and int(o.get("cls", -1)) not in self.classes:
                continue
            bbox = o.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cx, cy = _center(bbox)
            hit = False
            if self.rect and len(self.rect) == 4:
                hit = _in_rect(cx, cy, self.rect)
            elif self.poly:
                hit = _in_poly(cx, cy, self.poly)
            if hit:
                payload = {"cx": cx, "cy": cy}
                if "track_id" in o:
                    payload["track_id"] = o.get("track_id")
                if "label" in o:
                    payload["label"] = o.get("label")
                if "cls" in o:
                    payload["class_id"] = o.get("cls")
                events.append(Event(self.name, time.time(), payload))
        return events


class PoseStateRule:
    def __init__(self, name: str, classes: list[int], min_score: float):
        self.name = name
        self.classes = classes
        self.min_score = min_score

    def apply(self, item: dict) -> list[Event]:
        out: list[Event] = []
        dets = item.get("detections", []) or []
        for d in dets:
            cls = int(d.get("cls", -1))
            if self.classes and cls not in self.classes:
                continue
            label = str(d.get("label", "")).lower()
            if not self.classes and label and label != "person" and cls != 0:
                continue
            pose = _pose_state_from_keypoints(d.get("keypoints", []), d.get("bbox", []))
            if pose is None:
                continue
            state, score = pose
            if score < self.min_score:
                continue
            payload = {
                "pose_state": state,
                "score": round(float(score), 4),
                "class_id": cls,
                "label": d.get("label"),
            }
            out.append(Event(self.name, time.time(), payload))
            if state == "falling":
                out.append(
                    Event(
                        "FALL_ALERT",
                        time.time(),
                        {"pose_state": state, "score": round(float(score), 4)},
                    )
                )
        return out


class EventsProcessor(BaseService):
    def __init__(self, config, metrics):
        super().__init__("events")
        self.config = config
        self.metrics = metrics
        self._rules = []
        events_cfg = getattr(config, "events", {})
        for rule in events_cfg.get("rules", []):
            rtype = str(rule.get("type", "")).lower()
            name = str(rule.get("name", rtype or "rule"))
            prefer_tracks = bool(rule.get("prefer_tracks", False))
            classes = [int(c) for c in rule.get("classes", [])]
            if rtype == "count_threshold":
                threshold = int(rule.get("threshold", 1))
                self._rules.append(
                    CountThresholdRule(name, threshold, prefer_tracks, classes)
                )
            elif rtype == "zone_entry":
                rect = None
                poly = None
                if "rect" in rule:
                    rect = [float(x) for x in rule.get("rect", [0, 0, 0, 0])]
                if "polygon" in rule:
                    poly = [[float(a), float(b)] for a, b in rule.get("polygon", [])]
                if (rect and len(rect) == 4) or (poly and len(poly) >= 3):
                    self._rules.append(
                        ZoneEntryRule(name, rect, poly, prefer_tracks, classes)
                    )
            elif rtype == "pose_state":
                self._rules.append(
                    PoseStateRule(
                        name=name,
                        classes=classes,
                        min_score=float(rule.get("min_score", 0.55)),
                    )
                )

    def handle(self, item) -> None:
        events = []
        for rule in self._rules:
            events.extend(rule.apply(item))
        item["events"] = [
            {
                "name": e.name,
                "ts": e.ts,
                "payload": e.payload,
                "source_id": item.get("source_id"),
                "group_id": item.get("group_id"),
            }
            for e in events
        ]
        self.metrics.inc("events_built")
        self.push(item)
