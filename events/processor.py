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
            return [Event(self.name, time.time(), {"count": count})]
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
                events.append(Event(self.name, time.time(), {"cx": cx, "cy": cy}))
        return events


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

    def handle(self, item) -> None:
        events = []
        for rule in self._rules:
            events.extend(rule.apply(item))
        item["events"] = [
            {"name": e.name, "ts": e.ts, "payload": e.payload} for e in events
        ]
        self.metrics.inc("events_built")
        self.push(item)
