from __future__ import annotations

import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class Metrics:
    def __init__(self) -> None:
        self._started = False
        self._start_ts = time.time()

        self.frames_total = Counter(
            "heliosnet_frames_total",
            "Frames processed per stream",
            ["source_id"],
        )
        self.detections_total = Counter(
            "heliosnet_detections_total",
            "Detections per class/stream",
            ["source_id", "class_id", "label"],
        )
        self.inference_latency = Histogram(
            "heliosnet_inference_latency_seconds",
            "Inference latency in seconds",
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
        )
        self.confidence_mean = Gauge(
            "heliosnet_confidence_mean",
            "Mean confidence per stream (EMA)",
            ["source_id"],
        )
        self.objects_per_frame = Gauge(
            "heliosnet_objects_per_frame",
            "Objects per frame (instant)",
            ["source_id"],
        )
        self.tracks_total = Counter(
            "heliosnet_tracks_total",
            "Tracks updated",
        )
        self.events_built_total = Counter(
            "heliosnet_events_built_total",
            "Events built",
        )
        self.events_written_total = Counter(
            "heliosnet_events_written_total",
            "Events written to store",
        )

    def start_http(self, port: int) -> None:
        if self._started:
            return
        start_http_server(port)
        self._started = True

    def inc_frame(self, source_id: str) -> None:
        self.frames_total.labels(source_id=source_id).inc()

    def observe_inference(self, seconds: float) -> None:
        self.inference_latency.observe(seconds)

    def add_detections(self, source_id: str, dets: list[dict]) -> None:
        if not dets:
            self.objects_per_frame.labels(source_id=source_id).set(0)
            return
        self.objects_per_frame.labels(source_id=source_id).set(len(dets))
        sum_conf = 0.0
        for d in dets:
            cls = str(d.get("cls", -1))
            label = str(d.get("label", cls))
            conf = float(d.get("conf", 0.0))
            sum_conf += conf
            self.detections_total.labels(
                source_id=source_id, class_id=cls, label=label
            ).inc()
        mean_conf = sum_conf / max(1, len(dets))
        self._update_confidence_ema(source_id, mean_conf)

    def _update_confidence_ema(self, source_id: str, value: float) -> None:
        # EMA smoothing in gauge: new = 0.9 * old + 0.1 * value
        label = self.confidence_mean.labels(source_id=source_id)
        try:
            old = label._value.get()  # pylint: disable=protected-access
        except Exception:
            old = value
        ema = old * 0.9 + value * 0.1
        label.set(ema)

    def inc(self, name: str, value: int = 1) -> None:
        if name == "frames_tracked":
            self.tracks_total.inc(value)
        elif name == "events_built":
            self.events_built_total.inc(value)
        elif name == "events_written":
            self.events_written_total.inc(value)
