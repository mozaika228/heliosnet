from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class Metrics:
    def __init__(self) -> None:
        self._started = False
        self._start_ts = time.time()
        self._degraded = 0

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
        self.e2e_latency = Histogram(
            "heliosnet_e2e_latency_seconds",
            "End-to-end latency (ingest to decision/store) in seconds",
            buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
        )
        self.uptime_ratio = Gauge(
            "heliosnet_uptime_ratio",
            "Node uptime ratio approximation (healthy tick ratio)",
        )
        self.miss_rate = Gauge(
            "heliosnet_miss_rate",
            "Rolling miss rate (frames with zero detections)",
            ["source_id"],
        )
        self.slo_breaches_total = Counter(
            "heliosnet_slo_breaches_total",
            "SLO breach events",
            ["slo_name"],
        )
        self.shadow_inference_total = Counter(
            "heliosnet_shadow_inference_total",
            "Shadow inferences executed",
            ["model_version"],
        )
        self.drift_score = Gauge(
            "heliosnet_drift_score",
            "Current drift score by source",
            ["source_id", "drift_type"],
        )
        self.evaluation_score = Gauge(
            "heliosnet_edge_eval_score",
            "Continuous evaluation score",
            ["benchmark", "metric"],
        )
        self.watchdog_actions_total = Counter(
            "heliosnet_watchdog_actions_total",
            "Watchdog actions taken",
            ["action"],
        )
        self.runtime_mode = Gauge(
            "heliosnet_runtime_mode",
            "Runtime degradation mode (0=normal,1=degraded,2=safe)",
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

    def observe_e2e(self, seconds: float) -> None:
        self.e2e_latency.observe(seconds)

    def add_detections(self, source_id: str, dets: list[dict]) -> None:
        if not dets:
            self.objects_per_frame.labels(source_id=source_id).set(0)
            self.miss_rate.labels(source_id=source_id).set(1.0)
            return
        self.objects_per_frame.labels(source_id=source_id).set(len(dets))
        self.miss_rate.labels(source_id=source_id).set(0.0)
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

    def set_uptime_ratio(self, value: float) -> None:
        self.uptime_ratio.set(max(0.0, min(1.0, float(value))))

    def inc_slo_breach(self, slo_name: str) -> None:
        self.slo_breaches_total.labels(slo_name=slo_name).inc()

    def inc_shadow(self, model_version: str) -> None:
        self.shadow_inference_total.labels(model_version=model_version).inc()

    def set_drift_score(self, source_id: str, drift_type: str, score: float) -> None:
        self.drift_score.labels(source_id=source_id, drift_type=drift_type).set(float(score))

    def set_eval_score(self, benchmark: str, metric: str, score: float) -> None:
        self.evaluation_score.labels(benchmark=benchmark, metric=metric).set(float(score))

    def watchdog_action(self, action: str) -> None:
        self.watchdog_actions_total.labels(action=action).inc()

    def set_runtime_mode(self, mode: str) -> None:
        m = str(mode).lower()
        if m == "normal":
            self.runtime_mode.set(0)
        elif m == "degraded":
            self.runtime_mode.set(1)
        else:
            self.runtime_mode.set(2)
