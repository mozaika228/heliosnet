from __future__ import annotations

import json
from pathlib import Path
import time

import cv2

from core.service import BaseService


class ContinuousEvalService(BaseService):
    def __init__(self, config, metrics, inference_engine, audit=None):
        super().__init__("continuous_eval")
        self.config = config
        self.metrics = metrics
        self.inference_engine = inference_engine
        self.audit = audit
        mlops_cfg = getattr(config, "mlops", {}) or {}
        eval_cfg = mlops_cfg.get("evaluation", {}) or {}
        self._enabled = bool(eval_cfg.get("enabled", False))
        self._benchmark_path = Path(eval_cfg.get("benchmark_path", "./data/edge_benchmark.jsonl"))
        self._interval_sec = float(eval_cfg.get("interval_sec", 600))
        self._max_cases = int(eval_cfg.get("max_cases", 20))
        self._last = 0.0

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last < self._interval_sec:
            return
        self._last = now
        rows = self._load_benchmark()
        if not rows:
            return
        tested = 0
        matches = 0
        cls_hits = 0
        for row in rows[: self._max_cases]:
            path = Path(str(row.get("image", "")))
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            dets = self.inference_engine._runner.infer(frame)  # noqa: SLF001 - intentional for shared runner
            tested += 1
            exp_count = int(row.get("expected_count", -1))
            if exp_count >= 0 and len(dets) == exp_count:
                matches += 1
            exp_cls = row.get("expected_class")
            if exp_cls is not None:
                exp_cls = int(exp_cls)
                if any(int(d.get("cls", -1)) == exp_cls for d in dets):
                    cls_hits += 1
        if tested == 0:
            return
        count_acc = matches / tested
        class_hit_rate = cls_hits / tested
        self.metrics.set_eval_score("edge_benchmark", "count_acc", count_acc)
        self.metrics.set_eval_score("edge_benchmark", "class_hit_rate", class_hit_rate)
        if self.audit is not None:
            self.audit.write(
                "continuous_eval",
                {
                    "tested": tested,
                    "count_acc": round(count_acc, 4),
                    "class_hit_rate": round(class_hit_rate, 4),
                },
            )

    def _load_benchmark(self) -> list[dict]:
        if not self._benchmark_path.exists():
            return []
        out = []
        with self._benchmark_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
