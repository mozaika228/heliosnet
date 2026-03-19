from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from heliosnet.runtime.scheduler import BaseService

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


def _as_numpy(outputs: Any) -> list[np.ndarray]:
    if isinstance(outputs, (list, tuple)):
        return [np.asarray(x) for x in outputs]
    return [np.asarray(outputs)]


def _parse_nms_like(
    outputs: list[np.ndarray], conf_threshold: float, max_det: int
) -> list[dict]:
    # Expect shape like (1, N, 6) or (N, 6) where cols are xyxy, conf, cls
    if not outputs:
        return []
    arr = outputs[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 6:
        return []
    dets = []
    for row in arr:
        x1, y1, x2, y2, conf, cls = row[:6]
        if conf < conf_threshold:
            continue
        dets.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "cls": int(cls),
            }
        )
        if len(dets) >= max_det:
            break
    return dets


class BaseRunner:
    def infer(self, frame: Any) -> list[dict]:
        raise NotImplementedError


class StubRunner(BaseRunner):
    def infer(self, frame: Any) -> list[dict]:
        return []


class OnnxRuntimeRunner(BaseRunner):
    def __init__(
        self, model_path: str, device: str, conf_threshold: float, max_det: int
    ) -> None:
        self._session = None
        self._input_name = None
        self._ready = False
        self._conf = conf_threshold
        self._max_det = max_det

        if ort is None:
            return
        path = Path(model_path)
        if not path.exists():
            return

        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._ready = True

    def infer(self, frame: Any) -> list[dict]:
        if not self._ready or frame is None:
            return []
        # Minimal, generic preprocessing: convert to float32 NCHW.
        if not isinstance(frame, np.ndarray):
            return []
        img = frame.astype(np.float32)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
        outputs = self._session.run(None, {self._input_name: img})
        return _parse_nms_like(_as_numpy(outputs), self._conf, self._max_det)


class DetectService(BaseService):
    def __init__(self, config, metrics):
        super().__init__("detect")
        self.config = config
        self.metrics = metrics
        detect_cfg = getattr(config, "detect", {})
        backend = str(detect_cfg.get("backend", "stub")).lower()
        model_path = str(detect_cfg.get("model", ""))
        device = str(detect_cfg.get("device", "cpu"))
        conf_threshold = float(detect_cfg.get("conf_threshold", 0.25))
        max_det = int(detect_cfg.get("max_det", 300))
        if backend == "onnxruntime":
            self._runner = OnnxRuntimeRunner(model_path, device, conf_threshold, max_det)
        else:
            self._runner = StubRunner()

    def handle(self, item) -> None:
        frame = item.get("frame")
        item["detections"] = self._runner.infer(frame)
        self.metrics.inc("frames_detected")
        self.push(item)
