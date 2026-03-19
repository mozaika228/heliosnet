from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import cv2

from core.service import BaseService

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    from groundingdino.util.inference import load_model as gd_load_model
    from groundingdino.util.inference import predict as gd_predict
except Exception:  # pragma: no cover - optional dependency
    gd_load_model = None
    gd_predict = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


def _as_numpy(outputs: Any) -> list[np.ndarray]:
    if isinstance(outputs, (list, tuple)):
        return [np.asarray(x) for x in outputs]
    return [np.asarray(outputs)]


def _parse_nms_like(outputs: list[np.ndarray], conf_threshold: float, max_det: int) -> list[dict]:
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


def _parse_yolo_v11(
    outputs: list[np.ndarray],
    frame: Any,
    conf_threshold: float,
    max_det: int,
    input_hw: tuple[int, int] | None,
    scale_xy: tuple[float, float] | None,
) -> list[dict]:
    if not outputs:
        return []
    arr = outputs[0]
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] < 5:
        return []
    arr = arr[0]
    preds = np.transpose(arr, (1, 0))
    if preds.shape[1] < 5:
        return []

    boxes = preds[:, 0:4]
    scores = preds[:, 4:]
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]

    keep = confs >= conf_threshold
    if not np.any(keep):
        return []
    boxes = boxes[keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

    order = np.argsort(-confs)
    if max_det > 0:
        order = order[:max_det]
    boxes = boxes[order]
    confs = confs[order]
    cls_ids = cls_ids[order]

    if isinstance(frame, np.ndarray) and frame.size > 0:
        h, w = frame.shape[:2]
        if np.max(boxes) <= 1.5:
            boxes = boxes.copy()
            boxes[:, 0] *= w
            boxes[:, 1] *= h
            boxes[:, 2] *= w
            boxes[:, 3] *= h
        elif input_hw is not None and scale_xy is not None:
            sx, sy = scale_xy
            boxes = boxes.copy()
            boxes[:, 0] *= sx
            boxes[:, 1] *= sy
            boxes[:, 2] *= sx
            boxes[:, 3] *= sy

    dets = []
    for (cx, cy, bw, bh), conf, cls in zip(boxes, confs, cls_ids):
        x1 = float(cx - bw / 2.0)
        y1 = float(cy - bh / 2.0)
        x2 = float(cx + bw / 2.0)
        y2 = float(cy + bh / 2.0)
        dets.append({"bbox": [x1, y1, x2, y2], "conf": float(conf), "cls": int(cls)})
    return dets


def _iou_xyxy(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area2 = np.maximum(0, others[:, 2] - others[:, 0]) * np.maximum(
        0, others[:, 3] - others[:, 1]
    )
    union = area1 + area2 - inter + 1e-6
    return inter / union


def _nms(dets: list[dict], iou_thr: float) -> list[dict]:
    if not dets or iou_thr <= 0:
        return dets
    boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
    scores = np.array([d["conf"] for d in dets], dtype=np.float32)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        iou = _iou_xyxy(boxes[i], boxes[rest])
        idxs = rest[iou < iou_thr]
    return [dets[i] for i in keep]


class BaseRunner:
    def infer(self, frame: Any) -> list[dict]:
        raise NotImplementedError


class StubRunner(BaseRunner):
    def infer(self, frame: Any) -> list[dict]:
        return []


class OnnxRuntimeRunner(BaseRunner):
    def __init__(
        self,
        model_path: str,
        device: str,
        conf_threshold: float,
        max_det: int,
        output_format: str,
    ) -> None:
        self._session = None
        self._input_name = None
        self._ready = False
        self._conf = conf_threshold
        self._max_det = max_det
        self._output_format = output_format
        self._input_hw = None

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
        ishape = self._session.get_inputs()[0].shape
        if len(ishape) >= 4:
            h = ishape[2] if ishape[2] is not None else None
            w = ishape[3] if ishape[3] is not None else None
            if isinstance(h, int) and isinstance(w, int):
                self._input_hw = (h, w)
        self._ready = True

    def infer(self, frame: Any) -> list[dict]:
        if not self._ready or frame is None:
            return []
        if not isinstance(frame, np.ndarray):
            return []
        img = frame
        orig_h, orig_w = img.shape[:2]
        scale_xy = None
        if self._input_hw is not None:
            h, w = self._input_hw
            if img.shape[0] != h or img.shape[1] != w:
                scale_xy = (orig_w / w, orig_h / h)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
        outputs = self._session.run(None, {self._input_name: img})
        outs = _as_numpy(outputs)
        fmt = self._output_format
        if fmt in ("yolo11", "yolo_v11", "yolo"):
            return _parse_yolo_v11(
                outs, frame, self._conf, self._max_det, self._input_hw, scale_xy
            )
        if fmt == "nms":
            return _parse_nms_like(outs, self._conf, self._max_det)
        dets = _parse_yolo_v11(outs, frame, self._conf, self._max_det, self._input_hw, scale_xy)
        if dets:
            return dets
        return _parse_nms_like(outs, self._conf, self._max_det)


class GroundingDinoRunner(BaseRunner):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        device: str,
    ) -> None:
        self._ready = False
        self._text_prompt = text_prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device = device
        self._model = None
        self._classes = self._parse_classes(text_prompt)

        if gd_load_model is None or gd_predict is None:
            print("[groundingdino] FAILED import groundingdino.util.inference", flush=True)
            return
        if not config_path or not checkpoint_path:
            print(
                f"[groundingdino] FAILED missing config/checkpoint config={config_path} checkpoint={checkpoint_path}",
                flush=True,
            )
            return
        try:
            self._model = gd_load_model(config_path, checkpoint_path, device=device)
            self._ready = True
            print(
                f"[groundingdino] READY config={config_path} checkpoint={checkpoint_path} device={device}",
                flush=True,
            )
        except Exception as e:
            self._ready = False
            print(
                f"[groundingdino] FAILED config={config_path} checkpoint={checkpoint_path} device={device} err={e}",
                flush=True,
            )

    @staticmethod
    def _parse_classes(text_prompt: str) -> list[str]:
        parts = [p.strip() for p in text_prompt.split(".")]
        return [p for p in parts if p]

    def infer(self, frame: Any) -> list[dict]:
        if not self._ready or frame is None:
            return []
        if not isinstance(frame, np.ndarray):
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, logits, phrases = gd_predict(
            model=self._model,
            image=rgb,
            caption=self._text_prompt,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            device=self._device,
        )

        if boxes is None:
            return []
        boxes = np.asarray(boxes)
        logits = np.asarray(logits)
        h, w = frame.shape[:2]
        dets = []
        for i in range(len(boxes)):
            cx, cy, bw, bh = boxes[i].tolist()
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            phrase = phrases[i] if phrases is not None and i < len(phrases) else ""
            cls_idx = -1
            if phrase and self._classes:
                for j, c in enumerate(self._classes):
                    if c.lower() in phrase.lower():
                        cls_idx = j
                        break
            conf = float(logits[i]) if logits is not None and i < len(logits) else 0.0
            dets.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "cls": cls_idx,
                    "label": phrase,
                }
            )
        return dets


class UltralyticsRunner(BaseRunner):
    def __init__(self, model_path: str, device: str, conf: float, classes: list[int]):
        self._ready = False
        self._model = None
        self._device = device
        self._conf = conf
        self._classes = classes

        if YOLO is None:
            print("[ultralytics] FAILED import ultralytics", flush=True)
            return
        try:
            self._model = YOLO(model_path)
            self._ready = True
            self._names = getattr(self._model, "names", {}) or {}
            print(
                f"[ultralytics] READY model={model_path} device={device} conf={conf} classes={classes}",
                flush=True,
            )
        except Exception as e:
            print(f"[ultralytics] FAILED model={model_path} err={e}", flush=True)
            self._ready = False
            self._names = {}

    def infer(self, frame: Any) -> list[dict]:
        if not self._ready or frame is None:
            return []
        results = self._model(
            frame,
            device=self._device,
            conf=self._conf,
            classes=self._classes if self._classes else None,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
        clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
        dets = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            conf = float(confs[i]) if confs is not None else 0.0
            cls = int(clss[i]) if clss is not None else -1
            name = self._names.get(cls, str(cls)) if isinstance(self._names, dict) else str(cls)
            dets.append(
                {"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls, "label": name}
            )
        return dets


class InferenceEngine(BaseService):
    def __init__(self, config, metrics):
        super().__init__("inference")
        self.config = config
        self.metrics = metrics
        inf_cfg = getattr(config, "inference", {})
        backend = str(inf_cfg.get("backend", "stub")).lower()
        model_path = str(inf_cfg.get("model", ""))
        device = str(inf_cfg.get("device", "cpu"))
        conf_threshold = float(inf_cfg.get("conf", inf_cfg.get("conf_threshold", 0.25)))
        max_det = int(inf_cfg.get("max_det", 300))
        output_format = str(inf_cfg.get("output_format", "auto")).lower()
        self._preview = bool(inf_cfg.get("preview", False))
        self._preview_window = str(inf_cfg.get("preview_window", "HeliosNet"))
        self._nms_iou = float(inf_cfg.get("nms_iou", 0.0))
        self._classes = [int(c) for c in inf_cfg.get("classes", [])]

        if backend == "onnxruntime":
            self._runner = OnnxRuntimeRunner(
                model_path, device, conf_threshold, max_det, output_format
            )
        elif backend == "groundingdino":
            self._runner = GroundingDinoRunner(
                str(inf_cfg.get("gd_config", "")),
                str(inf_cfg.get("gd_checkpoint", "")),
                str(inf_cfg.get("text_prompt", "")),
                float(inf_cfg.get("box_threshold", 0.35)),
                float(inf_cfg.get("text_threshold", 0.25)),
                str(inf_cfg.get("device", "cpu")),
            )
        elif backend == "ultralytics":
            self._runner = UltralyticsRunner(
                model_path, device, conf_threshold, self._classes
            )
        else:
            self._runner = StubRunner()

    def handle(self, item) -> None:
        frame = item.get("frame")
        dets = self._runner.infer(frame)
        if self._classes and dets:
            dets = [d for d in dets if int(d.get("cls", -1)) in self._classes]
        dets = _nms(dets, self._nms_iou)
        item["detections"] = dets
        if self._preview and isinstance(frame, np.ndarray):
            self._show_preview(frame, dets)
        self.metrics.inc("frames_detected")
        self.push(item)

    def _show_preview(self, frame: np.ndarray, detections: list[dict]) -> None:
        vis = frame.copy()
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls = det.get("cls", -1)
            conf = det.get("conf", 0.0)
            name = det.get("label") or str(cls)
            label = f"{name}:{conf:.2f}"
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self._preview_window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit
