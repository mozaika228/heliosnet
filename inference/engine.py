from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import cv2
import time

from core.service import BaseService
from inference.phone_matcher import PhoneModelMatcher

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
        self._model_path = model_path

        if YOLO is None:
            print("[ultralytics] FAILED import ultralytics", flush=True)
            return
        model_to_load = self._resolve_fallback_model(model_path)
        try:
            self._model = YOLO(model_to_load)
            self._ready = True
            self._model_path = model_to_load
            self._names = getattr(self._model, "names", {}) or {}
            print(
                f"[ultralytics] READY model={model_to_load} device={device} conf={conf} classes={classes}",
                flush=True,
            )
        except Exception as e:
            print(f"[ultralytics] FAILED model={model_to_load} err={e}", flush=True)
            self._ready = False
            self._names = {}

    def _resolve_fallback_model(self, model_path: str) -> str:
        p = Path(model_path)
        if p.exists():
            return model_path
        name = p.name.lower()
        if "-pose" in name and name.endswith(".pt"):
            fallback_name = p.name.replace("-pose", "")
            fallback = p.with_name(fallback_name)
            if fallback.exists():
                print(
                    f"[ultralytics] pose model missing: {model_path}; fallback={fallback}",
                    flush=True,
                )
                return str(fallback)
        return model_path

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
        return self._results_to_dets(results[0])

    def _results_to_dets(self, r) -> list[dict]:
        if r.boxes is None or r.boxes.xyxy is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
        clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
        kpts_xy = None
        kpts_conf = None
        if getattr(r, "keypoints", None) is not None:
            if getattr(r.keypoints, "xy", None) is not None:
                kpts_xy = r.keypoints.xy.cpu().numpy()
            if getattr(r.keypoints, "conf", None) is not None:
                kpts_conf = r.keypoints.conf.cpu().numpy()
        dets = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            conf = float(confs[i]) if confs is not None else 0.0
            cls = int(clss[i]) if clss is not None else -1
            name = (
                self._names.get(cls, str(cls))
                if isinstance(self._names, dict)
                else str(cls)
            )
            det = {"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls, "label": name}
            if kpts_xy is not None and i < len(kpts_xy):
                pts = []
                for j, xy in enumerate(kpts_xy[i]):
                    xk = float(xy[0])
                    yk = float(xy[1])
                    ck = float(kpts_conf[i][j]) if kpts_conf is not None and i < len(kpts_conf) else 1.0
                    pts.append([xk, yk, ck])
                det["keypoints"] = pts
            dets.append(det)
        return dets

    def infer_batch(self, frames: list[Any]) -> list[list[dict]]:
        if not self._ready or not frames:
            return []
        results = self._model(
            frames,
            device=self._device,
            conf=self._conf,
            classes=self._classes if self._classes else None,
            verbose=False,
        )
        if not results:
            return []
        return [self._results_to_dets(r) for r in results]


class InferenceEngine(BaseService):
    def __init__(self, config, metrics, lifecycle=None):
        super().__init__("inference")
        self.config = config
        self.metrics = metrics
        self.lifecycle = lifecycle
        inf_cfg = getattr(config, "inference", {})
        self._backend = str(inf_cfg.get("backend", "stub")).lower()
        self._model_path = str(inf_cfg.get("model", ""))
        self._device = str(inf_cfg.get("device", "cpu"))
        self._conf_threshold = float(inf_cfg.get("conf", inf_cfg.get("conf_threshold", 0.25)))
        self._max_det = int(inf_cfg.get("max_det", 300))
        self._output_format = str(inf_cfg.get("output_format", "auto")).lower()
        self._preview = bool(inf_cfg.get("preview", False))
        self._preview_window = str(inf_cfg.get("preview_window", "HeliosNet"))
        self._nms_iou = float(inf_cfg.get("nms_iou", 0.0))
        self._classes = [int(c) for c in inf_cfg.get("classes", [])]
        self._batch_size = int(inf_cfg.get("batch_size", 1))
        self._batch_timeout_ms = int(inf_cfg.get("batch_timeout_ms", 10))
        self._queue: list[dict] = []
        self._last_flush = time.time()
        self._phone_id_enabled = bool(inf_cfg.get("phone_id_enabled", False))
        self._phone_catalog_dir = str(inf_cfg.get("phone_catalog_dir", "./data/phone_catalog"))
        self._phone_min_score = float(inf_cfg.get("phone_min_score", 0.45))
        self._phone_top_k = int(inf_cfg.get("phone_top_k", 3))
        self._phone_embed_backend = str(inf_cfg.get("phone_embed_backend", "resnet18"))
        self._phone_embed_device = str(inf_cfg.get("phone_embed_device", self._device))
        self._phone_cache_path = str(
            inf_cfg.get("phone_cache_path", "./data/phone_catalog_index.npz")
        )
        self._phone_matcher = None
        if self._phone_id_enabled:
            self._phone_matcher = PhoneModelMatcher(
                self._phone_catalog_dir,
                min_score=self._phone_min_score,
                top_k=self._phone_top_k,
                embed_backend=self._phone_embed_backend,
                embed_device=self._phone_embed_device,
                cache_path=self._phone_cache_path,
            )
            print(
                f"[phone_id] enabled backend={self._phone_matcher.active_backend} "
                f"catalog={self._phone_catalog_dir}",
                flush=True,
            )

        events_cfg = getattr(config, "events", {})
        self._zone_shapes = []
        for rule in events_cfg.get("rules", []):
            if str(rule.get("type", "")).lower() != "zone_entry":
                continue
            if "rect" in rule:
                rect = [float(x) for x in rule.get("rect", [0, 0, 0, 0])]
                if len(rect) == 4:
                    self._zone_shapes.append(("rect", rect))
            if "polygon" in rule:
                poly = [[float(a), float(b)] for a, b in rule.get("polygon", [])]
                if len(poly) >= 3:
                    self._zone_shapes.append(("poly", poly))
        self._runner = self._build_runner(self._backend, self._model_path)
        self._target_version = "default"
        self._shadow_runner = None
        self._shadow_version = ""
        self._shadow_enabled = bool(inf_cfg.get("shadow_enabled", True))
        self._shadow_sample_rate = int(inf_cfg.get("shadow_sample_rate", 10))
        self._processed_frames = 0
        self._runtime_mode = "normal"
        if self.lifecycle is not None:
            self._target_version, self._backend, self._model_path = self.lifecycle.current_target()
            self._runner = self._build_runner(self._backend, self._model_path)
            self._reload_shadow_runner()

    def handle(self, item) -> None:
        self._maybe_reload_runner()
        if self._batch_size <= 1:
            self._process_single(item)
            return
        self._queue.append(item)
        if len(self._queue) >= self._batch_size:
            self._flush()

    def tick(self) -> None:
        self._maybe_reload_runner()
        if self._batch_size <= 1:
            return
        now = time.time()
        if self._queue and (now - self._last_flush) * 1000.0 >= self._batch_timeout_ms:
            self._flush()

    def _flush(self) -> None:
        batch = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]
        frames = [it.get("frame") for it in batch]
        self._processed_frames += len(frames)
        dets_list: list[list[dict]] = []
        if hasattr(self._runner, "infer_batch"):
            t0 = time.time()
            dets_list = self._runner.infer_batch(frames)
            self.metrics.observe_inference(time.time() - t0)
        if not dets_list or len(dets_list) != len(batch):
            dets_list = []
            for f in frames:
                t0 = time.time()
                dets_list.append(self._runner.infer(f))
                self.metrics.observe_inference(time.time() - t0)
        for item, dets in zip(batch, dets_list):
            self._run_shadow_if_needed(item.get("frame"))
            self._finalize_item(item, dets)
        self._last_flush = time.time()

    def _process_single(self, item: dict) -> None:
        frame = item.get("frame")
        self._processed_frames += 1
        t0 = time.time()
        dets = self._runner.infer(frame)
        self.metrics.observe_inference(time.time() - t0)
        self._run_shadow_if_needed(frame)
        self._finalize_item(item, dets)

    def _finalize_item(self, item: dict, dets: list[dict]) -> None:
        frame = item.get("frame")
        source_id = item.get("source_id", "source")
        if self._classes and dets:
            dets = [d for d in dets if int(d.get("cls", -1)) in self._classes]
        dets = _nms(dets, self._nms_iou)
        if self._phone_matcher is not None and isinstance(frame, np.ndarray):
            self._annotate_phone_models(frame, dets)
        item["detections"] = dets
        if self._preview and isinstance(frame, np.ndarray):
            self._show_preview(frame, dets, source_id)
        self.metrics.add_detections(source_id, dets)
        self.push(item)

    def set_runtime_mode(self, mode: str) -> None:
        m = str(mode).lower()
        self._runtime_mode = m
        if m == "normal":
            self._nms_iou = float(getattr(self.config, "inference", {}).get("nms_iou", self._nms_iou))
            return
        if m == "degraded":
            self._nms_iou = max(self._nms_iou, 0.4)
            return
        # safe mode
        self._nms_iou = max(self._nms_iou, 0.5)
        self._batch_size = 1

    def _show_preview(self, frame: np.ndarray, detections: list[dict], source_id: str) -> None:
        vis = frame.copy()
        window = f"{self._preview_window}::{source_id}"
        skeleton_edges = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]
        for shape_type, data in self._zone_shapes:
            if shape_type == "rect":
                x1, y1, x2, y2 = [int(v) for v in data]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            elif shape_type == "poly":
                pts = np.array(data, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls = det.get("cls", -1)
            conf = det.get("conf", 0.0)
            name = det.get("label") or str(cls)
            if det.get("phone_model_guess"):
                name = f"{name}->{det.get('phone_model_guess')}"
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
            keypoints = det.get("keypoints", [])
            if isinstance(keypoints, list) and keypoints:
                for kp in keypoints:
                    if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                        continue
                    xk, yk = int(kp[0]), int(kp[1])
                    ck = float(kp[2]) if len(kp) > 2 else 1.0
                    if ck < 0.35:
                        continue
                    cv2.circle(vis, (xk, yk), 3, (0, 0, 255), -1)
                for a, b in skeleton_edges:
                    if a >= len(keypoints) or b >= len(keypoints):
                        continue
                    ka = keypoints[a]
                    kb = keypoints[b]
                    if len(ka) < 3 or len(kb) < 3:
                        continue
                    if float(ka[2]) < 0.35 or float(kb[2]) < 0.35:
                        continue
                    cv2.line(
                        vis,
                        (int(ka[0]), int(ka[1])),
                        (int(kb[0]), int(kb[1])),
                        (0, 128, 255),
                        2,
                    )
        cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit

    def _annotate_phone_models(self, frame: np.ndarray, dets: list[dict]) -> None:
        h, w = frame.shape[:2]
        for d in dets:
            label = str(d.get("label", "")).lower()
            cls = int(d.get("cls", -1))
            if label != "cell phone" and cls != 67:
                continue
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = frame[y1:y2, x1:x2]
            matches = self._phone_matcher.match(crop)
            if not matches:
                continue
            d["phone_candidates"] = [
                {"model": m.model_name, "score": round(m.score, 4)} for m in matches
            ]
            d["phone_model_guess"] = matches[0].model_name

    def _build_runner(self, backend: str, model_path: str):
        inf_cfg = getattr(self.config, "inference", {})
        if backend == "onnxruntime":
            return OnnxRuntimeRunner(
                model_path, self._device, self._conf_threshold, self._max_det, self._output_format
            )
        if backend == "groundingdino":
            return GroundingDinoRunner(
                str(inf_cfg.get("gd_config", "")),
                str(inf_cfg.get("gd_checkpoint", "")),
                str(inf_cfg.get("text_prompt", "")),
                float(inf_cfg.get("box_threshold", 0.35)),
                float(inf_cfg.get("text_threshold", 0.25)),
                str(inf_cfg.get("device", "cpu")),
            )
        if backend == "ultralytics":
            return UltralyticsRunner(model_path, self._device, self._conf_threshold, self._classes)
        return StubRunner()

    def _maybe_reload_runner(self) -> None:
        if self.lifecycle is None:
            return
        version, backend, model_path = self.lifecycle.current_target()
        backend = str(backend).lower()
        if version == self._target_version and backend == self._backend and model_path == self._model_path:
            return
        self._target_version = version
        self._backend = backend
        self._model_path = model_path
        self._runner = self._build_runner(self._backend, self._model_path)
        self._reload_shadow_runner()
        print(
            f"[inference] model switch version={version} backend={backend} model={model_path}",
            flush=True,
        )

    def _reload_shadow_runner(self) -> None:
        if self.lifecycle is None or not self._shadow_enabled:
            self._shadow_runner = None
            self._shadow_version = ""
            return
        target = self.lifecycle.shadow_target()
        if not target:
            self._shadow_runner = None
            self._shadow_version = ""
            return
        version, backend, model_path = target
        if version == self._target_version:
            self._shadow_runner = None
            self._shadow_version = ""
            return
        self._shadow_runner = self._build_runner(str(backend).lower(), str(model_path))
        self._shadow_version = version
        print(
            f"[inference] shadow mode version={version} backend={backend} model={model_path}",
            flush=True,
        )

    def _run_shadow_if_needed(self, frame: Any) -> None:
        if self._shadow_runner is None:
            return
        if self._shadow_sample_rate <= 0:
            return
        if self._processed_frames % self._shadow_sample_rate != 0:
            return
        try:
            _ = self._shadow_runner.infer(frame)
            self.metrics.inc_shadow(self._shadow_version or "shadow")
        except Exception:
            return
