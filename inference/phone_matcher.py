from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v
    return v / n


def _embed_bgr(img: np.ndarray) -> np.ndarray:
    # Lightweight embedding for offline matching: HSV histogram + coarse shape.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    resized = cv2.resize(img, (32, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    proj_x = gray.mean(axis=0)
    proj_y = gray.mean(axis=1)
    feat = np.concatenate([h_hist, s_hist, v_hist, proj_x, proj_y]).astype(np.float32)
    return _normalize(feat)


@dataclass
class Match:
    model_name: str
    score: float


class PhoneModelMatcher:
    def __init__(self, catalog_dir: str, min_score: float = 0.45, top_k: int = 3):
        self.catalog_dir = Path(catalog_dir)
        self.min_score = float(min_score)
        self.top_k = int(top_k)
        self._db: list[tuple[str, np.ndarray]] = []
        self._loaded = False

    def reload(self) -> None:
        self._db = []
        if not self.catalog_dir.exists():
            self._loaded = True
            return
        for p in sorted(self.catalog_dir.glob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            name = p.stem.split("__", 1)[0].strip() or p.stem
            emb = _embed_bgr(img)
            self._db.append((name, emb))
        self._loaded = True

    def match(self, crop_bgr: np.ndarray) -> list[Match]:
        if not self._loaded:
            self.reload()
        if crop_bgr is None or crop_bgr.size == 0 or not self._db:
            return []
        q = _embed_bgr(crop_bgr)
        scored = []
        for name, emb in self._db:
            score = float(np.dot(q, emb))
            scored.append(Match(model_name=name, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        out = [m for m in scored[: self.top_k] if m.score >= self.min_score]
        return out
