from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:  # pragma: no cover - optional dependency
    torch = None
    resnet18 = None
    ResNet18_Weights = None


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v
    return v / n


def _embed_histogram(img: np.ndarray) -> np.ndarray:
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


class _ResnetEmbedder:
    def __init__(self, device: str = "cpu") -> None:
        if torch is None or resnet18 is None or ResNet18_Weights is None:
            raise RuntimeError("torchvision backend unavailable")
        self.device = torch.device(device if device in ("cpu", "cuda") else "cpu")
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        # Feature extractor: all layers except final FC.
        self._model = torch.nn.Sequential(*(list(model.children())[:-1])).to(self.device)
        self._model.eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        x = (x - self._mean) / self._std
        with torch.no_grad():
            out = self._model(x)
        feat = out.flatten().detach().cpu().numpy().astype(np.float32)
        return _normalize(feat)


@dataclass
class Match:
    model_name: str
    score: float


class PhoneModelMatcher:
    def __init__(
        self,
        catalog_dir: str,
        min_score: float = 0.45,
        top_k: int = 3,
        embed_backend: str = "resnet18",
        embed_device: str = "cpu",
        cache_path: str = "./data/phone_catalog_index.npz",
    ):
        self.catalog_dir = Path(catalog_dir)
        self.min_score = float(min_score)
        self.top_k = int(top_k)
        self.embed_backend = str(embed_backend).lower()
        self.embed_device = str(embed_device).lower()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._loaded = False
        self._names: list[str] = []
        self._embs: np.ndarray | None = None
        self._embed_fn: Callable[[np.ndarray], np.ndarray] = _embed_histogram
        self.active_backend = "histogram"

    def reload(self) -> None:
        self._init_embedder()
        if self._load_cache_if_valid():
            self._loaded = True
            return

        names = []
        embs = []
        if self.catalog_dir.exists():
            for p in sorted(self.catalog_dir.glob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                name = p.stem.split("__", 1)[0].strip() or p.stem
                names.append(name)
                embs.append(self._embed_fn(img))

        self._names = names
        self._embs = np.vstack(embs).astype(np.float32) if embs else np.zeros((0, 1), dtype=np.float32)
        self._save_cache()
        self._loaded = True

    def match(self, crop_bgr: np.ndarray) -> list[Match]:
        if not self._loaded:
            self.reload()
        if crop_bgr is None or crop_bgr.size == 0 or self._embs is None or self._embs.shape[0] == 0:
            return []
        q = self._embed_fn(crop_bgr)
        # cosine since vectors are normalized
        scores = self._embs @ q
        order = np.argsort(-scores)
        out = []
        for idx in order[: self.top_k]:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            out.append(Match(model_name=self._names[idx], score=score))
        return out

    def _init_embedder(self) -> None:
        if self.embed_backend == "resnet18":
            try:
                embedder = _ResnetEmbedder(self.embed_device)
                self._embed_fn = embedder.embed
                self.active_backend = "resnet18"
                return
            except Exception:
                pass
        self._embed_fn = _embed_histogram
        self.active_backend = "histogram"

    def _catalog_stamp(self) -> str:
        if not self.catalog_dir.exists():
            return ""
        rows = []
        for p in sorted(self.catalog_dir.glob("*")):
            if p.is_file():
                rows.append(f"{p.name}:{int(p.stat().st_mtime)}:{p.stat().st_size}")
        return "|".join(rows)

    def _save_cache(self) -> None:
        try:
            stamp = self._catalog_stamp()
            names = np.array(self._names, dtype=object)
            embs = self._embs if self._embs is not None else np.zeros((0, 1), dtype=np.float32)
            np.savez_compressed(
                str(self.cache_path),
                names=names,
                embs=embs,
                stamp=np.array([stamp], dtype=object),
                backend=np.array([self.active_backend], dtype=object),
            )
        except Exception:
            return

    def _load_cache_if_valid(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(str(self.cache_path), allow_pickle=True)
            stamp = str(data["stamp"][0])
            backend = str(data["backend"][0])
            if stamp != self._catalog_stamp() or backend != self.active_backend:
                return False
            self._names = [str(x) for x in data["names"].tolist()]
            self._embs = data["embs"].astype(np.float32)
            return True
        except Exception:
            return False
