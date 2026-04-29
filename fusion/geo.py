from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraGeoModel:
    source_id: str
    lat: float
    lon: float
    meters_per_pixel: float


def pixel_to_world(model: CameraGeoModel, x: float, y: float) -> tuple[float, float]:
    # Lightweight approximation for MVP. Replace with full camera model + CRS transform.
    dlat = (y * model.meters_per_pixel) / 111_320.0
    dlon = (x * model.meters_per_pixel) / 111_320.0
    return model.lat + dlat, model.lon + dlon

