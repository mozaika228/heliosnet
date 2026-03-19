from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class EnergyProfile:
    level: str
    threshold: int
    resolution: str
    fps: int
    batch: int
    sync_interval_sec: int


@dataclass
class EnergyConfig:
    profiles: list[EnergyProfile] = field(default_factory=list)


@dataclass
class Config:
    node_id: str
    ingest: dict
    inference: dict
    tracker: dict
    events: dict
    sync: dict
    observability: dict
    energy: EnergyConfig
    runtime: dict


def load_config(path: str) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    energy = data.get("energy", {})
    profiles = []
    for p in energy.get("profiles", []):
        profiles.append(
            EnergyProfile(
                level=str(p.get("level", "")),
                threshold=int(p.get("threshold", 0)),
                resolution=str(p.get("resolution", "")),
                fps=int(p.get("fps", 0)),
                batch=int(p.get("batch", 0)),
                sync_interval_sec=int(p.get("sync_interval_sec", 0)),
            )
        )
    energy_cfg = EnergyConfig(profiles=profiles)
    return Config(
        node_id=str(data.get("node_id", "edge-001")),
        ingest=data.get("ingest", {}),
        inference=data.get("inference", {}),
        tracker=data.get("tracker", {}),
        events=data.get("events", {}),
        sync=data.get("sync", {}),
        observability=data.get("observability", {}),
        energy=energy_cfg,
        runtime=data.get("runtime", {}),
    )
