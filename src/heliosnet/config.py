from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    node_id: str
    ingest: dict
    detect: dict
    track: dict
    events: dict
    edge_bus: dict
    runtime: dict


def load_config(path: str) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(**data)
