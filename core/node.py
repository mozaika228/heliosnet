from __future__ import annotations

import sys
import time

from config.settings import load_config
from energy.scheduler import EnergyScheduler
from ingest.manager import IngestManager
from inference.engine import InferenceEngine
from tracker.coordinator import TrackCoordinator
from events.processor import EventsProcessor
from events.store import EventStore
from sync.engine import SyncEngine
from observability.metrics import Metrics
from core.service import BaseService


class Scheduler:
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        self._services: list[BaseService] = []
        runtime_cfg = getattr(config, "runtime", {})
        self._tick_ms = int(runtime_cfg.get("tick_ms", 10))

    def register(self, *services: BaseService) -> None:
        self._services.extend(services)

    def run(self) -> None:
        while True:
            for svc in self._services:
                svc.tick()
            if self._tick_ms > 0:
                time.sleep(self._tick_ms / 1000.0)


class Node:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.metrics = Metrics()

        self.energy = EnergyScheduler(self.config.energy.profiles)
        self.ingest = IngestManager(self.config, self.metrics)
        self.inference = InferenceEngine(self.config, self.metrics)
        self.tracker = TrackCoordinator(self.config, self.metrics)
        self.events = EventsProcessor(self.config, self.metrics)
        self.store = EventStore(self.config, self.metrics)
        self.sync = SyncEngine(self.config, self.metrics)

        self.scheduler = Scheduler(self.config, self.metrics)

        self.ingest.set_next(self.inference)
        self.inference.set_next(self.tracker)
        self.tracker.set_next(self.events)
        self.events.set_next(self.store)
        self.store.set_next(self.sync)

        self.scheduler.register(
            self.ingest,
            self.inference,
            self.tracker,
            self.events,
            self.store,
            self.sync,
        )

    def run(self) -> None:
        self.scheduler.run()


def main() -> None:
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config/config.example.yaml"
    node = Node(config_path)
    node.run()


if __name__ == "__main__":
    main()
