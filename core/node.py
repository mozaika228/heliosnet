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
from distributed.gossip import GossipNode
from distributed.raft import RaftController
from distributed.model_registry import ModelRegistry
from observability.metrics import Metrics
from core.service import BaseService


class Scheduler:
    def __init__(self, config, metrics, energy=None):
        self.config = config
        self.metrics = metrics
        self.energy = energy
        self._services: list[BaseService] = []
        runtime_cfg = getattr(config, "runtime", {})
        self._tick_ms = int(runtime_cfg.get("tick_ms", 10))

    def register(self, *services: BaseService) -> None:
        self._services.extend(services)

    def run(self) -> None:
        while True:
            tick = self.energy.next_tick() if self.energy is not None else 0
            for svc in self._services:
                if self.energy is not None and not self.energy.should_run(svc.name, tick):
                    continue
                svc.tick()
            if self._tick_ms > 0:
                time.sleep(self._tick_ms / 1000.0)


class Node:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.metrics = Metrics()
        obs_cfg = getattr(self.config, "observability", {})
        if bool(obs_cfg.get("metrics", False)):
            port = int(obs_cfg.get("metrics_port", 9090))
            self.metrics.start_http(port)

        self.energy = EnergyScheduler(self.config.energy.profiles)
        self.energy.update_battery(self.config.energy.simulated_percent)
        self.ingest = IngestManager(self.config, self.metrics, self.energy)
        self.inference = InferenceEngine(self.config, self.metrics)
        self.tracker = TrackCoordinator(self.config, self.metrics)
        self.events = EventsProcessor(self.config, self.metrics)
        self.store = EventStore(self.config, self.metrics)
        self.sync = SyncEngine(self.config, self.metrics, self.energy)
        self.gossip = GossipNode(self.config, self.metrics)
        self.raft = RaftController(self.config, self.metrics, self.gossip)
        self.model_registry = ModelRegistry(self.config, self.metrics, self.raft)

        self.scheduler = Scheduler(self.config, self.metrics, self.energy)

        self.ingest.set_next(self.inference)
        self.inference.set_next(self.tracker)
        self.tracker.set_next(self.events)
        self.events.set_next(self.store)
        self.store.set_next(self.sync)
        self.sync.set_next(self.gossip)
        self.gossip.set_next(self.raft)
        self.raft.set_next(self.model_registry)

        self.scheduler.register(
            self.ingest,
            self.inference,
            self.tracker,
            self.events,
            self.store,
            self.sync,
            self.gossip,
            self.raft,
            self.model_registry,
        )

    def run(self) -> None:
        try:
            self.scheduler.run()
        except KeyboardInterrupt:
            import cv2
            cv2.destroyAllWindows()


def main() -> None:
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config/config.example.yaml"
    node = Node(config_path)
    node.run()


if __name__ == "__main__":
    main()
