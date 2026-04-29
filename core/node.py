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
from distributed.policy import PolicyEngine
from distributed.config_consensus import ConfigConsensusService
from observability.metrics import Metrics
from core.service import BaseService
from core.audit import AuditLog
from core.live_state import LiveState
from core.command_center import CommandCenter
from core.safety import IncidentRecorder, SLOMonitor
from core.watchdog import WatchdogService
from notifier.telegram import TelegramNotifier
from ui.server import WebUIService
from mlops.drift import DriftMonitorService
from mlops.evaluator import ContinuousEvalService
from fusion.coordinator import FusionCoordinator


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
                svc.heartbeat()
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
        dist_cfg = getattr(self.config, "distributed", {})
        self.audit = AuditLog(str(dist_cfg.get("audit_log_path", "./data/audit_log.jsonl")))
        self.live_state = LiveState(max_events=1000)
        self.ingest = IngestManager(self.config, self.metrics, self.energy)
        self.fusion = FusionCoordinator(self.config, self.metrics, self.audit)
        self.gossip = GossipNode(self.config, self.metrics, self.audit)
        self.raft = RaftController(self.config, self.metrics, self.gossip, self.audit)
        self.policy = PolicyEngine(self.config, self.metrics, self.audit)
        self.config_consensus = ConfigConsensusService(self.config, self.metrics, self.raft, self.audit)
        self.model_registry = ModelRegistry(self.config, self.metrics, self.raft, self.audit)
        self.inference = InferenceEngine(self.config, self.metrics, self.model_registry)
        self.drift = DriftMonitorService(self.config, self.metrics, self.audit)
        self.tracker = TrackCoordinator(self.config, self.metrics, self.live_state)
        self.events = EventsProcessor(self.config, self.metrics)
        self.slo_monitor = SLOMonitor(self.config, self.metrics, self.audit)
        self.incident_recorder = IncidentRecorder(self.config, self.metrics, self.audit)
        self.store = EventStore(self.config, self.metrics, self.live_state)
        self.telegram = TelegramNotifier(self.config, self.metrics)
        self.sync = SyncEngine(self.config, self.metrics, self.energy)
        self.command_center = CommandCenter(
            self.config,
            self.metrics,
            self.energy,
            self.model_registry,
            self.policy,
            self.config_consensus,
        )
        self.web_ui = WebUIService(self.config, self.metrics, self.live_state)
        self.continuous_eval = ContinuousEvalService(self.config, self.metrics, self.inference, self.audit)
        self.watchdog = WatchdogService(
            self.config,
            self.metrics,
            {
                "ingest": self.ingest,
                "inference": self.inference,
                "fusion": self.fusion,
                "drift_monitor": self.drift,
                "tracker": self.tracker,
                "events": self.events,
                "slo_monitor": self.slo_monitor,
                "incident_recorder": self.incident_recorder,
                "event_store": self.store,
                "sync": self.sync,
                "gossip": self.gossip,
                "raft": self.raft,
                "model_registry": self.model_registry,
                "policy": self.policy,
                "config_consensus": self.config_consensus,
                "command_center": self.command_center,
                "web_ui": self.web_ui,
                "continuous_eval": self.continuous_eval,
            },
            self.audit,
        )

        self.scheduler = Scheduler(self.config, self.metrics, self.energy)

        self.ingest.set_next(self.fusion)
        self.fusion.set_next(self.inference)
        self.inference.set_next(self.drift)
        self.drift.set_next(self.tracker)
        self.tracker.set_next(self.events)
        self.events.set_next(self.slo_monitor)
        self.slo_monitor.set_next(self.incident_recorder)
        self.incident_recorder.set_next(self.store)
        self.store.set_next(self.telegram)
        self.telegram.set_next(self.sync)
        self.sync.set_next(self.gossip)
        self.gossip.set_next(self.raft)
        self.raft.set_next(self.config_consensus)
        self.config_consensus.set_next(self.model_registry)

        self.scheduler.register(
            self.ingest,
            self.fusion,
            self.inference,
            self.drift,
            self.tracker,
            self.events,
            self.slo_monitor,
            self.incident_recorder,
            self.store,
            self.telegram,
            self.sync,
            self.gossip,
            self.raft,
            self.policy,
            self.config_consensus,
            self.model_registry,
            self.command_center,
            self.web_ui,
            self.continuous_eval,
            self.watchdog,
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
