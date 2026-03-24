from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyState:
    level: str
    percent: int


class EnergyScheduler:
    def __init__(self, profiles):
        self._profiles = profiles or []
        self._state = EnergyState(level="NORMAL", percent=50)
        self._current_profile = None
        self._tick = 0

    def update_battery(self, percent: int) -> None:
        self._state.percent = percent
        self._current_profile = None
        for p in sorted(self._profiles, key=lambda x: x.threshold, reverse=True):
            if percent >= p.threshold:
                self._state.level = p.level
                self._current_profile = p
                return
        if self._profiles:
            self._state.level = self._profiles[-1].level
            self._current_profile = self._profiles[-1]

    def current(self) -> EnergyState:
        return self._state

    def current_profile(self):
        return self._current_profile

    def next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def should_run(self, service_name: str, tick: int) -> bool:
        level = (self._state.level or "NORMAL").upper()
        if level == "FULL":
            return True
        if level == "NORMAL":
            if service_name == "sync":
                return tick % 2 == 0
            return True
        if level == "LOW":
            if service_name in ("sync", "gossip", "raft"):
                return tick % 5 == 0
            return True
        if level == "CRITICAL":
            if service_name in ("sync", "gossip", "raft"):
                return tick % 20 == 0
            if service_name == "ingest":
                return tick % 2 == 0
            return True
        return True
