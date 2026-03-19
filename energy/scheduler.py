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

    def update_battery(self, percent: int) -> None:
        self._state.percent = percent
        for p in sorted(self._profiles, key=lambda x: x.threshold, reverse=True):
            if percent >= p.threshold:
                self._state.level = p.level
                return
        if self._profiles:
            self._state.level = self._profiles[-1].level

    def current(self) -> EnergyState:
        return self._state
