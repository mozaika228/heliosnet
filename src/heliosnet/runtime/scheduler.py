from __future__ import annotations

import time


class BaseService:
    def __init__(self, name: str):
        self.name = name
        self._next = None

    def set_next(self, svc: "BaseService") -> None:
        self._next = svc

    def push(self, item) -> None:
        if self._next is not None:
            self._next.handle(item)

    def handle(self, item) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        # Optional periodic work
        time.sleep(0.0)


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
