from __future__ import annotations

import time


class BaseService:
    def __init__(self, name: str):
        self.name = name
        self._next = None
        self._last_heartbeat = 0.0

    def set_next(self, svc: "BaseService") -> None:
        self._next = svc

    def push(self, item) -> None:
        if self._next is not None:
            self._next.handle(item)

    def handle(self, item) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        time.sleep(0.0)

    def heartbeat(self) -> None:
        self._last_heartbeat = time.time()

    def last_heartbeat(self) -> float:
        return self._last_heartbeat
