class Metrics:
    def __init__(self) -> None:
        self.counters = {}

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value
