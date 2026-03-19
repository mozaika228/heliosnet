from __future__ import annotations


class GossipNode:
    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = 7946) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port

    async def join(self, peers: list[tuple[str, int]]) -> None:
        _ = peers
        return None
