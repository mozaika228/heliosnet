from __future__ import annotations

import json
from pathlib import Path
import socket
import time

from core.service import BaseService
from distributed.security import MessageSecurity


class GossipNode(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("gossip")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        dist_cfg = getattr(config, "distributed", {})
        self._enabled = bool(dist_cfg.get("enabled", True))
        self._bind_host = str(dist_cfg.get("bind_host", "0.0.0.0"))
        self._bind_port = int(dist_cfg.get("bind_port", 7946))
        self._interval = int(dist_cfg.get("gossip_interval_sec", 2))
        self._max_silence = int(dist_cfg.get("peer_timeout_sec", 10))
        self._peers = {
            str(p): {"last_seen": 0.0, "state": {}, "alive": False}
            for p in dist_cfg.get("bootstrap_peers", [])
        }
        self._last_tick = 0.0
        self._control_inbox: list[dict] = []
        self._state_path = Path(dist_cfg.get("cluster_state_path", "./data/cluster_state.json"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._security = MessageSecurity(str(dist_cfg.get("shared_secret", "")))
        self._require_signed = bool(dist_cfg.get("require_signed_control", False))
        self._sock = None
        if self._enabled:
            self._open_socket()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        self._recv_loop(now)
        if now - self._last_tick < self._interval:
            return
        self._last_tick = now
        self._broadcast(now)
        self._cleanup(now)
        self._write_local_snapshot(now)

    def alive_peer_count(self) -> int:
        count = 0
        for rec in self._peers.values():
            if bool(rec.get("alive", False)):
                count += 1
        return count

    def _open_socket(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind((self._bind_host, self._bind_port))
            self._sock = sock
        except Exception:
            self._sock = None

    def _recv_loop(self, now: float) -> None:
        if self._sock is None:
            return
        for _ in range(64):
            try:
                data, addr = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            signature = msg.pop("signature", None)
            if self._require_signed and not self._security.verify(msg, signature):
                if self.audit is not None:
                    self.audit.write("gossip_invalid_signature", {"from": f"{addr[0]}:{addr[1]}"})
                continue
            kind = str(msg.get("kind", "HEARTBEAT")).upper()
            if kind == "CONTROL":
                self._control_inbox.append(msg)
                continue
            peer_key = f"{addr[0]}:{addr[1]}"
            rec = self._peers.get(peer_key, {"last_seen": 0.0, "state": {}, "alive": True})
            rec["last_seen"] = now
            rec["alive"] = True
            rec["state"] = msg.get("state", {})
            self._peers[peer_key] = rec

    def _broadcast(self, now: float) -> None:
        if self._sock is None:
            return
        payload = self._heartbeat_payload(now)
        signed = self._sign(payload)
        encoded = json.dumps(signed, ensure_ascii=True).encode("utf-8")
        for key in list(self._peers.keys()):
            host, port = self._parse_peer(key)
            if host is None:
                continue
            try:
                self._sock.sendto(encoded, (host, port))
            except Exception:
                continue

    def _cleanup(self, now: float) -> None:
        for rec in self._peers.values():
            last_seen = float(rec.get("last_seen", 0.0))
            rec["alive"] = (now - last_seen) <= self._max_silence

    def _write_local_snapshot(self, now: float) -> None:
        payload = {
            "node_id": self.config.node_id,
            "ts": now,
            "bind": f"{self._bind_host}:{self._bind_port}",
            "peers": self._peers,
            "battery_percent": getattr(getattr(self.config, "energy", None), "simulated_percent", None),
        }
        self._state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _heartbeat_payload(self, now: float) -> dict:
        return {
            "kind": "HEARTBEAT",
            "node_id": self.config.node_id,
            "ts": now,
            "state": {
                "battery": getattr(getattr(self.config, "energy", None), "simulated_percent", None),
            },
        }

    def _parse_peer(self, key: str):
        if ":" not in key:
            return None, None
        host, port_str = key.rsplit(":", 1)
        try:
            port = int(port_str)
        except Exception:
            return None, None
        return host, port

    async def join(self, peers: list[tuple[str, int]]) -> None:
        for host, port in peers:
            key = f"{host}:{port}"
            if key not in self._peers:
                self._peers[key] = {"last_seen": 0.0, "alive": False, "state": {}}

    def members(self) -> dict:
        return self._peers

    def drain_control(self) -> list[dict]:
        out = self._control_inbox[:]
        self._control_inbox = []
        return out

    def broadcast_control(self, payload: dict) -> None:
        if self._sock is None:
            return
        msg = {
            "kind": "CONTROL",
            "node_id": self.config.node_id,
            "ts": time.time(),
            "payload": payload,
        }
        signed = self._sign(msg)
        encoded = json.dumps(signed, ensure_ascii=True).encode("utf-8")
        for key in list(self._peers.keys()):
            host, port = self._parse_peer(key)
            if host is None:
                continue
            try:
                self._sock.sendto(encoded, (host, port))
            except Exception:
                continue

    def _sign(self, msg: dict) -> dict:
        out = dict(msg)
        if self._security.enabled:
            out["signature"] = self._security.sign(msg)
        return out
