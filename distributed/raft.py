from __future__ import annotations

import json
from pathlib import Path
import random
import time

from core.service import BaseService


class RaftController(BaseService):
    def __init__(self, config, metrics, gossip=None, audit=None):
        super().__init__("raft")
        self.config = config
        self.metrics = metrics
        self.gossip = gossip
        self.audit = audit
        dist_cfg = getattr(config, "distributed", {})
        self._enabled = bool(dist_cfg.get("enabled", True))
        self._state_path = Path(dist_cfg.get("raft_state_path", "./data/raft_state.json"))
        self._log_path = Path(dist_cfg.get("raft_log_path", "./data/raft_log.jsonl"))
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._term = 0
        self._role = "follower"
        self._leader_id = None
        self._last_heartbeat = time.time()
        self._election_timeout = self._new_timeout()
        self._heartbeat_interval = float(dist_cfg.get("raft_heartbeat_sec", 1.0))
        self._last_tick = 0.0

        self._last_log_index = 0
        self._commit_index = 0
        self._pending: list[dict] = []
        self._acks: dict[int, set[str]] = {}
        self._min_cluster_size = int(dist_cfg.get("raft_min_cluster_size", 1))
        self._load_state()

    @property
    def role(self) -> str:
        return self._role

    @property
    def is_leader(self) -> bool:
        return self._role == "leader"

    @property
    def leader_id(self):
        return self._leader_id

    @property
    def term(self) -> int:
        return self._term

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        self._process_control(now)
        if self._role == "leader":
            if now - self._last_tick >= self._heartbeat_interval:
                self._last_tick = now
                self._send_heartbeat()
                self._try_commit()
                self._write_state(now)
            return
        if now - self._last_heartbeat >= self._election_timeout:
            self._start_election(now)

    def propose(self, command: dict) -> bool:
        if self._role != "leader":
            return False
        self._last_log_index += 1
        entry = {
            "index": self._last_log_index,
            "ts": time.time(),
            "term": self._term,
            "leader_id": self.config.node_id,
            "command": command,
            "committed": False,
        }
        self._pending.append(entry)
        self._acks[self._last_log_index] = {self.config.node_id}
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        if self.audit is not None:
            self.audit.write("raft_propose", {"index": self._last_log_index, "term": self._term, "command": command})
        self._broadcast_append(entry)
        return True

    def _start_election(self, now: float) -> None:
        self._term += 1
        # In this foundation phase, votes are derived from known alive peers.
        cluster_size = self._cluster_size()
        votes = 1 + self._alive_peer_count()
        quorum = (cluster_size // 2) + 1
        if votes >= quorum:
            self._role = "leader"
            self._leader_id = self.config.node_id
            if self.audit is not None:
                self.audit.write("raft_elected_leader", {"term": self._term, "node_id": self.config.node_id})
        else:
            self._role = "follower"
            self._leader_id = None
        self._last_heartbeat = now
        self._election_timeout = self._new_timeout()
        self._write_state(now)

    def _try_commit(self) -> None:
        if not self._pending:
            return
        cluster_size = self._cluster_size()
        quorum = (cluster_size // 2) + 1
        remaining = []
        for entry in self._pending:
            idx = int(entry["index"])
            votes = len(self._acks.get(idx, {self.config.node_id}))
            if votes >= quorum:
                entry["committed"] = True
                self._commit_index = max(self._commit_index, idx)
            else:
                remaining.append(entry)
        self._pending = remaining
        self._rewrite_log_mark_committed()

    def _rewrite_log_mark_committed(self) -> None:
        if not self._log_path.exists():
            return
        rows = []
        with self._log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                idx = int(row.get("index", 0))
                row["committed"] = idx <= self._commit_index
                rows.append(row)
        with self._log_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _new_timeout(self) -> float:
        return random.uniform(1.5, 3.0)

    def _cluster_size(self) -> int:
        size = 1 + self._configured_peer_count()
        if size < self._min_cluster_size:
            size = self._min_cluster_size
        return size

    def _configured_peer_count(self) -> int:
        dist_cfg = getattr(self.config, "distributed", {})
        return len(dist_cfg.get("bootstrap_peers", []))

    def _alive_peer_count(self) -> int:
        if self.gossip is None:
            return 0
        return int(self.gossip.alive_peer_count())

    def _process_control(self, now: float) -> None:
        if self.gossip is None:
            return
        for msg in self.gossip.drain_control():
            payload = msg.get("payload", {}) or {}
            mtype = str(payload.get("type", "")).upper()
            src = str(msg.get("node_id", ""))
            term = int(payload.get("term", 0))
            if mtype == "RAFT_HEARTBEAT":
                if term >= self._term:
                    self._term = term
                    self._role = "follower"
                    self._leader_id = src
                    self._last_heartbeat = now
                    self._election_timeout = self._new_timeout()
            elif mtype == "RAFT_APPEND":
                if term >= self._term:
                    self._term = term
                    self._role = "follower"
                    self._leader_id = src
                    self._last_heartbeat = now
                    entry = payload.get("entry")
                    if entry is not None:
                        self._append_replica(entry)
                    self._send_ack(payload.get("index"))
            elif mtype == "RAFT_ACK" and self._role == "leader":
                idx = int(payload.get("index", 0))
                if idx not in self._acks:
                    self._acks[idx] = {self.config.node_id}
                self._acks[idx].add(src)

    def _send_heartbeat(self) -> None:
        if self.gossip is None:
            return
        self.gossip.broadcast_control(
            {"type": "RAFT_HEARTBEAT", "term": self._term, "leader_id": self.config.node_id}
        )

    def _broadcast_append(self, entry: dict) -> None:
        if self.gossip is None:
            return
        self.gossip.broadcast_control(
            {
                "type": "RAFT_APPEND",
                "term": self._term,
                "leader_id": self.config.node_id,
                "index": entry.get("index"),
                "entry": entry,
            }
        )

    def _send_ack(self, index) -> None:
        if self.gossip is None or index is None:
            return
        self.gossip.broadcast_control(
            {
                "type": "RAFT_ACK",
                "term": self._term,
                "leader_id": self._leader_id,
                "index": int(index),
            }
        )

    def _append_replica(self, entry: dict) -> None:
        idx = int(entry.get("index", 0))
        if idx <= self._last_log_index:
            return
        self._last_log_index = idx
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _write_state(self, now: float) -> None:
        payload = {
            "node_id": self.config.node_id,
            "term": self._term,
            "role": self._role,
            "leader_id": self._leader_id,
            "ts": now,
            "last_log_index": self._last_log_index,
            "commit_index": self._commit_index,
            "alive_peers": self._alive_peer_count(),
        }
        self._state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._term = int(payload.get("term", 0))
        self._role = str(payload.get("role", "follower"))
        self._leader_id = payload.get("leader_id")
        self._last_log_index = int(payload.get("last_log_index", 0))
        self._commit_index = int(payload.get("commit_index", 0))
