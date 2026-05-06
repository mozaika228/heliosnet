from __future__ import annotations

from collections import defaultdict, deque
import time

from core.service import BaseService


class MissionPlanner(BaseService):
    def __init__(self, config, metrics, energy=None, ingest=None, gossip=None, live_state=None, audit=None):
        super().__init__("mission_planner")
        self.config = config
        self.metrics = metrics
        self.energy = energy
        self.ingest = ingest
        self.gossip = gossip
        self.live_state = live_state
        self.audit = audit
        mission_cfg = getattr(config, "mission", {}) or {}
        self._enabled = bool(mission_cfg.get("enabled", True))
        self._retask_every_sec = float(mission_cfg.get("retask_every_sec", 5.0))
        self._top_k_sources = int(mission_cfg.get("top_k_sources", 1))
        self._alert_weight = float(mission_cfg.get("alert_weight", 1.0))
        self._slo_weight = float(mission_cfg.get("slo_weight", 2.0))
        self._fall_weight = float(mission_cfg.get("fall_weight", 2.5))
        self._drift_weight = float(mission_cfg.get("drift_weight", 1.5))
        self._risk_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._last_retask = 0.0
        self._last_decision = {}

    def handle(self, item) -> None:
        if not self._enabled:
            self.push(item)
            return
        source_id = str(item.get("source_id", "source"))
        risk = self._risk_from_events(item.get("events", []) or [])
        self._risk_hist[source_id].append(risk)
        self.push(item)

    def tick(self) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_retask < self._retask_every_sec:
            return
        self._last_retask = now
        ranked = self._rank_sources()
        target_sources = [x[0] for x in ranked[: self._top_k_sources]]
        battery = self._battery_percent()
        connectivity = self._connectivity_score()
        action, reason = self._decide_action(ranked, battery, connectivity, target_sources)
        if self.ingest is not None:
            self.ingest.set_target_sources(target_sources)
        self.metrics.watchdog_action(f"mission_{action}")
        evt = {
            "name": "MISSION_ACTION",
            "ts": now,
            "source_id": "mission",
            "payload": {
                "action": action,
                "target_sources": target_sources,
                "battery_percent": battery,
                "connectivity_score": round(connectivity, 3),
                "reason": reason,
            },
        }
        self._last_decision = evt["payload"]
        if self.live_state is not None:
            self.live_state.push_events([evt])
        if self.audit is not None:
            self.audit.write("mission_action", evt["payload"])

    def explain(self) -> dict:
        return dict(self._last_decision)

    def _risk_from_events(self, events: list[dict]) -> float:
        score = 0.0
        for e in events:
            name = str(e.get("name", "")).upper()
            if name == "FALL_ALERT":
                score += self._fall_weight
            elif name == "SLO_BREACH":
                score += self._slo_weight
            elif name == "DRIFT_ALERT":
                score += self._drift_weight
            else:
                score += self._alert_weight
        return score

    def _rank_sources(self) -> list[tuple[str, float]]:
        out = []
        for sid, rows in self._risk_hist.items():
            if not rows:
                continue
            out.append((sid, sum(rows) / len(rows)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _battery_percent(self) -> int:
        if self.energy is None:
            return 100
        return int(getattr(self.energy.current(), "percent", 100))

    def _connectivity_score(self) -> float:
        if self.gossip is None:
            return 1.0
        members = self.gossip.members() if hasattr(self.gossip, "members") else {}
        if not members:
            return 1.0
        alive = 0
        total = 0
        for rec in members.values():
            total += 1
            if bool(rec.get("alive", False)):
                alive += 1
        return alive / max(1, total)

    def _decide_action(
        self,
        ranked: list[tuple[str, float]],
        battery: int,
        connectivity: float,
        target_sources: list[str],
    ) -> tuple[str, str]:
        high_risk = ranked and ranked[0][1] >= 1.0
        if battery < 20:
            return "energy_save_focus", f"battery={battery}<20, focus={target_sources}"
        if connectivity < 0.5:
            return "link_degraded_local_priority", f"connectivity={connectivity:.2f}<0.5, focus={target_sources}"
        if high_risk:
            return "risk_focus", f"high_risk_source={ranked[0][0]} risk={ranked[0][1]:.2f}"
        return "balanced_scan", "no dominant risk; keep balanced observation"
