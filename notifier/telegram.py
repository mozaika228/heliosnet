from __future__ import annotations

import json
import urllib.parse
import urllib.request

from core.service import BaseService


class TelegramNotifier(BaseService):
    def __init__(self, config, metrics):
        super().__init__("telegram")
        self.config = config
        self.metrics = metrics
        obs_cfg = getattr(config, "observability", {})
        self._enabled = bool(obs_cfg.get("telegram_enabled", False))
        self._token = str(obs_cfg.get("telegram_bot_token", ""))
        self._chat_id = str(obs_cfg.get("telegram_chat_id", ""))
        self._min_interval_sec = int(obs_cfg.get("telegram_min_interval_sec", 5))
        self._last_sent = 0.0

    def handle(self, item) -> None:
        events = item.get("events", [])
        if self._enabled and self._token and self._chat_id and events:
            for evt in events:
                self._maybe_send(evt)
        self.push(item)

    def _maybe_send(self, evt: dict) -> None:
        import time

        now = time.time()
        if now - self._last_sent < self._min_interval_sec:
            return
        self._last_sent = now
        name = str(evt.get("name", "event"))
        src = str(evt.get("source_id", "source"))
        payload = evt.get("payload", {}) or {}
        text = f"HeliosNet alert\\n{name}\\nsource={src}\\npayload={json.dumps(payload, ensure_ascii=True)}"
        self._send_text(text)

    def _send_text(self, text: str) -> None:
        try:
            base = f"https://api.telegram.org/bot{self._token}/sendMessage"
            params = urllib.parse.urlencode({"chat_id": self._chat_id, "text": text})
            url = f"{base}?{params}"
            req = urllib.request.Request(url=url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            return
