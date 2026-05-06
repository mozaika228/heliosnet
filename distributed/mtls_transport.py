from __future__ import annotations

import json
import socket
import ssl
from pathlib import Path
import time

from core.service import BaseService


class MTLSControlTransport(BaseService):
    def __init__(self, config, metrics, audit=None):
        super().__init__("mtls_transport")
        self.config = config
        self.metrics = metrics
        self.audit = audit
        cp_cfg = getattr(config, "control_plane", {}) or {}
        self._enabled = bool(cp_cfg.get("mtls_enabled", False))
        self._host = str(cp_cfg.get("mtls_host", "0.0.0.0"))
        self._port = int(cp_cfg.get("mtls_port", 9443))
        self._cert = str(cp_cfg.get("tls_cert_path", "./secrets/node.crt"))
        self._key = str(cp_cfg.get("tls_key_path", "./secrets/node.key"))
        self._ca = str(cp_cfg.get("tls_ca_path", "./secrets/ca.crt"))
        self._sock = None
        if self._enabled:
            self._open()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        if not self._enabled or self._sock is None:
            return
        try:
            client, _addr = self._sock.accept()
        except BlockingIOError:
            return
        except Exception:
            return
        try:
            with client:
                data = client.recv(65535)
                if not data:
                    return
                msg = json.loads(data.decode("utf-8"))
                if self.audit is not None:
                    self.audit.write("mtls_recv", {"ts": time.time(), "keys": list(msg.keys())[:8]})
                client.sendall(b'{"ok":true}')
        except Exception:
            return

    def _open(self) -> None:
        if not (Path(self._cert).exists() and Path(self._key).exists() and Path(self._ca).exists()):
            self._enabled = False
            return
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_cert_chain(certfile=self._cert, keyfile=self._key)
        ctx.load_verify_locations(cafile=self._ca)
        base = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        base.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        base.bind((self._host, self._port))
        base.listen(16)
        base.setblocking(False)
        self._sock = ctx.wrap_socket(base, server_side=True, do_handshake_on_connect=False)

