from __future__ import annotations

import argparse
import hashlib
import hmac
import json


def sign(payload: dict, secret: str) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    mac = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--path", default="")
    parser.add_argument("--backend", default="")
    args = parser.parse_args()

    payload = {"action": args.action}
    if args.version:
        payload["version"] = args.version
    if args.path:
        payload["path"] = args.path
    if args.backend:
        payload["backend"] = args.backend
    payload["signature"] = sign(dict(payload), args.secret)
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
