from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic replay/verification for recorded incidents.")
    p.add_argument("--incident", required=True, help="Incident folder path or incident id")
    p.add_argument("--root", default="./data/incidents", help="Incidents root when --incident is id")
    return p.parse_args()


def _resolve_incident(incident: str, root: str) -> Path:
    p = Path(incident)
    if p.exists():
        return p
    return Path(root) / incident


def main() -> None:
    args = parse_args()
    incident_dir = _resolve_incident(args.incident, args.root)
    manifest_path = incident_dir / "manifest.json"
    item_path = incident_dir / "item.json"
    if not manifest_path.exists() or not item_path.exists():
        raise SystemExit(f"Missing files for incident replay in {incident_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = json.loads(item_path.read_text(encoding="utf-8"))
    decision = {
        "events": item.get("events", []),
        "detections_count": len(item.get("detections", []) or []),
        "tracks_count": len(item.get("tracks", []) or []),
    }
    replay_hash = hashlib.sha1(
        json.dumps(decision, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    expected = str(manifest.get("decision_hash", ""))
    ok = replay_hash == expected
    print(f"incident_dir={incident_dir}")
    print(f"expected_hash={expected}")
    print(f"replay_hash={replay_hash}")
    print(f"deterministic_match={ok}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
