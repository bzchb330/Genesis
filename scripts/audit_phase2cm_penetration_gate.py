#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from seqgrasp.experiments.phase2cm_penetration import run_penetration_gate_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the unchanged Phase 2CM penetration gate")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = run_penetration_gate_audit(args.workers)
    print(json.dumps({
        "status": payload["status"],
        "threshold_m": payload["threshold_m"],
        "frozen_release_violations": payload["frozen_200"]["violating_at_release_boundary_count"],
        "eligible_release_valid": payload["eligible_1521"]["penetration_valid_at_release_boundary_count"],
        "at_least_200": payload["eligible_1521"]["at_least_200_release_valid_states"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
