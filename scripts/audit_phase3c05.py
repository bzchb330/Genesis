from __future__ import annotations

import json
from pathlib import Path

from seqgrasp.config import ROOT
from seqgrasp.phase3c05 import exact_corridor_metric_audit, phase3c0_failure_audit


if __name__ == "__main__":
    output = ROOT / "outputs/phase3C05"
    output.mkdir(parents=True, exist_ok=True)
    failure = phase3c0_failure_audit()
    corridor = exact_corridor_metric_audit()
    (output / "failure_handoff_audit.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    (output / "corridor_metric_audit.json").write_text(json.dumps(corridor, indent=2), encoding="utf-8")
    print(json.dumps({
        "failure_trials": len(failure["trials"]),
        "timings": [{key: value for key, value in row.items() if key != "timeseries"}
                    for row in failure["trials"]],
        "corridor": corridor,
    }, indent=2))
