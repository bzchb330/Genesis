#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.physics_validation import diagnostic_rows, run_physics_validation
from seqgrasp.phase2_config import load_phase2_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 fixed-base physics replay")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    args = parser.parse_args()
    phase2, config_path = load_phase2_config(ROOT / args.config)
    output = ROOT / phase2.persistence.output_dir / "physics_validation"
    _, summary = run_physics_validation(phase2, config_path, output)
    print(f"{'quantity':28} {'measurement':24} {'configured check'}")
    print("-" * 76)
    for quantity, measurement, check in diagnostic_rows(summary, list(load_configs().hand.finger_geom_mapping)):
        print(f"{quantity:28} {measurement:24} {check}")
    print(f"\nPHASE2_PHYSICS_GATE={summary['verdict']}")
    if summary["missing_pi_inputs"]:
        print("missing PI inputs:", ", ".join(summary["missing_pi_inputs"]))
    print(f"summary: {output / 'physics_validation_summary.json'}")
    print(f"plot: {output / 'physics_validation.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
