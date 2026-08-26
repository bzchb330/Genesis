from __future__ import annotations

import argparse
import json

from seqgrasp.phase3b05 import (
    audit_phase3b0,
    generate_active_calibration,
    generate_feasibility_map,
    load_feasibility_rows,
)
from seqgrasp.phase3b05_analysis import analyze_phase3b05, render_representative_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3B-0.5 pre-RL calibration")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidates-per-level", type=int, default=200)
    parser.add_argument("--active-sources-per-level", type=int, default=2)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-feasibility", action="store_true")
    parser.add_argument("--skip-active", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    args = parser.parse_args()

    if not args.skip_audit:
        audit_phase3b0()
    if not args.skip_feasibility:
        generate_feasibility_map(
            candidates_per_level=args.candidates_per_level,
            workers=args.workers,
        )
    rows = load_feasibility_rows()
    if not args.skip_active:
        generate_active_calibration(rows, per_level=args.active_sources_per_level)
    summary = None
    if not args.skip_analysis:
        summary = analyze_phase3b05()
    videos = render_representative_videos() if args.render_videos else None
    print(
        json.dumps(
            {
                "feasibility_rows": len(rows),
                "active_trials": None if summary is None else summary["active"]["trial_count"],
                "ppo_readiness": None if summary is None else summary["ppo_readiness"]["status"],
                "videos": videos,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
