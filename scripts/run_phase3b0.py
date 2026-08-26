from __future__ import annotations

import argparse
import json

from seqgrasp.phase3b0 import generate_dataset
from seqgrasp.phase3b0_analysis import analyze_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and analyze the Phase 3B-0 Shadow reset dataset")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--attempt-cap", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-steps", type=int, default=1000)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        manifest = generate_dataset(
            target=args.target,
            attempt_cap=args.attempt_cap,
            workers=args.workers,
            batch_size=args.batch_size,
            retention_steps=args.retention_steps,
        )
        print(json.dumps(manifest, indent=2), flush=True)
        if not manifest["target_reached"]:
            print("PHASE3B0_RESET_TARGET_NOT_REACHED", flush=True)
    summary = analyze_dataset()
    print(
        json.dumps(
            {
                "attempts": summary["sampling"]["total_attempts"],
                "raw_valid": summary["sampling"]["raw_valid_release_states"],
                "deduplicated": summary["deduplication"]["exact_unique_count"],
                "target_reached": summary["sampling"]["target_reached"],
                "phase3a_reproduced": summary["phase3a_reproduction"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
