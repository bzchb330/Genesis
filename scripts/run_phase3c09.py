from __future__ import annotations

import json

from seqgrasp.phase3c09 import run_phase3c09


if __name__ == "__main__":
    result = run_phase3c09()
    print(json.dumps({
        "trajectory_count": result["trajectory"]["N"],
        "cspace_classifications": [row["classification"] for row in result["cspace"]["states"]],
        "contact_classifications": [row["classification"] for row in result["contact_accessibility"].get("modes", [])],
        "storage_candidates": result["storage_manifold"]["valid_configuration_center_pairs"],
        "storage_basins": result["storage_manifold"]["basin_count"],
        "contract": result["contract"],
    }, indent=2))
