"""Refresh downstream C09-C/D audits without repeating C09-B grids."""
from __future__ import annotations

import json

from seqgrasp.phase3c09 import OUTPUT, contact_accessibility_audit, storage_manifold_audit


if __name__ == "__main__":
    trajectory = json.loads((OUTPUT / "trajectory_failure_audit.json").read_text(encoding="utf-8"))
    cspace = json.loads((OUTPUT / "cspace_connectivity_audit.json").read_text(encoding="utf-8"))
    contact = contact_accessibility_audit(trajectory, cspace)
    storage = storage_manifold_audit(trajectory)
    (OUTPUT / "contact_accessibility_audit.json").write_text(json.dumps(contact, indent=2), encoding="utf-8")
    (OUTPUT / "storage_manifold_audit.json").write_text(json.dumps(storage, indent=2), encoding="utf-8")
    result_path = OUTPUT / "phase3c09_results.json"; result = json.loads(result_path.read_text(encoding="utf-8"))
    result["contact_accessibility"] = contact; result["storage_manifold"] = storage
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"contact": [row["classification"] for row in contact["modes"]],
                      "basins": storage["basin_count"],
                      "reachability": [row["reachability"] for row in storage["basins"]]}, indent=2))
