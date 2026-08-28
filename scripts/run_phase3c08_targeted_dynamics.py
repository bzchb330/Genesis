from __future__ import annotations

import json

from seqgrasp.phase3c08 import freeze_targeted_dynamic_manifest, run_targeted_dynamics


if __name__ == "__main__":
    manifest = freeze_targeted_dynamic_manifest()
    print(json.dumps({"manifest_frozen": True, "trial_count": manifest["trial_count"],
                      "state_ids": manifest["state_ids"], "sha256": manifest["sha256"]}, indent=2))
    result = run_targeted_dynamics()
    print(json.dumps({key: result[key] for key in (
        "trial_count", "static_forearm_pocket_entry", "coordinated_forearm_pocket_entry",
        "total_pocket_entry", "closest_pocket_distance_m", "ring_contact", "little_contact",
        "palm_root_contact", "sphere_loss", "corridor_clear", "maximum_penetration_by_surface_m"
    )}, indent=2))
