from __future__ import annotations

import json

from seqgrasp.phase3c08 import run_kinematic_audit


if __name__ == "__main__":
    result = run_kinematic_audit()
    reachable = result["reachable_gravity_audit"]
    print(json.dumps({
        "target_direction": {
            key: result["target_direction_audit"][key]
            for key in ("mean_direction", "median_direction", "angular_spread_deg", "mean_vs_previous_angle_deg")
        },
        "axis": result["forearm_axis_audit"],
        "zero_angle": result["zero_angle_backward_compatibility"],
        "native": {key: reachable["native"][key] for key in (
            "coarse_set_size", "minimum_residual_deg", "median_residual_deg", "p95_residual_deg", "maximum_projection"
        )},
        "augmented": {key: reachable["augmented"][key] for key in (
            "coarse_set_size", "minimum_residual_deg", "median_residual_deg", "p95_residual_deg", "maximum_projection",
            "fraction_below_10_deg", "fraction_below_15_deg", "fraction_below_20_deg"
        )},
        "classification": reachable["classification"],
        "targeted_dynamics_authorized": reachable["targeted_dynamics_authorized"],
    }, indent=2))
