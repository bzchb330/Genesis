"""Rerun only the frozen Phase 3C-0.5 one-finger release matrix.

The capture condition and frozen cohort are identical to the formal run.  This
script deliberately does not rerun the serial/simultaneous/wrist search.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3c05 import (
    CaptureStrategy,
    capture_trial,
    load_frozen_states,
    load_phase3c05_config,
    release_trial,
)


def main() -> None:
    cfg = load_phase3c05_config()
    states = load_frozen_states()
    result_path = ROOT / "outputs/phase3C05/phase3c05_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stored = {
        row["state_id"]: row for row in result["capture_trials"]
        if row["strategy"] == CaptureStrategy.WRIST_LOAD_TRANSFER.value
    }
    if set(stored) != {state.state_id for state in states}:
        raise RuntimeError("stored load-transfer cohort differs from frozen matched states")

    scene = build_shadow_scene()
    series_dir = ROOT / "outputs/phase3C05/timeseries"
    release_rows: list[dict] = []
    for state in states:
        capture = capture_trial(
            scene,
            state,
            ("middle", "ring", "little"),
            CaptureStrategy.WRIST_LOAD_TRANSFER,
            wrist_delta_deg=(-5.0, -5.0),
        )
        prior = stored[state.state_id]
        if (
            capture["A_retained"] != prior["A_retained"]
            or capture["persistence_first_reached"] != prior["persistence_first_reached"]
        ):
            raise RuntimeError(f"deterministic capture replay mismatch for {state.state_id}")
        for finger in cfg["release"]["fingers"]:
            for ramp in cfg["release"]["ramp_steps"]:
                row = release_trial(scene, capture, str(finger), int(ramp))
                row.update({
                    "state_id": state.state_id,
                    "capture_strategy": CaptureStrategy.WRIST_LOAD_TRANSFER.value,
                    "subset": ["middle", "ring", "little"],
                    "wrist_delta_command_deg": [-5.0, -5.0],
                })
                if row.get("samples"):
                    samples = row.pop("samples")
                    path = series_dir / f"RELEASE_{finger}_{ramp}_{state.state_id}.npz"
                    np.savez_compressed(
                        path,
                        step=np.asarray([sample["step"] for sample in samples]),
                        normal_forces_n=np.asarray([sample["normal_forces_n"] for sample in samples]),
                        alternate_fraction=np.asarray([sample["alternate_fraction"] for sample in samples]),
                        floor_contact=np.asarray([sample["floor_contact"] for sample in samples], dtype=np.int8),
                        retained_A=np.asarray([sample["retained_A"] for sample in samples], dtype=np.int8),
                        released_finger_contact=np.asarray(
                            [sample["released_finger_contact"] for sample in samples], dtype=np.int8
                        ),
                        A_position_palm_m=np.asarray([sample["A_position_palm_m"] for sample in samples]),
                        wrist_qpos=np.asarray([sample["wrist_qpos"] for sample in samples]),
                    )
                    row["timeseries_path"] = str(path)
                release_rows.append(row)

    successful = [row for row in release_rows if row.get("one_resource_recovered")]
    optional = {"executed": False, "reason": "one-finger recovery not robust across multiple matched states"}
    if len({row["state_id"] for row in successful}) >= 2:
        optional = {
            "executed": False,
            "reason": "mechanism met robustness precondition; PI review required before optional expansion",
        }
    result["release_trials"] = release_rows
    result["optional_second_release"] = optional
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "capture_replays": len(states),
        "release_candidates": len(release_rows),
        "release_executed": sum(row["executed"] for row in release_rows),
        "one_resource_recovered": len(successful),
        "successful_states": sorted({row["state_id"] for row in successful}),
        "optional_second_release": optional,
    }, indent=2))


if __name__ == "__main__":
    main()
