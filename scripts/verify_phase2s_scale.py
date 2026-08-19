#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.object_scale import compiled_object_geometry, vertical_half_extent
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import OBJECT_LINEAR_SCALE, load_phase2s_config
from seqgrasp.scene_builder import build_scene


def _scene(scene_filename: str) -> tuple[object, mujoco.MjModel, mujoco.MjData]:
    cfg = load_configs(scene_filename=scene_filename)
    model, data = build_scene(cfg)
    mujoco.mj_forward(model, data)
    return cfg, model, data


def main() -> int:
    phase2s, _ = load_phase2s_config()
    phase2, _ = load_phase2_config()
    large_cfg, large_model, _ = _scene("scene_two_object.yaml")
    small_cfg, small_model, _ = _scene(phase2s.scene_filename)
    table_top = float(small_cfg.scene.table_pos[2] + small_cfg.scene.table_size[2])
    objects = {}
    for name in ("object_a", "object_b"):
        large = compiled_object_geometry(large_model, name)
        small = compiled_object_geometry(small_model, name)
        ratio = np.asarray(small["physical_dimensions_m"]) / np.asarray(large["physical_dimensions_m"])
        if not np.array_equal(ratio, np.full(3, OBJECT_LINEAR_SCALE)):
            raise RuntimeError(f"PHASE2S_OBJECT_SCALE_VALIDATION_FAILED:{name}:{ratio.tolist()}")
        if large["mass_kg"] != 0.08 or small["mass_kg"] != 0.08:
            raise RuntimeError(f"Phase 2S mass changed for {name}")
        large_obj = next(row for row in large_cfg.scene.objects if row.name == name)
        small_obj = next(row for row in small_cfg.scene.objects if row.name == name)
        large_clearance = float(large_obj.pos[2] - vertical_half_extent(large_obj.shape, large_obj.size) - table_top)
        small_clearance = float(small_obj.pos[2] - vertical_half_extent(small_obj.shape, small_obj.size) - table_top)
        if not np.isclose(large_clearance, 0.001) or not np.isclose(small_clearance, 0.001):
            raise RuntimeError(f"default table clearance changed for {name}")
        objects[name] = {
            "large": {**large, "default_center_m": list(large_obj.pos), "table_clearance_m": large_clearance},
            "half_scale": {**small, "default_center_m": list(small_obj.pos), "table_clearance_m": small_clearance},
            "compiled_linear_dimension_ratio": ratio.tolist(),
        }
    characteristic = min(objects["object_a"]["half_scale"]["physical_dimensions_m"])
    thresholds = {
        "occupied_finger_force_threshold_N": {
            "value": phase2.resources.occupied_finger_normal_force_threshold_N,
            "normalized_by_min_dimension_N_per_m": phase2.resources.occupied_finger_normal_force_threshold_N / characteristic,
        },
        "tactile_binary_force_threshold_N": {
            "value": phase2.tactile.binary_contact_threshold_N,
            "normalized_by_min_dimension_N_per_m": phase2.tactile.binary_contact_threshold_N / characteristic,
        },
        "maximum_penetration_m": {
            "value": phase2.dataset.maximum_penetration_m,
            "fraction_of_min_dimension": phase2.dataset.maximum_penetration_m / characteristic,
        },
        "maximum_translation_drift_m": {
            "value": phase2.dataset.maximum_translation_drift_m,
            "fraction_of_min_dimension": phase2.dataset.maximum_translation_drift_m / characteristic,
        },
        "maximum_rotation_drift_rad": {"value": phase2.dataset.maximum_orientation_drift_rad},
    }
    result = {
        "object_linear_scale": OBJECT_LINEAR_SCALE,
        "yaml_size_is_passed_directly_to_MuJoCo_geom_size": True,
        "objects": objects,
        "unchanged_threshold_diagnostics": thresholds,
    }
    output = ROOT / phase2s.output_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ROOT / "docs" / "PHASE2S_GEOMETRY_VALIDATION.md"
    report.write_text(
        "# Phase 2S geometry validation\n\n"
        "`scene_builder.build_scene` writes each YAML object `size` directly to the MuJoCo geom `size` attribute. "
        "Compilation confirms standard box half-extent and cylinder [radius, half-height] semantics.\n\n"
        "```json\n" + json.dumps(result, indent=2) + "\n```\n\n"
        "Mass remains fixed, so the experiment isolates geometric scale and does not preserve density. "
        "The normalized-threshold table is diagnostic only; no scientific threshold was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
