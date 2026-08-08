import numpy as np
import mujoco

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.physics_validation import parameter_sensitivity, run_physics_validation
from seqgrasp.phase2_config import load_phase2_config, missing_contact_sweep_inputs
from seqgrasp.scene_builder import ContactParameterOverride, build_scene


def test_phase2_config_is_typed_and_preserves_pi_inputs_as_unresolved():
    phase2, path = load_phase2_config()
    assert path.name == "phase2_physics_validation.yaml"
    assert phase2.phase2_only is True
    assert phase2.validation.long_hold_steps == 1000
    assert phase2.validation.penetration_tolerance_m is None
    assert phase2.sweep.friction_vectors is None
    assert set(missing_contact_sweep_inputs(phase2.sweep)) == {
        "target_geom_names", "friction_vectors", "solref_values",
        "solimp_values", "timestep_values_s",
    }
    assert phase2.required_for_later_parts.occupied_finger_force_threshold_N is None
    assert phase2.required_for_later_parts.accepted_grasp_target == 200


def test_compiled_contact_parameter_override_uses_named_mujoco_fields():
    cfg = load_configs()
    override = ContactParameterOverride(
        geom_names=("object_a_geom",),
        friction=(0.7, 0.02, 0.003),
        solref=(0.03, 1.2),
        solimp=(0.8, 0.9, 0.002, 0.4, 2.0),
        timestep=0.001,
    )
    model, _ = build_scene(cfg, contact_override=override)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    np.testing.assert_allclose(model.geom_friction[geom], override.friction)
    np.testing.assert_allclose(model.geom_solref[geom], override.solref)
    np.testing.assert_allclose(model.geom_solimp[geom], override.solimp)
    assert model.opt.timestep == override.timestep


def test_physics_replay_is_deterministic_and_gate_waits_for_pi(tmp_path):
    phase2, config_path = load_phase2_config()
    first, first_summary = run_physics_validation(phase2, config_path, tmp_path / "first", write_plot=False)
    second, second_summary = run_physics_validation(phase2, config_path, tmp_path / "second", write_plot=False)
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], second.arrays[key])
    assert first_summary == second_summary
    assert first_summary["completed_hold_steps"] == 1000
    assert first_summary["checks"]["force_order_of_magnitude"] is True
    assert first_summary["measurements"]["numerical_validity"] is True
    assert first_summary["verdict"] == "PI_INPUT_REQUIRED"
    assert first_summary["missing_pi_inputs"]
    assert (tmp_path / "first" / "physics_validation_summary.json").is_file()
    assert (tmp_path / "first" / "physics_validation_timeseries.csv").is_file()


def test_sweep_sensitivity_reports_measurement_spans_without_selecting_configuration():
    rows = []
    for friction, force, penetration in (([0.5, 0.01, 0.001], 1.0, 0.002), ([1.0, 0.01, 0.001], 3.0, 0.001)):
        rows.append({
            "parameters": {"friction": friction, "solref": [0.02, 1.0], "solimp": [0.9] * 5, "timestep_s": 0.002},
            "summary": {"measurements": {
                "mean_total_normal_force_N": force,
                "maximum_penetration_m": penetration,
                "maximum_translational_drift_m": 0.003,
                "maximum_orientation_drift_rad": 0.1,
            }},
        })
    result = parameter_sensitivity(rows)
    assert result["friction"]["between_level_span"]["mean_total_normal_force_N"] == 2.0
    assert result["friction"]["between_level_span"]["maximum_penetration_m"] == 0.001
    assert result["timestep_s"]["between_level_span"]["mean_total_normal_force_N"] == 0.0
