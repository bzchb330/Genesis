# Phase 2 resource-component correlation results

> Status: PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED. No scalar J is defined or evaluated.

## Part A physics gate and sensitivity

The original baseline passed the PI-supplied hard gate and was retained a priori. The 81-condition sweep is a sensitivity study only; it did not select production physics. Baseline measurements and checks:

```json
{
  "verdict": "PASS",
  "completed_hold_steps": 1000,
  "required_hold_steps": 1000,
  "force_sanity_range_N": [
    0.1,
    10.0
  ],
  "measurements": {
    "maximum_penetration_m": 0.0019782649181727847,
    "maximum_vertical_drift_m": 0.0015184268502597409,
    "final_vertical_drift_m": 0.0008827873145980936,
    "maximum_translational_drift_m": 0.004242062748539936,
    "final_translational_drift_m": 0.004242062748539936,
    "maximum_orientation_drift_rad": 0.15820717826265318,
    "final_orientation_drift_rad": 0.15820717826265318,
    "mean_force_per_finger_N": [
      1.135670659135801,
      0.3664229696451583,
      1.4345363994137161,
      0.9156407267201483
    ],
    "final_force_per_finger_N": [
      0.7800779752621377,
      0.3063616067793121,
      1.3720341365411626,
      0.9176975081757204
    ],
    "mean_total_normal_force_N": 3.852270754914824,
    "final_total_normal_force_N": 3.3761712267583333,
    "total_normal_force_std_N": 0.4495404647104166,
    "minimum_active_object_contacts": 4,
    "maximum_active_object_contacts": 5,
    "table_recontact": false,
    "complete_object_hand_contact_loss": false,
    "numerical_validity": true
  },
  "checks": {
    "force_order_of_magnitude": true,
    "penetration": true,
    "vertical_drift": true,
    "translational_drift": true,
    "orientation_drift": true,
    "active_contacts": true,
    "table_recontact": true,
    "complete_contact_loss": true,
    "numerical_validity": true
  },
  "missing_pi_inputs": [],
  "metadata": {
    "seed": 0,
    "config_hash": "94e49677c3f81fa775213eebae580abe8f92469d6efbcced8267f62c4081347c",
    "git_commit_sha": "137a1fb55d35aa21b5fb488b831168a1432ca1ac",
    "fixed_base_interpretation": "palm fixed; lift replayed by removing external object support",
    "profile_path": "configs/grasps/resource_grasp_A_02.yaml",
    "contact_override": null
  }
}
```

# Phase 2 Contact-Parameter Sweep

The sweep is diagnostic and does not automatically select a physics configuration.

Status: **COMPLETE_BASELINE_RETAINED**

PASS: **36 / 81**; FAIL: **45 / 81**.

The PI specified a priori that the original baseline remains production physics when its hard gate passes. It passed, so no sweep row was selected and the original 0.002 s baseline mechanics remain unchanged.

Vector sensitivity figure: `docs/figures/phase2/contact_parameter_sensitivity.pdf`.

| Rank | Parameters | Gate | Force [N] | Penetration [m] | Translation drift [m] | Numerical |
|---:|---|---|---:|---:|---:|---|
| 1 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.5348 | 0.00024746 | 0.00470102 | True |
| 2 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.0892 | 0.000256648 | 0.00427073 | True |
| 3 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 4.04287 | 0.000278164 | 0.0045671 | True |
| 4 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.5572 | 0.000283458 | 0.0048769 | True |
| 5 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.60563 | 0.000284193 | 0.0048162 | True |
| 6 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 4.45506 | 0.000340738 | 0.00485941 | True |
| 7 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.5088 | 0.000395577 | 0.00438564 | True |
| 8 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.54377 | 0.000427563 | 0.00349355 | True |
| 9 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.49507 | 0.000468222 | 0.00475981 | True |
| 10 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | PASS | 3.21109 | 0.000529829 | 0.00497932 | True |
| 11 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.45464 | 0.000552563 | 0.00486265 | True |
| 12 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.83549 | 0.000569369 | 0.0047004 | True |
| 13 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.08576 | 0.000581331 | 0.00447108 | True |
| 14 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.79998 | 0.000593009 | 0.00477721 | True |
| 15 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.46519 | 0.000594911 | 0.00463016 | True |
| 16 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.93797 | 0.000634884 | 0.00443626 | True |
| 17 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.38796 | 0.000672348 | 0.00488536 | True |
| 18 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 4.28961 | 0.000693151 | 0.00486802 | True |
| 19 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.38168 | 0.00105935 | 0.00461606 | True |
| 20 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.22602 | 0.00108018 | 0.00494457 | True |
| 21 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.78041 | 0.0011102 | 0.00437294 | True |
| 22 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.60368 | 0.00112014 | 0.0047493 | True |
| 23 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.33311 | 0.00113536 | 0.00438489 | True |
| 24 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.21066 | 0.00120785 | 0.00495323 | True |
| 25 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.92186 | 0.00123733 | 0.00497756 | True |
| 26 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.93019 | 0.00150389 | 0.00429555 | True |
| 27 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.24252 | 0.00156298 | 0.00472904 | True |
| 28 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.3601 | 0.00158333 | 0.00441191 | True |
| 29 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.7908 | 0.00172112 | 0.00388985 | True |
| 30 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.58536 | 0.00183894 | 0.00437433 | True |
| 31 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 4.17415 | 0.00192674 | 0.00389899 | True |
| 32 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.85227 | 0.00197826 | 0.00424206 | True |
| 33 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.17105 | 0.00277702 | 0.00384628 | True |
| 34 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 2.90002 | 0.00278199 | 0.00412853 | True |
| 35 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | PASS | 3.50005 | 0.00285664 | 0.00361609 | True |
| 36 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | PASS | 3.19328 | 0.00292028 | 0.00457686 | True |
| 37 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.47302 | 0.000410002 | 0.00542243 | True |
| 38 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.28219 | 0.000533928 | 0.00563423 | True |
| 39 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.66152 | 0.000602135 | 0.0052647 | True |
| 40 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 4.02245 | 0.000605736 | 0.0052309 | True |
| 41 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.51487 | 0.000646654 | 0.00552589 | True |
| 42 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 4.08699 | 0.000654343 | 0.00514979 | True |
| 43 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.94128 | 0.000664048 | 0.00594537 | True |
| 44 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.73133 | 0.000751958 | 0.00573878 | True |
| 45 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 4.1228 | 0.000903811 | 0.00481163 | True |
| 46 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 3.29496 | 0.000993233 | 0.00511996 | True |
| 47 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.18768 | 0.00105695 | 0.00591456 | True |
| 48 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.38817 | 0.00106824 | 0.00516136 | True |
| 49 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.01, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.623 | 0.00121343 | 0.00553796 | True |
| 50 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 3.71762 | 0.00122823 | 0.00538858 | True |
| 51 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.45437 | 0.00128245 | 0.00772378 | True |
| 52 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.11355 | 0.00153642 | 0.00500483 | True |
| 53 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.33693 | 0.00160536 | 0.00545181 | True |
| 54 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 4.2625 | 0.00191284 | 0.00504583 | True |
| 55 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.42205 | 0.00204846 | 0.00537646 | True |
| 56 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 3.5781 | 0.00208703 | 0.00612321 | True |
| 57 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 0.258047 | 0.00217517 | 0.125076 | True |
| 58 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 3.7966 | 0.00220884 | 0.00645754 | True |
| 59 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.88209 | 0.00272581 | 0.00637456 | True |
| 60 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.72003 | 0.00293873 | 0.00661876 | True |
| 61 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 2.81504 | 0.00301137 | 0.00496616 | True |
| 62 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 3.00956 | 0.00301486 | 0.00431822 | True |
| 63 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.02, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.86526 | 0.00330865 | 0.0061704 | True |
| 64 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 2.09634 | 0.00369658 | 0.00270695 | True |
| 65 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 2.15096 | 0.00376945 | 0.00308644 | True |
| 66 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 2.20836 | 0.00416506 | 0.00412189 | True |
| 67 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 2.20408 | 0.00432716 | 0.00360245 | True |
| 68 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.43477 | 0.00518159 | 0.00877475 | True |
| 69 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.28433 | 0.00520523 | 0.00960149 | True |
| 70 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 2.31451 | 0.00529744 | 0.010422 | True |
| 71 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 2.54481 | 0.00533345 | 0.00525157 | True |
| 72 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 2.43452 | 0.00556744 | 0.00733721 | True |
| 73 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 1.64626 | 0.00696036 | 0.00425022 | True |
| 74 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 1.72716 | 0.0077814 | 0.0071151 | True |
| 75 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 1.72665 | 0.00782132 | 0.00761446 | True |
| 76 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 2.34977 | 0.00823896 | 0.00507523 | True |
| 77 | `{"friction": [0.6, 0.005, 0.0005], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 1.90789 | 0.00834475 | 0.00631854 | True |
| 78 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.001}` | FAIL | 1.92074 | 0.00867701 | 0.00509256 | True |
| 79 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.004}` | FAIL | 1.87679 | 0.00871478 | 0.0100651 | True |
| 80 | `{"friction": [0.8, 0.01, 0.001], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 1.87779 | 0.00883396 | 0.00629667 | True |
| 81 | `{"friction": [1.0, 0.02, 0.002], "solimp": [0.85, 0.9, 0.002, 0.5, 2.0], "solref": [0.04, 1.0], "target_geom_names": ["object_a_geom", "ff_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"], "timestep_s": 0.002}` | FAIL | 1.91826 | 0.00943173 | 0.00786475 | True |

Rows are ordered for inspection by numerical validity, configured gate state, penetration, and drift. This ordering is not a final physics selection.

## Descriptive sensitivity

Between-level spans compare the mean measurement at each configured level. They do not select a configuration.

| Parameter | Force span [N] | Penetration span [m] | Translation span [m] | Rotation span [rad] |
|---|---:|---:|---:|---:|
| friction | 0.638543 | 0.000380011 | 0.00438209 | 0.0496738 |
| solref | 1.31157 | 0.00416781 | 0.00573574 | 0.0556462 |
| solimp | 0.813197 | 0.00325464 | 0.00430353 | 0.0340326 |
| timestep_s | 0.365458 | 0.000181955 | 0.00640206 | 0.0654996 |

Largest descriptive spans: mean_total_normal_force_N -> solref, maximum_penetration_m -> solref, maximum_translational_drift_m -> timestep_s, maximum_orientation_drift_rad -> timestep_s.


## Parts B, C, and E

Accepted-grasp generation statistics:

```json
{
  "accepted": 200,
  "candidate_attempts": 4355,
  "commanded_subset_distribution": {
    "index+middle+ring+thumb": 160,
    "middle+ring+thumb": 35,
    "index+middle+thumb": 2,
    "index+ring+thumb": 3
  },
  "occupied_finger_count_distribution": {
    "4": 59,
    "3": 138,
    "2": 3
  },
  "ferrari_canny_epsilon": {
    "min": 0.0018724284671326823,
    "max": 0.12441705104617955,
    "mean": 0.056921285203159255,
    "std": 0.030952294899398946
  }
}
```

Resource records: 200. Raw component and convergence summary:

```json
{
  "records": 200,
  "occupied_finger_count_distribution": {
    "4": 59,
    "3": 138,
    "2": 3
  },
  "components": {
    "occupied_finger_count": {
      "min": 2.0,
      "max": 4.0,
      "mean": 3.28,
      "std": 0.4812483766206386
    },
    "free_finger_workspace_vol_m3": {
      "min": 0.0,
      "max": 0.00027100000000000003,
      "mean": 9.69975e-05,
      "std": 6.569632190998215e-05
    },
    "free_palm_volume_m3": {
      "min": 0.0031523750000000007,
      "max": 0.0032035000000000006,
      "mean": 0.0031642931250000007,
      "std": 1.3340156702954236e-05
    }
  },
  "workspace_convergence_records": 45,
  "workspace_convergence": {
    "1000": {
      "representative_grasps": 9,
      "mean_volume_m3": 2.9708333333333338e-05,
      "mean_relative_change_from_previous_budget": null,
      "maximum_absolute_relative_change_from_previous_budget": null
    },
    "2500": {
      "representative_grasps": 9,
      "mean_volume_m3": 5.852777777777779e-05,
      "mean_relative_change_from_previous_budget": 0.9389640561376879,
      "maximum_absolute_relative_change_from_previous_budget": 1.0807860262008735
    },
    "5000": {
      "representative_grasps": 9,
      "mean_volume_m3": 8.855555555555557e-05,
      "mean_relative_change_from_previous_budget": 0.48173108675675763,
      "maximum_absolute_relative_change_from_previous_budget": 0.5809935205183585
    },
    "10000": {
      "representative_grasps": 9,
      "mean_volume_m3": 0.00012054166666666668,
      "mean_relative_change_from_previous_budget": 0.3346770460279192,
      "maximum_absolute_relative_change_from_previous_budget": 0.4174496644295301
    },
    "20000": {
      "representative_grasps": 9,
      "mean_volume_m3": 0.0001500138888888889,
      "mean_relative_change_from_previous_budget": 0.2215342779063374,
      "maximum_absolute_relative_change_from_previous_budget": 0.3143939393939394
    }
  },
  "free_palm_method": "palm-frame voxel-centre occupancy against actual box/capsule collision geometry",
  "palm_axes": "PI reference [x,y,z] maps to compiled Allegro palm [-z,y,x]",
  "resource_method_id": "allegro_palm_axis_transform_v1",
  "free_palm_box_debug": {
    "fingertip_centres_inside": 4,
    "fingertip_centres_total": 4,
    "held_object_centre_inside": true,
    "reference_palm_frame_points_m": [
      [
        0.002371554695752896,
        0.04905702022227934,
        -0.08114130915377397
      ],
      [
        0.038988200599648935,
        0.01178916107360043,
        -0.08046989629718249
      ],
      [
        0.025351363980535238,
        -0.028750047513678954,
        -0.0809368011960932
      ],
      [
        -0.04380991013885853,
        0.07118986904251298,
        -0.0607165606109469
      ],
      [
        -0.02070293426791647,
        0.0005532201613618147,
        -0.06409795923465232
      ]
    ],
    "orientation": "PI reference [x,y,z] maps to compiled Allegro palm [-z,y,x]"
  },
  "scalar_J": null
}
```

Occupied fingers use summed A normal force >0.20 N. Free-finger workspace uses 10,000 Monte Carlo joint samples, actual MuJoCo collision geometry, and 0.005 m voxels. Free-palm volume uses the supplied palm-frame AABB and actual box/capsule collision geometry. The components are not combined.

The three unnormalised tactile features per finger are: binary contact (>0.05 N), total normal force [N], and tangential/normal force ratio. Ratio zero at zero normal force means no slip-proxy signal, not a physical loaded ratio of zero.

## Parts D and F

The B centre distribution is x=[-0.02, 0.05] m, y=[-0.07, 0.07] m, table-resting z from actual cylinder geometry, and uniform yaw=[0.0, 6.283185307179586] rad. Geometry preflight:

```json
{
  "placements": 100,
  "representative_grasps": 10,
  "valid_placements_any_representative": 100,
  "approachable_placements_any_representative": 0,
  "status": "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED"
}
```

Completed 0 / 4000 trials. Outcome counts: `{}`. BOTH_RETAINED rate among valid trials: `None`.

Binned intervals are 95% Wilson binomial intervals. Continuous components use five equal-frequency bins when enough distinct values exist; occupied count uses integer categories. INVALID is reported separately and excluded from the primary logistic model.

```json
{
  "dataset_dir": "outputs/phase2/grasp_dataset/fcc7835446a4",
  "planned_trials": 4000,
  "completed_trials": 0,
  "valid_trials": 0,
  "invalid_trials": 0,
  "outcome_counts": {},
  "BOTH_RETAINED_rate_valid": null,
  "success_bins_with_95_percent_Wilson_intervals": {},
  "logistic_regression": {
    "status": "NOT_IDENTIFIABLE",
    "reason": "no trials"
  },
  "failure_modes": {},
  "greedy_Ferrari_Canny_baseline": {},
  "grasp_dataset_statistics": {
    "accepted": 200,
    "candidate_attempts": 4355,
    "commanded_subset_distribution": {
      "index+middle+ring+thumb": 160,
      "middle+ring+thumb": 35,
      "index+middle+thumb": 2,
      "index+ring+thumb": 3
    },
    "occupied_finger_count_distribution": {
      "4": 59,
      "3": 138,
      "2": 3
    },
    "ferrari_canny_epsilon": {
      "min": 0.0018724284671326823,
      "max": 0.12441705104617955,
      "mean": 0.056921285203159255,
      "std": 0.030952294899398946
    }
  },
  "B_geometry_preflight": {
    "placements": 100,
    "representative_grasps": 10,
    "valid_placements_any_representative": 100,
    "approachable_placements_any_representative": 0,
    "status": "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED"
  },
  "baseline_physics": {
    "verdict": "PASS",
    "completed_hold_steps": 1000,
    "required_hold_steps": 1000,
    "force_sanity_range_N": [
      0.1,
      10.0
    ],
    "measurements": {
      "maximum_penetration_m": 0.0019782649181727847,
      "maximum_vertical_drift_m": 0.0015184268502597409,
      "final_vertical_drift_m": 0.0008827873145980936,
      "maximum_translational_drift_m": 0.004242062748539936,
      "final_translational_drift_m": 0.004242062748539936,
      "maximum_orientation_drift_rad": 0.15820717826265318,
      "final_orientation_drift_rad": 0.15820717826265318,
      "mean_force_per_finger_N": [
        1.135670659135801,
        0.3664229696451583,
        1.4345363994137161,
        0.9156407267201483
      ],
      "final_force_per_finger_N": [
        0.7800779752621377,
        0.3063616067793121,
        1.3720341365411626,
        0.9176975081757204
      ],
      "mean_total_normal_force_N": 3.852270754914824,
      "final_total_normal_force_N": 3.3761712267583333,
      "total_normal_force_std_N": 0.4495404647104166,
      "minimum_active_object_contacts": 4,
      "maximum_active_object_contacts": 5,
      "table_recontact": false,
      "complete_object_hand_contact_loss": false,
      "numerical_validity": true
    },
    "checks": {
      "force_order_of_magnitude": true,
      "penetration": true,
      "vertical_drift": true,
      "translational_drift": true,
      "orientation_drift": true,
      "active_contacts": true,
      "table_recontact": true,
      "complete_contact_loss": true,
      "numerical_validity": true
    },
    "missing_pi_inputs": [],
    "metadata": {
      "seed": 0,
      "config_hash": "94e49677c3f81fa775213eebae580abe8f92469d6efbcced8267f62c4081347c",
      "git_commit_sha": "137a1fb55d35aa21b5fb488b831168a1432ca1ac",
      "fixed_base_interpretation": "palm fixed; lift replayed by removing external object support",
      "profile_path": "configs/grasps/resource_grasp_A_02.yaml",
      "contact_override": null
    }
  },
  "figure_paths": [
    "docs/figures/phase2/B_geometry_preflight_panel.pdf",
    "docs/figures/phase2/contact_parameter_sensitivity.pdf",
    "docs/figures/phase2/free_palm_measurement_box.pdf",
    "docs/figures/phase2/representative_actual_grasps.pdf",
    "docs/figures/phase2/resource_component_histograms.pdf",
    "docs/figures/phase2/workspace_convergence.pdf"
  ],
  "representative_render_status": "docs/figures/phase2/representative_actual_grasps.pdf",
  "scalar_J": null
}
```

## Interpretation and limitations

These analyses test association between raw resource components and sequential acquisition outcomes. They do not establish causality and do not claim that J predicts success; J remains PI-blocked. Incomplete batches must not be interpreted as final estimates.

The prescribed 10,000-sample workspace budget was retained even though the representative convergence study still changed materially between 5,000, 10,000, and 20,000 samples. This limitation is reported rather than used to tune the production budget.

## Remaining PI decisions

Every active scientific placeholder is enumerated with file and line in `docs/PI_DECISIONS.md`. Scalar J, general task transition/drop criteria, reward design, and closed-loop retention remain unresolved.

## Reproducibility

Phase-2 config: `configs/phase2_physics_validation.yaml`. Dataset/config hashes and source git SHAs are stored in every incremental JSONL record. No correlation-batch resume command is authorized under the current config: the required geometry preflight returned `PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED`. PI input changing the B-placement geometry would be required before a new experiment namespace may be run.
