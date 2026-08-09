# Phase 2 resource-component correlation results

> Status: complete. No scalar J is defined or evaluated.

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
    "config_hash": "a44d3cf6d12815391c78ce9ecbebaa5c80b76e24686196483e8a9d47af6f9abe",
    "git_commit_sha": "8c8116fac57d7f2982c8c23ca3ec7c5113586883",
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
  "accepted": 227,
  "original_candidate_attempts": 4355,
  "extension_candidate_attempts": 320,
  "total_candidate_attempts": 4675,
  "commanded_subset_distribution": {
    "index+middle+ring+thumb": 183,
    "middle+ring+thumb": 35,
    "index+middle+thumb": 2,
    "index+ring+thumb": 7
  },
  "occupied_finger_count_distribution": {
    "4": 59,
    "3": 138,
    "2": 30
  },
  "ferrari_canny_epsilon": {
    "min": 0.0018724284671326823,
    "max": 0.12441705104617955,
    "mean": 0.05739012098940495,
    "std": 0.03162569912916755
  }
}
```

Resource records: 227. Raw component and convergence summary:

```json
{
  "records": 227,
  "occupied_finger_count_distribution": {
    "4": 59,
    "3": 138,
    "2": 30
  },
  "components": {
    "occupied_finger_count": {
      "min": 2.0,
      "max": 4.0,
      "mean": 3.1277533039647576,
      "std": 0.6129841579580879
    },
    "free_finger_workspace_vol_m3": {
      "min": 0.0,
      "max": 0.00028275000000000007,
      "mean": 0.00011175881057268724,
      "std": 7.493813843416004e-05
    },
    "free_palm_volume_m3": {
      "min": 0.0031523750000000007,
      "max": 0.0032035000000000006,
      "mean": 0.0031641426211453747,
      "std": 1.304612467404225e-05
    }
  },
  "workspace_convergence_records": 45,
  "workspace_convergence": {
    "1000": {
      "representative_grasps": 9,
      "mean_volume_m3": 2.5777777777777785e-05,
      "mean_relative_change_from_previous_budget": null,
      "maximum_absolute_relative_change_from_previous_budget": null
    },
    "2500": {
      "representative_grasps": 9,
      "mean_volume_m3": 5.111111111111112e-05,
      "mean_relative_change_from_previous_budget": 0.9549248511042757,
      "maximum_absolute_relative_change_from_previous_budget": 1.1213592233009706
    },
    "5000": {
      "representative_grasps": 9,
      "mean_volume_m3": 7.729166666666667e-05,
      "mean_relative_change_from_previous_budget": 0.49295032106913156,
      "maximum_absolute_relative_change_from_previous_budget": 0.63882618510158
    },
    "10000": {
      "representative_grasps": 9,
      "mean_volume_m3": 0.0001047638888888889,
      "mean_relative_change_from_previous_budget": 0.3355368533399217,
      "maximum_absolute_relative_change_from_previous_budget": 0.41184573002754815
    },
    "20000": {
      "representative_grasps": 9,
      "mean_volume_m3": 0.0001287638888888889,
      "mean_relative_change_from_previous_budget": 0.2162630544160167,
      "maximum_absolute_relative_change_from_previous_budget": 0.28975609756097576
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

The earlier table-resting distribution was legal but geometrically unreachable (0/100) for the fixed-base hand, so Part D correctly stopped before outcomes. `docs/PHASE2_B_WORKSPACE_CALIBRATION.md` preserves that negative result and the outcome-free redesign audit.

The frozen fixture-presented B centre distribution is x=[0.055, 0.065] m, y=[0.115, 0.125] m, z=[0.215, 0.225] m, vertical axis, and uniform yaw=[0.0, 6.283185307179586] rad. Fixture release is timestep 900; all final-hold steps are unsupported. Geometry preflight:

```json
{
  "placements": 200,
  "representative_grasps": 20,
  "representative_grasp_ids": [
    "phase2_grasp_0006",
    "phase2_grasp_0211",
    "phase2_grasp_0226",
    "phase2_grasp_0001",
    "phase2_grasp_0103",
    "phase2_grasp_0199",
    "phase2_grasp_0000",
    "phase2_grasp_0091",
    "phase2_grasp_0198",
    "phase2_grasp_0090",
    "phase2_grasp_0190",
    "phase2_grasp_0067",
    "phase2_grasp_0171",
    "phase2_grasp_0044",
    "phase2_grasp_0123",
    "phase2_grasp_0078",
    "phase2_grasp_0201",
    "phase2_grasp_0082",
    "phase2_grasp_0151",
    "phase2_grasp_0014"
  ],
  "reachable_placements_any_representative": 195,
  "unreachable_placements_all_representatives": 5,
  "reachable_grasp_pose_pairs": 1439,
  "total_grasp_pose_pairs": 4000,
  "reachable_pair_fraction": 0.35975,
  "minimum_distance_m": {
    "min": 0.0023712195565306834,
    "median": 0.015242279404144651,
    "max": 0.021619540971775656
  },
  "reachable_free_finger_count_max": 1,
  "invalid_overlap_pair_fraction": 0.001,
  "status": "PASS"
}
```

The engineering pilot contributed 60 records marked `pilot_only: true`; all are excluded here. Completed 4540 / 4540 formal non-pilot trials. Outcome counts: `{"BOTH_LOST": 1834, "B_NOT_ACQUIRED": 2700, "INVALID": 6}`. BOTH_RETAINED rate among valid trials: `0.0`.

Binned intervals are 95% Wilson binomial intervals. Continuous components use five equal-frequency bins when enough distinct values exist; occupied count uses integer categories. INVALID is reported separately and excluded from the primary logistic model.

```json
{
  "dataset_dir": "outputs/phase2/grasp_dataset/fcc7835446a4",
  "planned_trials": 4540,
  "completed_trials": 4540,
  "valid_trials": 4534,
  "invalid_trials": 6,
  "pilot_trials_present_and_excluded": 60,
  "outcome_counts": {
    "B_NOT_ACQUIRED": 2700,
    "BOTH_LOST": 1834,
    "INVALID": 6
  },
  "BOTH_RETAINED_rate_valid": 0.0,
  "success_bins_with_95_percent_Wilson_intervals": {
    "occupied_finger_count": [
      {
        "category": "2",
        "x_mean": 2.0,
        "successes": 0,
        "count": 600,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.00636170101370071
      },
      {
        "category": "3",
        "x_mean": 3.0,
        "successes": 0,
        "count": 2754,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0013929222828990345
      },
      {
        "category": "4",
        "x_mean": 4.0,
        "successes": 0,
        "count": 1180,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0032449098585556092
      }
    ],
    "free_finger_workspace_vol_m3": [
      {
        "category": "Q1",
        "x_mean": 0.0,
        "successes": 0,
        "count": 920,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0041581364248340184
      },
      {
        "category": "Q2",
        "x_mean": 8.445721476510068e-05,
        "successes": 0,
        "count": 894,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004278549161385177
      },
      {
        "category": "Q3",
        "x_mean": 0.00013374444444444445,
        "successes": 0,
        "count": 900,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004250146730054126
      },
      {
        "category": "Q4",
        "x_mean": 0.00013835000000000003,
        "successes": 0,
        "count": 900,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004250146730054126
      },
      {
        "category": "Q5",
        "x_mean": 0.00020273641304347832,
        "successes": 0,
        "count": 920,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0041581364248340184
      }
    ],
    "free_palm_volume_m3": [
      {
        "category": "Q1",
        "x_mean": 0.0031550597826086957,
        "successes": 0,
        "count": 920,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0041581364248340184
      },
      {
        "category": "Q2",
        "x_mean": 0.003156944444444445,
        "successes": 0,
        "count": 900,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004250146730054126
      },
      {
        "category": "Q3",
        "x_mean": 0.003158247222222223,
        "successes": 0,
        "count": 900,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004250146730054126
      },
      {
        "category": "Q4",
        "x_mean": 0.0031607265100671144,
        "successes": 0,
        "count": 894,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.004278549161385177
      },
      {
        "category": "Q5",
        "x_mean": 0.0031893532608695658,
        "successes": 0,
        "count": 920,
        "rate": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0041581364248340184
      }
    ]
  },
  "logistic_regression": {
    "status": "NOT_IDENTIFIABLE",
    "reason": "requires >=10 valid trials and both outcome classes"
  },
  "failure_modes": {
    "occupied_finger_count": {
      "2": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.4633333333333333,
        "BOTH_LOST": 0.5366666666666666,
        "INVALID": 0.0
      },
      "3": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.45,
        "BOTH_LOST": 0.5478260869565217,
        "INVALID": 0.002173913043478261
      },
      "4": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 1.0,
        "BOTH_LOST": 0.0,
        "INVALID": 0.0
      }
    },
    "free_finger_workspace_vol_m3": {
      "(-0.001, 0.000134]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.7271929824561404,
        "BOTH_LOST": 0.27017543859649124,
        "INVALID": 0.002631578947368421
      },
      "(0.000134, 0.00014]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.47719298245614034,
        "BOTH_LOST": 0.5228070175438596,
        "INVALID": 0.0
      },
      "(0.00014, 0.000283]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.4446428571428571,
        "BOTH_LOST": 0.5553571428571429,
        "INVALID": 0.0
      }
    },
    "free_palm_volume_m3": {
      "(0.003052, 0.003157]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.5796610169491525,
        "BOTH_LOST": 0.42033898305084744,
        "INVALID": 0.0
      },
      "(0.003157, 0.003158]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.6213114754098361,
        "BOTH_LOST": 0.37868852459016394,
        "INVALID": 0.0
      },
      "(0.003158, 0.003161]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.5333333333333333,
        "BOTH_LOST": 0.4666666666666667,
        "INVALID": 0.0
      },
      "(0.003161, 0.003204]": {
        "BOTH_RETAINED": 0.0,
        "A_DROPPED": 0.0,
        "B_NOT_ACQUIRED": 0.6375,
        "BOTH_LOST": 0.35714285714285715,
        "INVALID": 0.005357142857142857
      }
    }
  },
  "failure_trigger_counts": {
    "B_not_acquired": 2636,
    "A_retention_bound": 1898,
    "initial_overlap:8": 5,
    "initial_overlap:10": 1
  },
  "failure_phase_counts": {
    "final_hold": 2961,
    "approach": 1263,
    "close": 316
  },
  "B_criterion_failure_counts_valid_trials": {
    "no_final_free_finger_contact": 4534,
    "no_final_hand_support": 4534,
    "force_not_above_0.20_N": 4534,
    "table_contact": 4534,
    "complete_hand_contact_loss": 4534
  },
  "greedy_Ferrari_Canny_baseline": {
    "top_decile_epsilon_cutoff": 0.10193945484662907,
    "top_decile_grasps": 23,
    "top_decile_BOTH_RETAINED_rate": 0.0,
    "full_population_rate": 0.0,
    "remaining_90_percent_rate": 0.0
  },
  "grasp_dataset_statistics": {
    "accepted": 227,
    "original_candidate_attempts": 4355,
    "extension_candidate_attempts": 320,
    "total_candidate_attempts": 4675,
    "commanded_subset_distribution": {
      "index+middle+ring+thumb": 183,
      "middle+ring+thumb": 35,
      "index+middle+thumb": 2,
      "index+ring+thumb": 7
    },
    "occupied_finger_count_distribution": {
      "4": 59,
      "3": 138,
      "2": 30
    },
    "ferrari_canny_epsilon": {
      "min": 0.0018724284671326823,
      "max": 0.12441705104617955,
      "mean": 0.05739012098940495,
      "std": 0.03162569912916755
    }
  },
  "dataset_extension": {
    "status": "TARGET_REACHED",
    "original_grasps_preserved": 200,
    "final_accepted_grasps": 227,
    "additional_candidate_attempts": 320,
    "occupied_finger_count_distribution": {
      "4": 59,
      "3": 138,
      "2": 30
    },
    "attempted_sampling_mode_distribution": {
      "targeted_occupied2_anchor": 8,
      "targeted_two_finger_command": 156,
      "targeted_occupied2_neighborhood": 156
    },
    "accepted_extension_sampling_mode_distribution": {
      "targeted_occupied2_neighborhood": 27
    },
    "attempted_commanded_subset_distribution": {
      "ring+thumb": 55,
      "middle+thumb": 55,
      "index+thumb": 54,
      "index+middle+ring+thumb": 52,
      "index+ring+thumb": 52,
      "middle+ring+thumb": 52
    },
    "acceptance_thresholds_relaxed": false,
    "resumable_attempt_store": "outputs/phase2/grasp_dataset/fcc7835446a4/extension_candidate_attempts.jsonl"
  },
  "B_geometry_preflight": {
    "placements": 200,
    "representative_grasps": 20,
    "representative_grasp_ids": [
      "phase2_grasp_0006",
      "phase2_grasp_0211",
      "phase2_grasp_0226",
      "phase2_grasp_0001",
      "phase2_grasp_0103",
      "phase2_grasp_0199",
      "phase2_grasp_0000",
      "phase2_grasp_0091",
      "phase2_grasp_0198",
      "phase2_grasp_0090",
      "phase2_grasp_0190",
      "phase2_grasp_0067",
      "phase2_grasp_0171",
      "phase2_grasp_0044",
      "phase2_grasp_0123",
      "phase2_grasp_0078",
      "phase2_grasp_0201",
      "phase2_grasp_0082",
      "phase2_grasp_0151",
      "phase2_grasp_0014"
    ],
    "reachable_placements_any_representative": 195,
    "unreachable_placements_all_representatives": 5,
    "reachable_grasp_pose_pairs": 1439,
    "total_grasp_pose_pairs": 4000,
    "reachable_pair_fraction": 0.35975,
    "minimum_distance_m": {
      "min": 0.0023712195565306834,
      "median": 0.015242279404144651,
      "max": 0.021619540971775656
    },
    "reachable_free_finger_count_max": 1,
    "invalid_overlap_pair_fraction": 0.001,
    "status": "PASS"
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
      "config_hash": "a44d3cf6d12815391c78ce9ecbebaa5c80b76e24686196483e8a9d47af6f9abe",
      "git_commit_sha": "8c8116fac57d7f2982c8c23ca3ec7c5113586883",
      "fixed_base_interpretation": "palm fixed; lift replayed by removing external object support",
      "profile_path": "configs/grasps/resource_grasp_A_02.yaml",
      "contact_override": null
    }
  },
  "figure_paths": [
    "docs/figures/phase2/B_geometry_preflight_panel.pdf",
    "docs/figures/phase2/contact_parameter_sensitivity.pdf",
    "docs/figures/phase2/free_finger_workspace_vs_success.pdf",
    "docs/figures/phase2/free_palm_measurement_box.pdf",
    "docs/figures/phase2/free_palm_volume_vs_success.pdf",
    "docs/figures/phase2/occupied_fingers_vs_success.pdf",
    "docs/figures/phase2/outcomes_by_resource_component.pdf",
    "docs/figures/phase2/phase2_correlation_summary.pdf",
    "docs/figures/phase2/representative_actual_grasps.pdf",
    "docs/figures/phase2/resource_component_histograms.pdf",
    "docs/figures/phase2/workspace_convergence.pdf"
  ],
  "representative_render_status": "docs/figures/phase2/representative_actual_grasps.pdf",
  "scalar_J": null,
  "resource_component_associations": {
    "occupied_finger_count": "weak/no detectable association",
    "free_finger_workspace_vol_m3": "weak/no detectable association",
    "free_palm_volume_m3": "weak/no detectable association"
  }
}
```

## Interpretation and limitations

These analyses test association between raw resource components and sequential acquisition outcomes. They do not establish causality and do not claim that J predicts success; J remains PI-blocked. Incomplete batches must not be interpreted as final estimates.

The prescribed 10,000-sample workspace budget was retained even though the representative convergence study still changed materially between 5,000, 10,000, and 20,000 samples. This limitation is reported rather than used to tune the production budget.

## Remaining PI decisions

Every active scientific placeholder is enumerated with file and line in `docs/PI_DECISIONS.md`. Scalar J, general task transition/drop criteria, reward design, and closed-loop retention remain unresolved.

## Reproducibility

Phase-2 config: `configs/phase2_physics_validation.yaml`. Dataset/config hashes and source git SHAs are stored in every incremental JSONL record. Resume with `python scripts/build_grasp_dataset.py --workers 8`, then `python scripts/compute_resource_components.py --workers 8`, then `python scripts/run_correlation_experiment.py --workers 8`, and rerun this script.
