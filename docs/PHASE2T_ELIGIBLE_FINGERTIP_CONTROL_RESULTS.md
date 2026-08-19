# Phase 2T digit-eligible fingertip control results

**Stop code: `PHASE2T_NO_PAIR_SPECIFIC_B_CONTROL`**

## Motivation

Phase 2S established digit eligibility but could not isolate a conditional dynamic palmar effect. Phase 2T held digit count and identity fixed. Physics and all thresholds remained identical to Phase 2S.

## Endpoint search

```json
{
  "FINGERTIP": {
    "experiment_id": "phase2T_eligible_fingertip_vs_palmar",
    "status": "TARGET_REACHED",
    "attempts": 5896,
    "valid_states": 150,
    "attempted_by_support_pair": {
      "index+middle": 983,
      "index+ring": 983,
      "index+thumb": 983,
      "middle+ring": 983,
      "middle+thumb": 982,
      "ring+thumb": 982
    },
    "valid_by_occupied_pair": {
      "middle+ring": 2,
      "ring+thumb": 148
    },
    "valid_by_free_pair": {
      "index+middle": 148,
      "index+thumb": 2
    },
    "rejection_reasons": {
      "no_table_support": 4999,
      "translation": 559,
      "orientation": 87,
      "penetration": 101
    },
    "config_hash": "ba9ba78e71fc57d8cda9da2de579134dfe2c004475480aa81c7dd58ea0270ef0",
    "git_commit_sha": "b5a2f6e5aba9aa6547b2c138e747d245754c34b2"
  },
  "PALMAR_SECURED": {
    "experiment_id": "phase2T_eligible_fingertip_vs_palmar",
    "status": "TARGET_REACHED",
    "attempts": 1468,
    "valid_states": 151,
    "valid_by_occupied_pair": {
      "ring+thumb": 151
    },
    "valid_by_free_pair": {
      "index+middle": 151
    },
    "rejection_reasons": {
      "force_closure": 143,
      "exactly_two_load_bearing_fingers": 801,
      "orientation": 106,
      "translation": 138,
      "penetration": 124,
      "maximum_load_bearing_fingers": 1,
      "no_complete_hand_contact_loss": 3,
      "occupied_pair_matches_target": 1
    },
    "config_hash": "29bca64f141499a1e53c1a41cdc36ea2c5ed99d8c56af9ba994500dbdcebb08d",
    "git_commit_sha": "b5a2f6e5aba9aa6547b2c138e747d245754c34b2"
  }
}
```

The only topology satisfying the 50-state endpoint-population requirement in both groups had occupied `ring+thumb` and free `index+middle`. Both groups therefore had exactly two occupied and two free fingers with identical free-finger identity.

## Mandatory pair-specific B-only control

Only index and middle were commanded/permitted as acquisition digits. Phase 2S regions and successful trajectories were tried first, followed by deterministic bounded trajectory/pose refinement. The authorized cap was exhausted: 0 strict successes in 4,096 candidates.

```json
{
  "status": "PHASE2T_NO_PAIR_SPECIFIC_B_CONTROL",
  "free_finger_topology": [
    "index",
    "middle"
  ],
  "candidate_count": 4096,
  "strict_success_count": 0,
  "target_success_count": 3,
  "source_profile_successes": {},
  "failure_mechanisms": {
    "NO_B_CONTACT_BEFORE_RELEASE": 1272,
    "B_SLIPPED_TO_TABLE": 1801,
    "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE": 414,
    "INITIAL_INVALID_CONTACT": 174,
    "SINGLE_UNOPPOSED_CONTACT": 144,
    "CONTACT_FORCE_TOO_LOW": 6,
    "B_ROTATED_OUT": 217,
    "OTHER": 67,
    "null": 1
  },
  "best_successes": [],
  "config_hash": "7cc403cfcff6d5a4f79f08423a487955a54404891b92ab9f071f8260cab302b5",
  "git_commit_sha": "b5a2f6e5aba9aa6547b2c138e747d245754c34b2"
}
```

## Resource descriptors under equal digit count

```json
{
  "free_finger_workspace_vol_m3": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.00027537500000000007,
      "maximum": 0.00032262500000000005,
      "mean": 0.000290191722972973,
      "standard_deviation": 6.94408571612837e-06,
      "median": 0.0002893125000000001
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.000135875,
      "maximum": 0.00035037500000000005,
      "mean": 0.00021496357615894042,
      "standard_deviation": 6.820502423667797e-05,
      "median": 0.00017937500000000004
    },
    "mean_difference_palmar_minus_fingertip": -7.52281468140326e-05,
    "standardized_mean_difference": -1.5518141328870991
  },
  "free_palm_volume_m3": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.0033295000000000004,
      "maximum": 0.0033327500000000006,
      "mean": 0.003331269425675676,
      "standard_deviation": 6.009333793450886e-07,
      "median": 0.0033312500000000004
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.0033272500000000003,
      "maximum": 0.0033338750000000005,
      "mean": 0.003331099337748345,
      "standard_deviation": 1.1565319819034952e-06,
      "median": 0.0033311250000000008
    },
    "mean_difference_palmar_minus_fingertip": -1.7008792733107236e-07,
    "standardized_mean_difference": -0.1845575370411021
  },
  "COM_to_palm_origin_distance_m": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.06562216085841603,
      "maximum": 0.07057226225455089,
      "mean": 0.06833660049052977,
      "standard_deviation": 0.0013118197103186785,
      "median": 0.0681192363207583
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.029006305775299304,
      "maximum": 0.03198169296939009,
      "mean": 0.03039397799066982,
      "standard_deviation": 0.00045165435487023233,
      "median": 0.03039351091699753
    },
    "mean_difference_palmar_minus_fingertip": -0.03794262249985995,
    "standardized_mean_difference": -38.67608220743378
  },
  "palm_A_contact_fraction": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.0,
      "maximum": 0.0,
      "mean": 0.0,
      "standard_deviation": 0.0,
      "median": 0.0
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.98,
      "maximum": 1.0,
      "mean": 0.9974039735099338,
      "standard_deviation": 0.003715156000218603,
      "median": 1.0
    },
    "mean_difference_palmar_minus_fingertip": 0.9974039735099338,
    "standardized_mean_difference": 379.6724084855565
  },
  "ferrari_canny_epsilon": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.04876162484780513,
      "maximum": 0.18502567006187026,
      "mean": 0.14680275258301226,
      "standard_deviation": 0.023808746672177437,
      "median": 0.1488315662502332
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.0003811358271476051,
      "maximum": 0.2531812596102272,
      "mean": 0.12415835311709718,
      "standard_deviation": 0.0756372037801955,
      "median": 0.11713511725855599
    },
    "mean_difference_palmar_minus_fingertip": -0.02264439946591508,
    "standardized_mean_difference": -0.4038545314775473
  },
  "total_A_normal_force_N": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 2.8608639073905406,
      "maximum": 5.402900300585179,
      "mean": 3.596344704834796,
      "standard_deviation": 0.37202417950070715,
      "median": 3.5299240490015285
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 1.0535162521441346,
      "maximum": 10.023490565879884,
      "mean": 2.3366243927274533,
      "standard_deviation": 1.3090073837601157,
      "median": 1.9344785623335863
    },
    "mean_difference_palmar_minus_fingertip": -1.2597203121073428,
    "standardized_mean_difference": -1.3091219115324846
  },
  "A_translation_drift_m": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.0013345580823815085,
      "maximum": 0.0030618499689578933,
      "mean": 0.0020810384072996067,
      "standard_deviation": 0.0003603151236820828,
      "median": 0.0020544025644861565
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.00017543301639339867,
      "maximum": 0.0035929845294704354,
      "mean": 0.0014833465800870237,
      "standard_deviation": 0.0008729594285000134,
      "median": 0.00152276263295171
    },
    "mean_difference_palmar_minus_fingertip": -0.000597691827212583,
    "standardized_mean_difference": -0.8950305271363871
  },
  "A_rotation_drift_rad": {
    "FINGERTIP": {
      "count": 148,
      "minimum": 0.059805744005694904,
      "maximum": 0.19883521542124377,
      "mean": 0.1470436187491666,
      "standard_deviation": 0.02765134211169131,
      "median": 0.14795162147885574
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": 0.006389378911005214,
      "maximum": 0.1996918343271846,
      "mean": 0.07566033159872888,
      "standard_deviation": 0.05152719966162678,
      "median": 0.06557068030778043
    },
    "mean_difference_palmar_minus_fingertip": -0.07138328715043772,
    "standardized_mean_difference": -1.7263178992718102
  },
  "minimum_joint_margin_rad": {
    "FINGERTIP": {
      "count": 148,
      "minimum": -0.05741125530264024,
      "maximum": 0.04565857200942891,
      "mean": -0.013422404038095383,
      "standard_deviation": 0.024170667713707964,
      "median": -0.015499192399399564
    },
    "PALMAR_SECURED": {
      "count": 151,
      "minimum": -0.003449775606846117,
      "maximum": 0.10216451296329221,
      "mean": 0.052987881888648065,
      "standard_deviation": 0.020775039758881068,
      "median": 0.05495809390034201
    },
    "mean_difference_palmar_minus_fingertip": 0.06641028592674345,
    "standardized_mean_difference": 2.9467360064510557
  }
}
```

## Formal inference

No B distribution or controller was frozen, no calibration A+B outcomes were run, no formal matching dataset was created, and no A+B formal outcomes were inspected. `INSUFFICIENT_FREE_DIGITS_PRECHECK` is absent from the constructed endpoint populations, but the dynamic experiment is unidentifiable because its mandatory positive control failed.

## Interpretation

Case T4. Eligible FINGERTIP states exist, but the pair-specific second-acquisition control failed. No palmar-versus-fingertip dynamic effect is estimated.

## Limitations

The failure is specific to the tested half-scale B, index+middle topology, existing Phase 2S proposal regions, and bounded 4,096-candidate search. It does not prove all possible two-digit B acquisition is impossible. No scalar J, transfer, gaiting, wrist control, third object, reward change, or RL was introduced.
