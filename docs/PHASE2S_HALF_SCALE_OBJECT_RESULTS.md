# Phase 2S half-scale object results

## Scope and geometry revision

Phase 2S reduces every physical linear dimension of A and B to exactly 50% of the Phase 2R geometry while retaining each mass at 0.08 kg and retaining friction, gravity, timestep, contact parameters, actuation, gains, thresholds, and outcome semantics. This isolates geometry rather than material density. Phase 2R remains the historical large-object baseline. No transfer trajectory is simulated.

# Phase 2S geometry validation

`scene_builder.build_scene` writes each YAML object `size` directly to the MuJoCo geom `size` attribute. Compilation confirms standard box half-extent and cylinder [radius, half-height] semantics.

```json
{
  "object_linear_scale": 0.5,
  "yaml_size_is_passed_directly_to_MuJoCo_geom_size": true,
  "objects": {
    "object_a": {
      "large": {
        "shape": "cube",
        "size": [
          0.025,
          0.025,
          0.025
        ],
        "physical_dimensions_m": [
          0.05,
          0.05,
          0.05
        ],
        "mass_kg": 0.08,
        "default_center_m": [
          0.08,
          0.0,
          0.026
        ],
        "table_clearance_m": 0.0009999999999999974
      },
      "half_scale": {
        "shape": "cube",
        "size": [
          0.0125,
          0.0125,
          0.0125
        ],
        "physical_dimensions_m": [
          0.025,
          0.025,
          0.025
        ],
        "mass_kg": 0.08,
        "default_center_m": [
          0.08,
          0.0,
          0.0135
        ],
        "table_clearance_m": 0.0009999999999999992
      },
      "compiled_linear_dimension_ratio": [
        0.5,
        0.5,
        0.5
      ]
    },
    "object_b": {
      "large": {
        "shape": "cylinder",
        "size": [
          0.025,
          0.04
        ],
        "physical_dimensions_m": [
          0.05,
          0.05,
          0.08
        ],
        "mass_kg": 0.08,
        "default_center_m": [
          -0.08,
          0.0,
          0.041
        ],
        "table_clearance_m": 0.0010000000000000009
      },
      "half_scale": {
        "shape": "cylinder",
        "size": [
          0.0125,
          0.02
        ],
        "physical_dimensions_m": [
          0.025,
          0.025,
          0.04
        ],
        "mass_kg": 0.08,
        "default_center_m": [
          -0.08,
          0.0,
          0.021
        ],
        "table_clearance_m": 0.0010000000000000009
      },
      "compiled_linear_dimension_ratio": [
        0.5,
        0.5,
        0.5
      ]
    }
  },
  "unchanged_threshold_diagnostics": {
    "occupied_finger_force_threshold_N": {
      "value": 0.2,
      "normalized_by_min_dimension_N_per_m": 8.0
    },
    "tactile_binary_force_threshold_N": {
      "value": 0.05,
      "normalized_by_min_dimension_N_per_m": 2.0
    },
    "maximum_penetration_m": {
      "value": 0.003,
      "fraction_of_min_dimension": 0.12
    },
    "maximum_translation_drift_m": {
      "value": 0.005,
      "fraction_of_min_dimension": 0.19999999999999998
    },
    "maximum_rotation_drift_rad": {
      "value": 0.2
    }
  }
}
```

Mass remains fixed, so the experiment isolates geometric scale and does not preserve density. The normalized-threshold table is diagnostic only; no scientific threshold was changed.


## Regenerated endpoint populations, resources, and matching

# Phase 2S matching and resource report

No Phase 2R labels or resource values enter this dataset. All states were revalidated and all resource components recomputed with half-scale geometry.

- Populations: FINGERTIP 201, PALMAR_SECURED 200
- Calibration: 20+20
- Matched pairs: 100

## Balance before

```json
{
  "ferrari_canny_epsilon": {
    "FINGERTIP": {
      "mean": 0.1261626838535998,
      "standard_deviation": 0.011452958440248037
    },
    "PALMAR_SECURED": {
      "mean": 0.15308204098499345,
      "standard_deviation": 0.0654173912263576
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.573232206633185
  },
  "total_A_normal_force_N": {
    "FINGERTIP": {
      "mean": 4.557726647131821,
      "standard_deviation": 0.5667921398807048
    },
    "PALMAR_SECURED": {
      "mean": 5.514366263761597,
      "standard_deviation": 1.885232291491735
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.6872387876436186
  },
  "A_translation_drift_m": {
    "FINGERTIP": {
      "mean": 0.0029624649531168595,
      "standard_deviation": 0.0003544058681277618
    },
    "PALMAR_SECURED": {
      "mean": 0.0028045903270747605,
      "standard_deviation": 0.0009504515907616034
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.22010389001847203
  },
  "A_rotation_drift_rad": {
    "FINGERTIP": {
      "mean": 0.12071608080624736,
      "standard_deviation": 0.03439598824476208
    },
    "PALMAR_SECURED": {
      "mean": 0.1032569737256051,
      "standard_deviation": 0.04048817802019307
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.4647608610187982
  },
  "minimum_joint_margin_rad": {
    "FINGERTIP": {
      "mean": 0.051977883254535874,
      "standard_deviation": 0.02124327421742622
    },
    "PALMAR_SECURED": {
      "mean": 0.02774501752650528,
      "standard_deviation": 0.05030105303779657
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.6276309620333539
  }
}
```

## Balance after

```json
{
  "ferrari_canny_epsilon": {
    "FINGERTIP": {
      "mean": 0.1277626259866487,
      "standard_deviation": 0.012031472244979047
    },
    "PALMAR_SECURED": {
      "mean": 0.12739439990872664,
      "standard_deviation": 0.037959637564481175
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.013077367859712431
  },
  "total_A_normal_force_N": {
    "FINGERTIP": {
      "mean": 4.511783398798636,
      "standard_deviation": 0.6023012878493641
    },
    "PALMAR_SECURED": {
      "mean": 4.5786307336701375,
      "standard_deviation": 1.1434100901343527
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.07315110723628677
  },
  "A_translation_drift_m": {
    "FINGERTIP": {
      "mean": 0.003023799958492491,
      "standard_deviation": 0.00039688076674780875
    },
    "PALMAR_SECURED": {
      "mean": 0.003003195235651461,
      "standard_deviation": 0.0007091129826870433
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.03585856070403904
  },
  "A_rotation_drift_rad": {
    "FINGERTIP": {
      "mean": 0.11949900408820195,
      "standard_deviation": 0.03331239830625448
    },
    "PALMAR_SECURED": {
      "mean": 0.11452218151112968,
      "standard_deviation": 0.035378351634694764
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.14483968083360924
  },
  "minimum_joint_margin_rad": {
    "FINGERTIP": {
      "mean": 0.04854564209580996,
      "standard_deviation": 0.023792769095617103
    },
    "PALMAR_SECURED": {
      "mean": 0.033323874856145894,
      "standard_deviation": 0.05276401936156856
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.3719193309891041
  }
}
```

## Resource distributions

```json
{
  "FINGERTIP": {
    "occupied_finger_count": {
      "count": 201,
      "minimum": 3.0,
      "maximum": 3.0,
      "mean": 3.0,
      "standard_deviation": 0.0,
      "median": 3.0
    },
    "free_finger_count": {
      "count": 201,
      "minimum": 1.0,
      "maximum": 1.0,
      "mean": 1.0,
      "standard_deviation": 0.0,
      "median": 1.0
    },
    "free_finger_workspace_vol_m3": {
      "count": 201,
      "minimum": 0.000132,
      "maximum": 0.00015087500000000002,
      "mean": 0.0001419676616915423,
      "standard_deviation": 3.883584717110615e-06,
      "median": 0.00014187500000000002
    },
    "free_palm_volume_m3": {
      "count": 201,
      "minimum": 0.0032982500000000004,
      "maximum": 0.0033047500000000004,
      "mean": 0.0033013756218905474,
      "standard_deviation": 1.1572122693432102e-06,
      "median": 0.0033013750000000005
    },
    "COM_to_palm_origin_distance_m": {
      "count": 201,
      "minimum": 0.06561318929338138,
      "maximum": 0.07254792647189866,
      "mean": 0.0675743663582731,
      "standard_deviation": 0.0009231335460027991,
      "median": 0.06757236207666135
    },
    "palm_A_contact_fraction": {
      "count": 201,
      "minimum": 0.0,
      "maximum": 0.0,
      "mean": 0.0,
      "standard_deviation": 0.0,
      "median": 0.0
    },
    "nearest_palm_storage_boundary_m": {
      "count": 201,
      "minimum": 0.045870668313355026,
      "maximum": 0.05085703069612281,
      "mean": 0.04875553856525659,
      "standard_deviation": 0.0009315107155818264,
      "median": 0.048739759433672286
    }
  },
  "PALMAR_SECURED": {
    "occupied_finger_count": {
      "count": 200,
      "minimum": 1.0,
      "maximum": 2.0,
      "mean": 1.05,
      "standard_deviation": 0.21849186132974543,
      "median": 1.0
    },
    "free_finger_count": {
      "count": 200,
      "minimum": 2.0,
      "maximum": 3.0,
      "mean": 2.95,
      "standard_deviation": 0.21849186132974535,
      "median": 3.0
    },
    "free_finger_workspace_vol_m3": {
      "count": 200,
      "minimum": 0.00013887500000000003,
      "maximum": 0.0006346250000000001,
      "mean": 0.00045746437500000006,
      "standard_deviation": 0.00019576410633561327,
      "median": 0.0005810000000000001
    },
    "free_palm_volume_m3": {
      "count": 200,
      "minimum": 0.0033268750000000004,
      "maximum": 0.0033878750000000007,
      "mean": 0.003345280625000001,
      "standard_deviation": 2.3621546627172268e-05,
      "median": 0.0033320000000000008
    },
    "COM_to_palm_origin_distance_m": {
      "count": 200,
      "minimum": 0.026391900103303858,
      "maximum": 0.057357659022828064,
      "mean": 0.04052601382486938,
      "standard_deviation": 0.00938340430304132,
      "median": 0.04050563628831526
    },
    "palm_A_contact_fraction": {
      "count": 200,
      "minimum": 0.82,
      "maximum": 1.0,
      "mean": 0.98047,
      "standard_deviation": 0.0294123192321358,
      "median": 0.988
    },
    "nearest_palm_storage_boundary_m": {
      "count": 200,
      "minimum": 0.03457680931802501,
      "maximum": 0.05513728804641679,
      "mean": 0.04643157339675602,
      "standard_deviation": 0.007378683150943909,
      "median": 0.05009584058744313
    }
  }
}
```

## Packing description

```json
{
  "method": "axis-aligned bounding-box grid count inside configured palm measurement volume",
  "palm_box_dimensions_m": [
    0.14,
    0.16,
    0.16
  ],
  "object_A_bounding_dimensions_m": [
    0.025,
    0.025,
    0.025
  ],
  "copies_per_axis": [
    5,
    6,
    6
  ],
  "maximum_non_overlapping_axis_aligned_copies": 180,
  "interpretation": "geometry-only descriptive upper-bound construction; not a manipulation success metric"
}
```


## Small-B graspability, common region, and controller calibration

The small-B map evaluated 8,192 geometry candidates. The strict dynamic search found 24 successful pose/trajectory combinations after 4,392 evaluated candidates. Ten correctly paired profiles were locally perturbed; the frozen common distribution and controller are recorded in `PHASE2S_B_DISTRIBUTION_FREEZE.md` and `PHASE2S_CONTROLLER_FREEZE.md`.

## Formal, paired, eligibility, resource, and Ferrari–Canny results

```json
{
  "primary": {
    "FINGERTIP": {
      "successes": 0,
      "valid_trials": 2000,
      "rate": 0.0,
      "wilson_95_CI": [
        1.0842021724855044e-19,
        0.0019170472812529349
      ]
    },
    "PALMAR_SECURED": {
      "successes": 65,
      "valid_trials": 2000,
      "rate": 0.0325,
      "wilson_95_CI": [
        0.02558069882617027,
        0.04121174038180121
      ]
    },
    "absolute_percentage_point_difference_palmar_minus_fingertip": 3.25,
    "relative_risk_palmar_over_fingertip": "infinite",
    "odds_ratio_palmar_over_fingertip": "infinite"
  },
  "paired": {
    "paired_cells": 2000,
    "palmar_succeeds_fingertip_fails": 65,
    "fingertip_succeeds_palmar_fails": 0,
    "both_succeed": 0,
    "both_fail": 1935,
    "excluded_invalid_or_incomplete": 0,
    "McNemar_exact_two_sided_p_value": 5.421010862427522e-20,
    "pair_level_bootstrap_replicates": 10000,
    "pair_level_bootstrap_pairs_with_valid_trials_in_both_groups": 100,
    "absolute_rate_difference_bootstrap_95_CI": [
      0.0115,
      0.055999999999999994
    ]
  },
  "eligibility": {
    "FINGERTIP": {
      "eligible_states": 0,
      "state_count": 100,
      "fraction": 0.0,
      "wilson_95_CI": [
        0.0,
        0.03699349820698569
      ]
    },
    "PALMAR_SECURED": {
      "eligible_states": 100,
      "state_count": 100,
      "fraction": 1.0,
      "wilson_95_CI": [
        0.9630065017930143,
        1.0
      ]
    }
  },
  "eligible_reachable": {
    "FINGERTIP": {
      "successes": 0,
      "valid_trials": 0,
      "rate": null,
      "wilson_95_CI": null
    },
    "PALMAR_SECURED": {
      "successes": 65,
      "valid_trials": 1900,
      "rate": 0.034210526315789476,
      "wilson_95_CI": [
        0.02693150030423494,
        0.043369237291636206
      ]
    }
  },
  "resources": {
    "occupied_finger_count": {
      "FINGERTIP_mean": 3.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 1.08,
      "PALMAR_SECURED_standard_deviation": 0.2726599243442907,
      "mean_difference_palmar_minus_fingertip": -1.92,
      "standardized_mean_difference": -9.958522677236994
    },
    "free_finger_count": {
      "FINGERTIP_mean": 1.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 2.92,
      "PALMAR_SECURED_standard_deviation": 0.2726599243442907,
      "mean_difference_palmar_minus_fingertip": 1.92,
      "standardized_mean_difference": 9.958522677236994
    },
    "free_finger_workspace_vol_m3": {
      "FINGERTIP_mean": 0.00014221125000000003,
      "FINGERTIP_standard_deviation": 4.018686426363016e-06,
      "PALMAR_SECURED_mean": 0.000494135,
      "PALMAR_SECURED_standard_deviation": 0.00017310299351437102,
      "mean_difference_palmar_minus_fingertip": 0.00035192375,
      "standardized_mean_difference": 2.8743655122999385
    },
    "free_palm_volume_m3": {
      "FINGERTIP_mean": 0.0033014275000000006,
      "FINGERTIP_standard_deviation": 1.1228486725352673e-06,
      "PALMAR_SECURED_mean": 0.00333724875,
      "PALMAR_SECURED_standard_deviation": 1.7682672985913854e-05,
      "mean_difference_palmar_minus_fingertip": 3.582124999999955e-05,
      "standardized_mean_difference": 2.859130572271104
    },
    "COM_to_palm_origin_distance_m": {
      "FINGERTIP_mean": 0.06745830698299222,
      "FINGERTIP_standard_deviation": 0.0009829914275893189,
      "PALMAR_SECURED_mean": 0.03915467450623607,
      "PALMAR_SECURED_standard_deviation": 0.008727779791402669,
      "mean_difference_palmar_minus_fingertip": -0.028303632476756158,
      "standardized_mean_difference": -4.557390014369412
    },
    "palm_A_contact_fraction": {
      "FINGERTIP_mean": 0.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 0.9816400000000001,
      "PALMAR_SECURED_standard_deviation": 0.028246219019346444,
      "mean_difference_palmar_minus_fingertip": 0.9816400000000001,
      "standardized_mean_difference": 49.148121396958786
    },
    "palm_A_normal_force_N": {
      "FINGERTIP_mean": 0.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 2.0154783993231042,
      "PALMAR_SECURED_standard_deviation": 0.5949826180315947,
      "mean_difference_palmar_minus_fingertip": 2.0154783993231042,
      "standardized_mean_difference": 4.790588498908709
    },
    "ferrari_canny_epsilon": {
      "FINGERTIP_mean": 0.1277626259866487,
      "FINGERTIP_standard_deviation": 0.012031472244979047,
      "PALMAR_SECURED_mean": 0.12739439990872664,
      "PALMAR_SECURED_standard_deviation": 0.037959637564481175,
      "mean_difference_palmar_minus_fingertip": -0.00036822607792205697,
      "standardized_mean_difference": -0.013077367859712431
    },
    "total_A_normal_force_N": {
      "FINGERTIP_mean": 4.511783398798636,
      "FINGERTIP_standard_deviation": 0.6023012878493641,
      "PALMAR_SECURED_mean": 4.5786307336701375,
      "PALMAR_SECURED_standard_deviation": 1.1434100901343527,
      "mean_difference_palmar_minus_fingertip": 0.0668473348715013,
      "standardized_mean_difference": 0.07315110723628677
    },
    "A_translation_drift_m": {
      "FINGERTIP_mean": 0.003023799958492491,
      "FINGERTIP_standard_deviation": 0.00039688076674780875,
      "PALMAR_SECURED_mean": 0.003003195235651461,
      "PALMAR_SECURED_standard_deviation": 0.0007091129826870433,
      "mean_difference_palmar_minus_fingertip": -2.0604722841029675e-05,
      "standardized_mean_difference": -0.03585856070403904
    },
    "A_rotation_drift_rad": {
      "FINGERTIP_mean": 0.11949900408820195,
      "FINGERTIP_standard_deviation": 0.03331239830625448,
      "PALMAR_SECURED_mean": 0.11452218151112968,
      "PALMAR_SECURED_standard_deviation": 0.035378351634694764,
      "mean_difference_palmar_minus_fingertip": -0.004976822577072271,
      "standardized_mean_difference": -0.14483968083360924
    }
  },
  "Ferrari_Canny": {
    "matched_group_distributions": {
      "FINGERTIP": {
        "count": 100,
        "mean": 0.1277626259866487,
        "standard_deviation": 0.012031472244979047,
        "minimum": 0.09919955041053785,
        "median": 0.12751061671296868,
        "maximum": 0.15499985464585508
      },
      "PALMAR_SECURED": {
        "count": 100,
        "mean": 0.12739439990872664,
        "standard_deviation": 0.037959637564481175,
        "minimum": 0.044321507094275056,
        "median": 0.11947675063763491,
        "maximum": 0.21362905241031688
      }
    },
    "top_decile_cutoff": 0.15742464016986618,
    "top_decile_success_rate": 0.1625,
    "full_population_success_rate": 0.01625,
    "point_biserial_correlation": {
      "r": 0.3764411150428898,
      "p_value": 7.264760029703512e-135
    }
  },
  "failures": {
    "FINGERTIP": {
      "outcomes": {
        "BOTH_RETAINED": 0,
        "A_DROPPED": 0,
        "B_NOT_ACQUIRED": 2000,
        "BOTH_LOST": 0,
        "INVALID": 0
      },
      "subreasons": {
        "INSUFFICIENT_FREE_DIGITS_PRECHECK": 2000
      }
    },
    "PALMAR_SECURED": {
      "outcomes": {
        "BOTH_RETAINED": 65,
        "A_DROPPED": 35,
        "B_NOT_ACQUIRED": 1215,
        "BOTH_LOST": 685,
        "INVALID": 0
      },
      "subreasons": {
        "B_SLIP": 1120,
        "A_DESTABILIZED": 720,
        "B_NOT_ACQUIRED": 95,
        "None": 65
      }
    }
  }
}
```

## Large-versus-half-scale descriptive comparison

This cross-experiment comparison is descriptive only; raw trials are not pooled as though size were randomized within one experiment, and it does not establish causal significance.

```json
{
  "interpretation": "cross-experiment descriptive comparison only; objects were not randomized within one experiment",
  "palmar_state_acceptance": {
    "Phase2R_large": {
      "accepted": 150,
      "attempts": 4672,
      "rate": 0.03210616438356165
    },
    "Phase2S_half_scale": {
      "accepted": 200,
      "attempts": 544,
      "rate": 0.36764705882352944
    }
  },
  "palmar_resource_means": {
    "occupied_finger_count": {
      "Phase2R_large": 1.38,
      "Phase2S_half_scale": 1.08
    },
    "free_finger_count": {
      "Phase2R_large": 2.62,
      "Phase2S_half_scale": 2.92
    },
    "free_palm_volume_m3": {
      "Phase2R_large": 0.0032443262500000004,
      "Phase2S_half_scale": 0.00333724875
    }
  },
  "digit_eligibility": {
    "Phase2R_large": {
      "FINGERTIP": {
        "eligible_states": 1,
        "state_count": 100,
        "fraction": 0.01,
        "wilson_95_CI": [
          0.001767432064140647,
          0.05448619617870533
        ]
      },
      "PALMAR_SECURED": {
        "eligible_states": 100,
        "state_count": 100,
        "fraction": 1.0,
        "wilson_95_CI": [
          0.9630065017930143,
          1.0
        ]
      }
    },
    "Phase2S_half_scale": {
      "FINGERTIP": {
        "eligible_states": 0,
        "state_count": 100,
        "fraction": 0.0,
        "wilson_95_CI": [
          0.0,
          0.03699349820698569
        ]
      },
      "PALMAR_SECURED": {
        "eligible_states": 100,
        "state_count": 100,
        "fraction": 1.0,
        "wilson_95_CI": [
          0.9630065017930143,
          1.0
        ]
      }
    }
  },
  "B_geometry_access": {
    "Phase2R_large": {
      "FINGERTIP": {
        "state_placement_pairs": 20000,
        "reachable_pairs": 3800,
        "access_fraction": 0.19,
        "states_with_any_access": 19,
        "initial_A_overlap_pairs": 0,
        "initial_hand_overlap_pairs": 15052
      },
      "PALMAR_SECURED": {
        "state_placement_pairs": 20000,
        "reachable_pairs": 16717,
        "access_fraction": 0.83585,
        "states_with_any_access": 86,
        "initial_A_overlap_pairs": 0,
        "initial_hand_overlap_pairs": 0
      }
    },
    "Phase2S_half_scale": {
      "FINGERTIP": {
        "state_placement_pairs": 20000,
        "reachable_pairs": 20000,
        "access_fraction": 1.0,
        "states_with_any_access": 100,
        "initial_A_overlap_pairs": 0,
        "initial_hand_overlap_pairs": 0
      },
      "PALMAR_SECURED": {
        "state_placement_pairs": 20000,
        "reachable_pairs": 18922,
        "access_fraction": 0.9461,
        "states_with_any_access": 95,
        "initial_A_overlap_pairs": 0,
        "initial_hand_overlap_pairs": 0
      }
    }
  },
  "BOTH_RETAINED": {
    "Phase2R_large": {
      "FINGERTIP": {
        "successes": 0,
        "valid_trials": 1980,
        "rate": 0.0,
        "wilson_95_CI": [
          0.0,
          0.0019363738990401504
        ]
      },
      "PALMAR_SECURED": {
        "successes": 66,
        "valid_trials": 2000,
        "rate": 0.033,
        "wilson_95_CI": [
          0.026022752700289677,
          0.041767769460400554
        ]
      },
      "absolute_percentage_point_difference_palmar_minus_fingertip": 3.3000000000000003,
      "relative_risk_palmar_over_fingertip": "infinite",
      "odds_ratio_palmar_over_fingertip": "infinite"
    },
    "Phase2S_half_scale": {
      "FINGERTIP": {
        "successes": 0,
        "valid_trials": 2000,
        "rate": 0.0,
        "wilson_95_CI": [
          1.0842021724855044e-19,
          0.0019170472812529349
        ]
      },
      "PALMAR_SECURED": {
        "successes": 65,
        "valid_trials": 2000,
        "rate": 0.0325,
        "wilson_95_CI": [
          0.02558069882617027,
          0.04121174038180121
        ]
      },
      "absolute_percentage_point_difference_palmar_minus_fingertip": 3.25,
      "relative_risk_palmar_over_fingertip": "infinite",
      "odds_ratio_palmar_over_fingertip": "infinite"
    }
  },
  "failure_modes": {
    "Phase2R_large": {
      "FINGERTIP": {
        "outcomes": {
          "BOTH_RETAINED": 0,
          "A_DROPPED": 0,
          "B_NOT_ACQUIRED": 1980,
          "BOTH_LOST": 0,
          "INVALID": 20
        },
        "subreasons": {
          "INSUFFICIENT_FREE_DIGITS_PRECHECK": 1980,
          "INITIAL_OVERLAP": 20
        }
      },
      "PALMAR_SECURED": {
        "outcomes": {
          "BOTH_RETAINED": 66,
          "A_DROPPED": 9,
          "B_NOT_ACQUIRED": 1434,
          "BOTH_LOST": 491,
          "INVALID": 0
        },
        "subreasons": {
          "B_SLIP": 1360,
          "A_DESTABILIZED": 500,
          "B_NOT_ACQUIRED": 74,
          "None": 66
        }
      }
    },
    "Phase2S_half_scale": {
      "FINGERTIP": {
        "outcomes": {
          "BOTH_RETAINED": 0,
          "A_DROPPED": 0,
          "B_NOT_ACQUIRED": 2000,
          "BOTH_LOST": 0,
          "INVALID": 0
        },
        "subreasons": {
          "INSUFFICIENT_FREE_DIGITS_PRECHECK": 2000
        }
      },
      "PALMAR_SECURED": {
        "outcomes": {
          "BOTH_RETAINED": 65,
          "A_DROPPED": 35,
          "B_NOT_ACQUIRED": 1215,
          "BOTH_LOST": 685,
          "INVALID": 0
        },
        "subreasons": {
          "B_SLIP": 1120,
          "A_DESTABILIZED": 720,
          "B_NOT_ACQUIRED": 95,
          "None": 65
        }
      }
    }
  }
}
```

## Limitations

Endpoint initialization does not establish transfer controllability. The fixed-mass scale change increases density. One scripted B controller and one pre-frozen region probe only a narrow acquisition family. No scalar resource score, wrist controller, three-object task, transfer controller, or RL policy is defined.
