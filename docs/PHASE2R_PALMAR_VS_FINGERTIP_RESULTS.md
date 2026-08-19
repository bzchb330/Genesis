# Phase 2R palmar-secured versus fingertip results

## Interpretation boundary

The previous Phase 2 state was an acquisition-state grasp. Phase 2R directly initializes and validates a post-transfer palmar-secured endpoint for comparison; no transfer dynamics are simulated and no weld/equality constraint is present during validation or formal trials. Physics and thresholds remain frozen.

The experiment does not demonstrate the control process required to transfer an object from fingertip acquisition to palmar secure storage. It evaluates whether the post-transfer endpoint state provides greater capability for subsequent acquisition.

## Endpoint definitions and physical validation

FINGERTIP states replay accepted Phase 2 acquisition grasps and require at least two participating finger contacts, no persistent palm contact, no table support, and the frozen unsupported-hold gate. PALMAR_SECURED states are sampled directly in the existing palm region, require persistent physical palm contact for at least 80% of the stable window, one or two load-bearing fingers at the frozen 0.20 N threshold, no table support, and the same unsupported-hold gate.

Palmar candidates use a temporary free-joint pose fixture only during initialization and closure. The fixture is removed before validation; accepted states have a free object joint, zero equality constraints, and retention only through palm/finger contact, friction, gravity, and unchanged MuJoCo dynamics.

The FINGERTIP filter accepted 221 of 227 replayed states. The deterministic PALMAR_SECURED search reached 150 accepted states after 4,672 of the authorized 30,000 attempts without relaxing criteria. Palmar load-bearing topologies were thumb 76, middle 13, middle+thumb 53, ring+thumb 6, and middle+ring 2.

## Dataset, matching, and freezes

Validated endpoint states: FINGERTIP 221 and PALMAR_SECURED 150. Formal matching used 100 non-reused pairs after reserving 20+20 calibration states. See `PHASE2R_MATCHING_REPORT.md`. The common B region and generic controller were frozen before formal outcomes; see `PHASE2R_B_DISTRIBUTION_FREEZE.md` and `PHASE2R_CONTROLLER_FREEZE.md`.

Matching used standardized Ferrari–Canny epsilon, total A normal force, translation drift, rotation drift, and minimum joint margin only. It did not use B outcomes or the resource variables hypothesized to differ between endpoint states. Residual matching imbalance is disclosed below and in the matching report.

## Common B distribution and controller calibration

Geometry-only selection froze the Phase 2.6 `index_thumb_region` for both groups: x=[0.0453599871, 0.0473599871] m, y=[0.0840990493, 0.0860990493] m, z=[0.2239900112, 0.2259900112] m, and yaw=[-0.1, 0.1] rad. Matched-state geometric access was 0.19 for FINGERTIP and 0.83585 for PALMAR_SECURED, with zero initial A-overlap pairs in either group.

The separate 20+20-state calibration evaluated three existing Phase 2.6 controller families with five B seeds per state (200 planned records per candidate). Candidate 01 produced 0 BOTH_RETAINED, candidate 02 produced 2 across one A state and two B seeds, and candidate 03 produced 0. The pooled lexicographic rule selected `phase2_6_b_only_02`; no group-rate difference was used and no trajectory-search expansion was needed. The fixed acquisition digits are assigned once by geometry before motion and never reassigned.

## Formal paired experiment

All 4,000 planned records completed: 100 matched pairs × two endpoint types × 20 shared formal B seeds. Records are deterministic, incremental, resumable, and use the separate `phase2R_palmar_vs_fingertip_formal` experiment ID. Ineligible states are recorded as B_NOT_ACQUIRED with `INSUFFICIENT_FREE_DIGITS_PRECHECK` without a meaningless dynamic attempt.

## Eligibility and formal results

```json
{
  "eligibility": {
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
  "primary": {
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
  "conditional_eligible_reachable": {
    "FINGERTIP": {
      "successes": 0,
      "valid_trials": 0,
      "rate": null,
      "wilson_95_CI": null
    },
    "PALMAR_SECURED": {
      "successes": 66,
      "valid_trials": 1653,
      "rate": 0.039927404718693285,
      "wilson_95_CI": [
        0.03150647688196469,
        0.05048172910932147
      ]
    }
  }
}
```

The conditional eligible-and-reachable FINGERTIP success rate is not identifiable because no FINGERTIP trial entered that analysis set; it must not be interpreted as zero.

## Paired comparison

```json
{
  "paired_cells": 2000,
  "palmar_succeeds_fingertip_fails": 66,
  "fingertip_succeeds_palmar_fails": 0,
  "both_succeed": 0,
  "both_fail": 1914,
  "excluded_invalid_or_incomplete": 20,
  "McNemar_exact_two_sided_p_value": 2.710505431213761e-20,
  "pair_level_bootstrap_replicates": 10000,
  "pair_level_bootstrap_pairs_with_valid_trials_in_both_groups": 99,
  "absolute_rate_difference_bootstrap_95_CI": [
    0.011111111111111112,
    0.05909090909090909
  ]
}
```

## Clustered models

The adjusted model includes only the prespecified baseline stability covariates and does not control away digit occupancy, workspace, palm contact, COM position, or free-palm volume. Because FINGERTIP had zero successes, both logistic fits exhibit complete or quasi-complete separation and very large coefficient uncertainty. These numerical fits are unstable and are not the primary inferential result; the prespecified paired exact comparison is primary.

```json
{
  "primary": {
    "status": "FIT",
    "formula": "BOTH_RETAINED ~ palmar",
    "cluster": "matched_pair_id",
    "N": 3980,
    "terms": {
      "const": {
        "coefficient": -27.56606852316694,
        "cluster_robust_standard_error": 512.5164032305785,
        "z": -0.05378572929453169,
        "p_value": 0.9571058794344299,
        "confidence_interval_95": [
          -1032.0797603411086,
          976.9476232947748
        ],
        "odds_ratio": 1.0671084345210936e-12
      },
      "palmar": {
        "coefficient": 24.18837758918025,
        "cluster_robust_standard_error": 512.5165457320131,
        "z": 0.04719531065018137,
        "p_value": 0.9623575649183957,
        "confidence_interval_95": [
          -980.325593526441,
          1028.7023487048016
        ],
        "odds_ratio": 31980033413.617832
      }
    },
    "standardized_adjustment_covariates": {}
  },
  "adjusted": {
    "status": "FIT",
    "formula": "BOTH_RETAINED ~ palmar + ferrari_canny_epsilon + A_translation_drift_m + A_rotation_drift_rad + minimum_joint_margin_rad",
    "cluster": "matched_pair_id",
    "N": 3980,
    "terms": {
      "const": {
        "coefficient": -33.21172895774724,
        "cluster_robust_standard_error": 188.58964765604296,
        "z": -0.1761057903789081,
        "p_value": 0.8602108319577477,
        "confidence_interval_95": [
          -402.8406462206901,
          336.41718830519557
        ],
        "odds_ratio": 3.769896084034482e-15
      },
      "palmar": {
        "coefficient": 24.885643658202625,
        "cluster_robust_standard_error": 188.57218410641434,
        "z": 0.13196879368040437,
        "p_value": 0.8950089748619716,
        "confidence_interval_95": [
          -344.7090456764259,
          394.48033299283117
        ],
        "odds_ratio": 64224054503.2253
      },
      "ferrari_canny_epsilon": {
        "coefficient": 1.8552577333269527,
        "cluster_robust_standard_error": 0.9173193438272648,
        "z": 2.022477500132501,
        "p_value": 0.04312704573438394,
        "confidence_interval_95": [
          0.057344857103599045,
          3.6531706095503065
        ],
        "odds_ratio": 6.393345816583118
      },
      "A_translation_drift_m": {
        "coefficient": 0.47806066365407096,
        "cluster_robust_standard_error": 0.3782389691741703,
        "z": 1.2639117135335018,
        "p_value": 0.20626171701806384,
        "confidence_interval_95": [
          -0.2632740934768586,
          1.2193954207850006
        ],
        "odds_ratio": 1.6129433274518563
      },
      "A_rotation_drift_rad": {
        "coefficient": -0.503977276874674,
        "cluster_robust_standard_error": 0.6368791648338762,
        "z": -0.7913232284904965,
        "p_value": 0.42875539688169295,
        "confidence_interval_95": [
          -1.75223750245302,
          0.744282948703672
        ],
        "odds_ratio": 0.6041231102650462
      },
      "minimum_joint_margin_rad": {
        "coefficient": -2.9233051944467645,
        "cluster_robust_standard_error": 0.6940278301097502,
        "z": -4.212086414437426,
        "p_value": 2.530227053719495e-05,
        "confidence_interval_95": [
          -4.283574745730358,
          -1.5630356431631707
        ],
        "odds_ratio": 0.053755720246470365
      }
    },
    "standardized_adjustment_covariates": {
      "ferrari_canny_epsilon": {
        "mean": 0.11947008838624337,
        "standard_deviation": 0.04386227579928865
      },
      "A_translation_drift_m": {
        "mean": 0.0021932905730514497,
        "standard_deviation": 0.001135807425328063
      },
      "A_rotation_drift_rad": {
        "mean": 0.049804393983340994,
        "standard_deviation": 0.02518781689166017
      },
      "minimum_joint_margin_rad": {
        "mean": 0.14016604563845556,
        "standard_deviation": 0.08807396526630688
      }
    }
  }
}
```

## Resource distributions and Ferrari–Canny baseline

```json
{
  "resources": {
    "occupied_finger_count": {
      "FINGERTIP_mean": 3.49,
      "FINGERTIP_standard_deviation": 0.5221362491019587,
      "PALMAR_SECURED_mean": 1.38,
      "PALMAR_SECURED_standard_deviation": 0.48783173121456336,
      "mean_difference_palmar_minus_fingertip": -2.1100000000000003,
      "standardized_mean_difference": -4.175942119043603
    },
    "free_finger_count": {
      "FINGERTIP_mean": 0.51,
      "FINGERTIP_standard_deviation": 0.5221362491019587,
      "PALMAR_SECURED_mean": 2.62,
      "PALMAR_SECURED_standard_deviation": 0.48783173121456336,
      "mean_difference_palmar_minus_fingertip": 2.1100000000000003,
      "standardized_mean_difference": 4.175942119043603
    },
    "free_finger_workspace_vol_m3": {
      "FINGERTIP_mean": 6.360750000000001e-05,
      "FINGERTIP_standard_deviation": 6.68366441502162e-05,
      "PALMAR_SECURED_mean": 0.00013545750000000005,
      "PALMAR_SECURED_standard_deviation": 0.00012866417559705853,
      "mean_difference_palmar_minus_fingertip": 7.185000000000004e-05,
      "standardized_mean_difference": 0.7008237532679823
    },
    "free_palm_volume_m3": {
      "FINGERTIP_mean": 0.003166598750000001,
      "FINGERTIP_standard_deviation": 1.4696103970323768e-05,
      "PALMAR_SECURED_mean": 0.0032443262500000004,
      "PALMAR_SECURED_standard_deviation": 1.87562522824882e-05,
      "mean_difference_palmar_minus_fingertip": 7.77274999999994e-05,
      "standardized_mean_difference": 4.613202633581652
    },
    "COM_to_palm_origin_distance_m": {
      "FINGERTIP_mean": 0.06962754452070827,
      "FINGERTIP_standard_deviation": 0.0055156106663110166,
      "PALMAR_SECURED_mean": 0.044350839833396834,
      "PALMAR_SECURED_standard_deviation": 0.003683814694285788,
      "mean_difference_palmar_minus_fingertip": -0.025276704687311434,
      "standardized_mean_difference": -5.389473853036743
    },
    "palm_A_contact_fraction": {
      "FINGERTIP_mean": 0.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 0.97292,
      "PALMAR_SECURED_standard_deviation": 0.022071362678934222,
      "mean_difference_palmar_minus_fingertip": 0.97292,
      "standardized_mean_difference": 62.33945221774914
    },
    "palm_A_normal_force_N": {
      "FINGERTIP_mean": 0.0,
      "FINGERTIP_standard_deviation": 0.0,
      "PALMAR_SECURED_mean": 1.3820653502989344,
      "PALMAR_SECURED_standard_deviation": 0.5004341638696648,
      "mean_difference_palmar_minus_fingertip": 1.3820653502989344,
      "standardized_mean_difference": 3.9056797149199487
    },
    "ferrari_canny_epsilon": {
      "FINGERTIP_mean": 0.1143242808863501,
      "FINGERTIP_standard_deviation": 0.0420498940823379,
      "PALMAR_SECURED_mean": 0.12434636064984791,
      "PALMAR_SECURED_standard_deviation": 0.045338852005174554,
      "mean_difference_palmar_minus_fingertip": 0.01002207976349781,
      "standardized_mean_difference": 0.22920547060407784
    },
    "total_A_normal_force_N": {
      "FINGERTIP_mean": 4.388303026321433,
      "FINGERTIP_standard_deviation": 1.4568108428890458,
      "PALMAR_SECURED_mean": 4.31586093421058,
      "PALMAR_SECURED_standard_deviation": 1.5746044557715062,
      "mean_difference_palmar_minus_fingertip": -0.07244209211085284,
      "standardized_mean_difference": -0.047758196169481223
    },
    "A_translation_drift_m": {
      "FINGERTIP_mean": 0.0014871713244170891,
      "FINGERTIP_standard_deviation": 0.00042417994733609434,
      "PALMAR_SECURED_mean": 0.002886998124519403,
      "PALMAR_SECURED_standard_deviation": 0.0011996032806938916,
      "mean_difference_palmar_minus_fingertip": 0.001399826800102314,
      "standardized_mean_difference": 1.5558545993956658
    },
    "A_rotation_drift_rad": {
      "FINGERTIP_mean": 0.04389730476708177,
      "FINGERTIP_standard_deviation": 0.020643031687395605,
      "PALMAR_SECURED_mean": 0.05607275764241559,
      "PALMAR_SECURED_standard_deviation": 0.02806018792631413,
      "mean_difference_palmar_minus_fingertip": 0.012175452875333816,
      "standardized_mean_difference": 0.49428637391996233
    }
  },
  "Ferrari_Canny": {
    "matched_group_distributions": {
      "FINGERTIP": {
        "count": 100,
        "mean": 0.1143242808863501,
        "standard_deviation": 0.0420498940823379,
        "minimum": 0.0009168948568676895,
        "median": 0.10932878025931561,
        "maximum": 0.24252410261445445
      },
      "PALMAR_SECURED": {
        "count": 100,
        "mean": 0.12434636064984791,
        "standard_deviation": 0.045338852005174554,
        "minimum": 0.008395326140255234,
        "median": 0.12756745367174183,
        "maximum": 0.21547840392588588
      }
    },
    "top_decile_cutoff": 0.17286194804481275,
    "top_decile_success_rate": 0.0975,
    "full_population_success_rate": 0.016582914572864323,
    "point_biserial_correlation": {
      "r": 0.14010632170987747,
      "p_value": 6.68748280054636e-19
    }
  }
}
```

## Failure modes

```json
{
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
}
```

## Limitations

Direct endpoint initialization does not establish transfer controllability. Scripted B acquisition probes one frozen controller and region. The matched groups retain residual imbalance in baseline translation drift, rotation drift, and minimum joint margin, reported transparently in the matching report. No scalar J, wrist controller, three-object task, or RL policy is defined.
