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
