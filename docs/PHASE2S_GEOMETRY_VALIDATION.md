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
