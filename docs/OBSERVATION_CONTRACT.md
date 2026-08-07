# Observation Contract

The flat vector is assembled in the order below. `seqgrasp.env.observations.observation_spec` is the authoritative machine-readable API; `docs/observation_spec.json` records the default configuration.

| Name | Dimension | Unit | Source | Privileged | Default |
|---|---:|---|---|---|---|
| joint_positions | configured hand DoF | rad | joint encoders | No | Enabled |
| joint_velocities | configured hand DoF | rad/s | joint encoders | No | Enabled |
| tactile_contact_flags | configured finger count | 1 | reference contact sensing | No | Enabled |
| tactile_normal_forces | configured finger count | N when normalization is null | reference contact sensing | No | Enabled |
| palm_pose | 7 | m and unit quaternion | robot state estimator | No | Enabled |
| phase_one_hot | 5 | 1 | task state machine | No | Enabled |
| privileged_target_position | 3 | m | MuJoCo body pose | **Yes** | Disabled |

Tests verify that the flat dimension is exactly the sum of enabled component dimensions and that privileged target position can be toggled independently.
