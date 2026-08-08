# Raw Resource Feature Contract

`ResourceState` and the resource-probing export expose measurements; they do not aggregate them into a resource score. `compute_resource_metric` continues to return `None` pending a PI definition of J.

| Group | Exported quantity | Shape / columns | Unit | Source |
|---|---|---|---|---|
| Hand state | joint position, velocity, lower/upper limit | 16 each | rad, rad/s | MuJoCo qpos/qvel/model ranges |
| Joint occupancy | lower/upper margin, normalized range position | 16×2, 16 | rad, dimensionless | Derived independently per joint |
| Actuation | control, lower/upper control limit | 16, 16×2 | N m | MuJoCo control plus configured torque limit |
| Actuation headroom | absolute utilization, positive/negative reserve | 16 each | dimensionless, N m | Derived independently per actuator |
| Object-A contact | active flag, count, position, normal, signed distance, summed normal force | per configured finger | 1, count, m, 1, m, N | Official MuJoCo contacts/contact force |
| Tactile reference | binary flag, total normal force | per configured finger | 1, N | Existing tactile feature extractor |
| Object A | pose, linear/angular velocity, clearance, orientation change | 3+4, 3, 3, scalar | m, quaternion, m/s, rad/s, m, rad | MuJoCo state and descriptive derivation |
| Hand geometry | palm pose, fingertip positions | 7, 4×3 | m/quaternion, m | MuJoCo body poses |
| Object B | pose, fingertip signed geom distance, contact count | 3+4, 4, 4 | m/quaternion, m, count | MuJoCo body pose, `mj_geomDistance`, contact extraction |

Column names in `resource_raw_samples.csv` contain unit suffixes where applicable (`_rad`, `_rad_s`, `_Nm`, `_N`, `_m`, `_m_s`). Joint and actuator columns follow the configured hand order; finger columns follow the configured mapping order: index, middle, ring, thumb. The experiment emitted 3,000 post-release rows across three grasps and 20 seeds.

Reachability exports contain 12 coordinates per sample: fingertip world xyz and xyz offsets relative to the palm, object A, and object B. Target joint vectors are stored alongside them. Each selected grasp has 32 seeded samples for each of four fingers, for 384 points total.

The following are deliberately absent: a scalar J, feature weights, normalization across heterogeneous units, a retained-capacity label, a success threshold, and a policy-facing finger gate.
