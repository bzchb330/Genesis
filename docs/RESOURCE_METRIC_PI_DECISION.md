# PI Worksheet: Resource Metric J

The engineering pass provides raw measurements and exploratory associations only. It does not define J. Ranges below describe the three selected grasps and 240 B probes; they are not operating limits.

| Possible component | Physical meaning and unit | Measured range / variation | Exploratory relationship with B probing | Sim-to-real measurability | Advantages | Limitations / PI decision |
|---|---|---|---|---|---|---|
| Free/non-A-contacting fingers | Mean number of four configured fingertips not contacting A [count] | 0.789–1.841 across grasps | Pearson r with minimum B distance -0.174; with A translation -0.454 | Contact/tactile sensors can estimate it, subject to detection quality | Simple, interpretable contact occupancy | A non-contacting finger may still be kinematically blocked; decide temporal/contact definition |
| Joint-range reserve | Minimum two-sided joint margin [rad] | 0.00355–0.16071; grasp 03 is close to one limit | r with B distance +0.382; with A translation +0.087 | Directly available from encoders and calibrated limits | Purely kinematic and unit-bearing | A single minimum discards distribution/direction; decide aggregation and relevant joints |
| Actuator reserve | Mean two-sided command reserve [N m] | 0.82894–0.94386 N m; mean utilization 0.0561–0.1711 | r with B distance +0.315; with A translation +0.442 | Requires current/torque estimation and calibrated limits | Exposes command headroom and saturation proximity | Command is not delivered joint torque; sign, dynamics, and aggregation require a decision |
| Fingertip reachable workspace | Sampled palm-relative bounding-box volume [m³] | 4.66e-6–1.22e-5 across 12 grasp/finger clouds; axis extents 0.0123–0.0405 m | r with B distance +0.268; with A translation +0.124 | Can be predicted from encoders/kinematics after calibration | Finger-specific geometric capability | Sample count, bounds, collision checks, dynamics, and volume estimator are protocol choices |
| B reachability/contact | Minimum fingertip-to-B signed distance [m] and physical contact [binary/count] | 0.107–0.170 m; 0 contacts in 240 probes | Contact correlation undefined because the indicator has zero variance | Distance needs object pose/model; contact can use tactile sensing | Directly tied to configured downstream geometry | Current placements lie outside sampled reach; decide whether this is admissible and how reach is defined |
| A retention disturbance | Translation [m], rotation [rad], vertical displacement [m], force redistribution [N] | translation 0.000831–0.011364; rotation 0.0290–0.6272; vertical -0.00911–+0.00421; redistribution 0.238–5.065 | Outcomes used above; no scalar disturbance score formed | Pose tracking plus tactile/force estimates; accuracy depends on sensing | Directly measures cost imposed on the held object | Horizon, aggregation, and any constraint/threshold remain PI-owned |
| Contact redundancy | Active A-contacting fingers [count], per-finger occupancy and force [N] | means 2.159–3.211; mean per-finger force 0–6.214 N; no complete loss in probes | Non-contact count relationships above; force distribution not collapsed | Multi-finger tactile sensing can estimate contacts/forces | Retains topology and load-sharing evidence | Contact count is not independent robustness; decide persistence and force treatment |

The correlations are repeated-probe descriptive statistics over only three grasp families. They do not establish causality, generalization, or a preferred component.

Before J can be implemented, the PI must decide:

1. What physical construct J represents and whether it is scalar, vector-valued, finger-specific, phase-specific, or object-specific.
2. Which raw quantities are admissible and whether privileged simulator state may be used during training, evaluation, or only analysis.
3. Units, normalization, directionality, aggregation across fingers/joints, and treatment of saturation or joint-limit proximity.
4. Whether contact topology, normal force, B contact, and A disturbance are inputs, constraints, or separately reported outcomes.
5. How reachability is sampled and whether collision-free or dynamically feasible reach is required.
6. Validation protocol, uncertainty reporting, and treatment of simulator contact penetration.
7. Whether and where J enters reward, termination, observation, evaluation, or controller logic, plus any weights or thresholds.

Until these decisions are supplied, `seqgrasp/env/resource.py::compute_resource_metric` returns `None`, reward resource weight remains zero, and all raw resource diagnostics remain descriptive.
