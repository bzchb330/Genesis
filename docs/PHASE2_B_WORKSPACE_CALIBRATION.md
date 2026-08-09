# Phase 2 B workspace calibration

## Preserved negative result

The original table-resting B distribution was geometrically legal but unreachable by the fixed-base Allegro hand while it retained A. The audited preflight found 0/100 B poses reachable for the representative A grasps. This was a fixed-base geometry limitation, not a controller-learning failure; no learning controller was involved. The former Part D run correctly stopped before collecting formal outcomes. Its raw preflight and negative figure remain in the experiment outputs and prior Git history.

## PI-authorized presentation design

Part D presents B temporarily in a reachable world-frame 3-D region. B is still the second free rigid object with its configured mass, friction, cylinder geometry, and MuJoCo contacts. No arm, third manipulated object, palm motion, or extra dynamic object was added. A kinematic free-joint pose fixture supports B through approach and closure, is removed at timestep 900, and is inactive for all 500 final-hold steps. The final hold is therefore unsupported.

## Geometry-only map

Calibration used the original 200 accepted A grasps and no second-grasp outcomes. For every grasp it reconstructed the occupied/free mask, generated 1,000 deterministic collision-filtered Monte Carlo free-fingertip workspace samples, and evaluated actual fingertip, palm, A-box, and B-cylinder geometry. A 392-centre common world-frame grid covered x `[-0.02, 0.12]` m, y `[0.00, 0.12]` m, and z `[0.10, 0.22]` m at 0.02 m spacing. Each grasp/centre record contains minimum free-tip-to-cylinder distance, reachable free-finger count, A collision, hand collision, and membership in the measured free-palm region.

The deterministic selection minimized initial overlap first and distance from a 50% population-accessibility centre second, subject to the PI-authorized 20%–80% engineering calibration range. It selected centre `[0.060, 0.120, 0.220]` m: 33.5% of the original A-grasp population was geometrically reachable, A overlap was 0%, hand overlap was 0.5%, and median minimum distance was 0.0151212 m. This choice used no dynamic outcomes.

The resulting globally fixed uniform box is:

- x: `[0.055, 0.065]` m
- y: `[0.115, 0.125]` m
- z: `[0.215, 0.225]` m
- cylinder axis: vertical
- yaw: uniform on `[0, 2*pi)` rad
- roll/pitch: zero

These exact bounds were written to `configs/phase2_physics_validation.yaml` and frozen before the pilot or formal outcome collection. They are identical for every A grasp and do not depend on occupied count, any resource value, Ferrari-Canny epsilon, or outcomes.

## Final deterministic geometry preflight

After the targeted dataset extension, 200 deterministic B poses were evaluated against 20 deterministically stratified A grasps spanning occupied-count categories, workspace quartiles, free-palm range, and epsilon range. Of 4,000 grasp-pose pairs, 1,439 were reachable (35.975%). Some B pose was reachable for 195/200 placements; 5/200 remained unreachable for all representatives. The maximum reachable free-finger count was one, the invalid-overlap pair fraction was 0.1%, and the minimum-distance distribution had minimum 0.00237122 m, median 0.0152423 m, and maximum representative median 0.0216195 m. The gate therefore passed: access was nonzero, challenging cases remained, and invalid penetration was not systematic.

The full ignored audit records are under `outputs/phase2/grasp_dataset/fcc7835446a4/correlation/calibration/` and `correlation/preflight/`.
