# Object-A Baseline Failure Mechanism

## Scope

This report analyzes the engineering reference profile `grasp_A_01` over deterministic placement seeds 0–9. It explains the observed mechanics without defining a scientific grasp-success or drop criterion.

## Contact geometry at support release

All ten runs have the same object-contact pattern on the first unsupported sample: middle and thumb contact object A; index and ring do not. The index tactile channel reports force, but contact-pair inspection shows that this is a non-object fingertip contact. Index never contacts object A anywhere in the baseline seed-0 trajectory. Ring is absent at release and only contacts later as the moving object passes it.

For seed 0, the object-contact centroids immediately after release are:

| Finger | Contact position xyz (m) | Inward normal xyz | Object normal force (N) |
|---|---|---|---:|
| middle | `[0.08301, 0.00004, 0.18480]` | `[-0.00031, 0.00270, -0.999996]` | 5.836 |
| thumb | `[0.07946, 0.04204, 0.15721]` | `[0.00011, -0.999996, -0.00270]` | 4.558 |

The middle contact is on the cube's upper face and points almost vertically downward into the object. The thumb contact is on a lateral face and points almost entirely in negative y. Their normal dot product is approximately zero: the contacts are orthogonal, not opposed. Neither contact supplies an upward-facing normal that can directly balance gravity, and there is no opposed pinch across the cube.

## Post-release sequence

1. The kinematic support is disabled and the object free joint evolves under gravity and normal MuJoCo contact dynamics.
2. The desired joint command remains at the configured closed target; there is no continued closing trajectory or active tactile correction after release.
3. The middle/object contact disappears 0.032–0.044 s after release across seeds. The thumb is then the only release-time contact still acting on the object.
4. The object translates downward and rotates while sliding past the lateral thumb contact. Later index/ring contacts are transient consequences of that motion, not the initial force-closure set.
5. Complete configured object-fingertip contact loss occurs when thumb contact disappears, 0.534–0.892 s after release.
6. Explicit object/table contact occurs in nine runs between 0.646 s and 0.994 s. One run does not register table contact inside the fixed window but still moves downward substantially.

Final vertical displacement ranges from -0.14070 m to -0.11757 m over the ten runs. The motion is therefore not a stationary pinch that later fails abruptly: the object begins falling/sliding immediately, loses the top contact first, rotates through the hand, and then loses the lateral thumb contact.

## Controls, limits, and geometry

- No actuator reaches the configured ±1 N m control bound in any baseline run.
- After initializing the diagnostic directly at its configured open pose, no joint-limit excess is observed.
- The post-release target remains fixed rather than continuing to close.
- Object A is a 50 mm cube centered near `[0.07, 0.02, 0.165]` m during fixture-held contact establishment. At the baseline posture, this pose places the middle fingertip above the cube and the thumb on a side face, while index and ring do not establish object contact.

The baseline failure is therefore attributable to the contact geometry and its evolution—not actuator saturation, a joint limit, or termination logic. The release-time set lacks opposed normals and quickly loses its only upper contact, leaving a lateral contact that cannot balance gravity.
