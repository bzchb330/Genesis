# Phase 2 PI Inputs Required

The repository audit found no PI-approved values for the thresholds and parameter ranges below. Phase 1 diagnostic windows, search penalties, and descriptive measurements are explicitly marked engineering-only and are not promoted to Phase 2 scientific criteria.

## Fixed-base interpretation and reused inputs

The palm pose may vary between future candidate trials but remains fixed within each trial. Part A reuses the validated external-support removal mechanism as the fixed-base equivalent of lift/clearance; it does not translate the palm or introduce an arm. The replay trajectory is the committed `resource_grasp_A_02` profile because Phase 1 identifies source 02 as a lower-penetration, multi-digit retained reference. This is a fixed replay selection, not a physics-parameter or success-threshold decision.

The existing 0.002 s timestep, object geometry/mass, gravity, mechanical workspace bounds, and controller limits are reusable baseline mechanics. The Phase 2 brief directly supplies the 1 N order-of-magnitude reference, factor 10 sanity band, and minimum 1000-step long hold.

## Consolidated missing values

| Parameter | Unit/type | Required in | Decision affected | Interpretation requested by the brief |
|---|---|---|---|---|
| Maximum valid penetration | m | Part A gate, Part D INVALID | Physical plausibility and invalid trials | Largest allowed negative geom distance; no value is proposed |
| Maximum vertical drift | m | Part A long-hold gate | Stable unsupported hold | Allowed absolute z change during configured long hold |
| Maximum translational drift | m | Part A/B stability | Stable hold and grasp acceptance | Allowed 3-D object translation |
| Maximum orientation drift | rad | Part A/B stability | Stable hold and grasp acceptance | Allowed quaternion angular change |
| Minimum active object contacts | count | Part A gate | Contact stability | Minimum object-hand contacts during hold |
| Table re-contact policy | boolean | Part A/B/D | Retention classification | Whether any table contact can pass |
| Complete contact-loss policy | boolean | Part A/B/D | Retention classification | Whether temporary zero object-hand contact can pass |
| Sweep target geoms | names | Part A sweep | Scope of friction/solver changes | Exact geoms whose compiled contact parameters are varied |
| Friction sweep vectors | slide/spin/roll coefficients | Part A sweep | Sensitivity evidence | Explicit vectors; no range is inferred from baseline friction |
| `solref` sweep | 2-vector | Part A sweep | Contact solver sensitivity | Explicit MuJoCo values |
| `solimp` sweep | 5-vector | Part A sweep | Contact impedance sensitivity | Explicit MuJoCo values |
| Timestep sweep | s | Part A sweep | Integration sensitivity | Explicit values around or including baseline as PI chooses |
| Load-bearing finger threshold | N | Part C1 | Occupied-finger mask/count | Summed per-finger normal force required for occupancy |
| Tactile binary threshold | N | Part E1 | Binary tactile feature | Summed normal force required for contact flag |
| Short-hold drift tolerance | m, plus any rotation criterion | Part B2 | Accepted grasp filtering | Stability tolerance during dataset hold |
| Acquisition criterion/thresholds | specified raw units | Part B/D | A/B acquired state | Required contacts, force, clearance, and persistence |
| Retained-object criterion/thresholds | specified raw units | Part D | BOTH_RETAINED and B_NOT_ACQUIRED | End-of-hold retention evidence |
| Loss/drop criterion/thresholds | specified raw units | Part D | A_DROPPED and BOTH_LOST | Loss evidence and persistence |
| B placement lower/upper bounds | world xyz m and orientation distribution | Part D2 | Correlation trial distribution | Explicit reachable Phase 2 B workspace, replacing invalid exploratory bounds |
| Workspace production sample count | count | Part C2 | Monte Carlo volume precision | To be selected after the requested convergence study |
| Workspace voxel size | m | Part C2 | Reachable-volume resolution | Side length for occupied-point voxels |
| Workspace collision tolerance | m | Part C2 | Sample rejection | Clearance used against A and occupied-finger geoms |
| Free-palm voxel-box bounds | palm-frame xyz m | Part C3 | Region being measured | Axis-aligned lower and upper bounds |
| Free-palm voxel size | m | Part C3 | Free-volume resolution | Voxel side length |
| Second-grasp trials per A grasp | count | Part D4 | Correlation sample size | Deterministic B placements per accepted grasp |

The accepted-grasp target is configured as 200, the lower bound explicitly allowed by the Phase 2 brief. It is not a filtering threshold.

Until the Part A gate values and sweep ranges are supplied, the gate verdict must remain `PI_INPUT_REQUIRED`; Parts B-F scientific execution is prohibited.
