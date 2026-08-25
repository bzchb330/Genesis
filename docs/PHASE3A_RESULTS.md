# Phase 3A Results: Shadow Hand Contact Handoff

## Scope and provenance

Phase 3A runs on branch `codex/phase3A-shadow-hand-contact-handoff` from commit
`e5cf696605b4f5dd5a7e3d7d41287403ea157e3c`. The platform is the official
MuJoCo Menagerie right Shadow Hand E3M5 from repository
`https://github.com/google-deepmind/mujoco_menagerie.git`, vendored from commit
`c1a4eeb85694ae1dffe33ff1797d4e528928a133` under Apache-2.0.

The upstream MJCF, geometry, masses/inertias, joint topology/limits, wrist,
tendons, actuator structure, collision geometry, solver settings, and contact
parameters are unchanged. Runtime scene construction only assigns semantic names
to existing anonymous collision geoms and adds the diagnostic object, floor, and
temporary mocap weld fixture. The Allegro path is unchanged and has a regression
test.

## Platform audit

- Hand bodies: 25 excluding world; generated scene bodies: 28 including world,
  diagnostic object, and non-colliding mocap fixture anchor.
- Hand joints: 24; actuators: 20; fixed distal tendons: 4.
- Wrist: `rh_WRJ2`, `rh_WRJ1` (2 DOFs).
- Palm: `rh_palm` with eight upstream primitive collision geoms.
- Fingers: thumb, index, middle, ring, little.
- Each fingertip is the upstream distal posterior mesh collision geom.
- Runtime fingertip and palm `condim`: 3.
- Runtime fingertip and palm friction: `[1.0, 0.005, 0.0001]`.
- Upstream plastic contact parameters: `solref=[0.005, 1.0]` and
  `solimp=[0.5, 0.99, 0.0001, 0.5, 2.0]` after compilation.
- Solver: 0.002 s timestep, elliptic cone, `impratio=10`.

The complete per-joint, per-actuator, per-geom audit is in
`docs/PHASE3A_SHADOW_HAND_AUDIT.md` and its machine-readable companion is
`outputs/phase3A/shadow_hand_audit.json`.

## Diagnostic object and acquisition protocol

The object is the Menagerie scene's ellipsoid geometry and contact model:
size `[0.03, 0.04, 0.02]` m, compiled mass `0.10053096491487337` kg,
friction `[0.5, 0.01, 0.003]`, and `condim=6`. The seven deterministic cohort
poses are offsets about `[0.379, -0.040, 0.023]` m.

The object is temporarily welded to a mocap anchor during approach. Thumb and
index close independently. The first finger whose object normal force reaches
the configured diagnostic contact value (0.02 N) latches its current actuator
target; it does not continue blindly along the full closing trajectory. The
fixture is disabled only after closing and settling. Object qpos is never set
after fixture release.

The existing 0.003 m penetration value is used only as a diagnostic reference,
not redefined as a final Shadow-Hand scientific validity threshold.

## Minimal thumb-index cohort

Seven thumb-index-only attempts were run. Middle, ring, and little were not
commanded during this cohort.

| Classification | Count |
|---|---:|
| `THUMB_INDEX_SUCCESS` | 5 |
| `THUMB_INDEX_CONTACT_BUT_UNSTABLE` | 0 |
| `THUMB_INDEX_GEOMETRICALLY_INADEQUATE` | 0 |
| `EXCESSIVE_PENETRATION` | 0 |
| `CONTACT_LOSS` | 0 |
| `OBJECT_SLIP` | 2 |
| `OTHER` | 0 |

All seven states had thumb+index contact at release. Release penetration was:

- median: 0.0006632552638284937 m
- p95: 0.0007090892793086595 m
- maximum: 0.0007118674810662214 m

Thus every release was below the unchanged 0.003 m diagnostic reference. Five
states retained both thumb and index contact without floor contact for the
configured 250-step (0.5 s) unsupported diagnostic window. This is a diagnostic
classification, not a PI-approved final definition of acquisition success.

## Progressive middle recruitment

Only the two matched thumb-index slip cases triggered the second condition.
Middle contact was achieved in both recruited trials, but both still slipped:

- middle-recruitment attempts: 2
- retained with recruited support: 0
- improvement over the matched minimal condition: 0/2

Ring and little were not recruited in this matched comparison. No claim of
middle-finger benefit is supported by this cohort.

## Dynamic handoff demonstration

The handoff uses a separately recorded, valid thumb-index acquisition at object
position `[0.375, -0.036, 0.018]` m. After fixture release and an unsupported
thumb-index hold, middle is recruited first. Ring and little are commanded only
in a later support stage. All object motion after release is produced by MuJoCo
dynamics. No endpoint is initialized, no object teleport occurs, and no
kinematic object motion is used.

Raw outcomes:

- total dynamic object displacement: 0.044473320846964315 m
- progress toward the moving palm: 0.04018417677749049 m
- initial/final object-palm distance: 0.1336633871145841 / 0.0934792103370936 m
- palm contact achieved dynamically: yes
- maximum palm normal force: 1.4197131025399765 N
- maximum non-acquisition/palm support-load fraction: 0.5202257637952804
- configured acquisition finger released: thumb
- final contact mask `[thumb,index,middle,ring,little,palm]`:
  `[0, 1, 0, 0, 0, 1]`
- final normal forces `[thumb,index,middle,ring,little,palm]`:
  `[0.0, 0.6967483817765072, 0.0, 0.0, 0.0, 0.7516723113412745]` N
- final floor contact: no
- released thumb available-motion raw value: 2.187825864235061 rad
- diagnostic resource-recovered event: yes

The final object is retained by index+palm after the thumb is unloaded and
released. The palm carries the larger of the two final normal forces. This is a
real single-object contact handoff and one-finger resource recovery result; it
is not multi-object grasping and is not a final `PALMAR_SECURE` scientific
classification.

## State, control, and environment interfaces

Finger roles are `FREE`, `PROBING`, `ACQUIRING`, `SUPPORTING`, `TRANSFERRING`,
and `RELEASING`. High-level phases are `PROBE`, `MINIMAL_ACQUIRE`, `RECRUIT`,
`TRANSFER`, `PALMAR_SECURE`, `RELEASE_ACQUISITION_FINGERS`, and
`RESOURCE_RECOVERED`. Roles can change within an episode.

The six-entry support vector and normalized load fraction retain the semantic
order `[thumb,index,middle,ring,little,palm]` and are calculated from measured
contact normal forces. The five-entry free mask retains finger identity and is
accompanied by role, contact, force, normalized joint margin, tip-relative object
displacement, available joint motion, and a raw 3-axis first-order fingertip
workspace envelope computed from the current MuJoCo body Jacobian and joint-limit
margins. The local envelope is diagnostic and is not treated as a reachability
success threshold.

The variable-impedance action has 26 elements: 20 bounded desired actuator
displacements plus six bounded stiffness scales for wrist and five semantic
finger groups. Fixed-impedance compatibility remains available. No scientific
gain tuning was performed.

The Gymnasium actor observation has 127 values: hand q/qvel, explicit wrist
state, per-finger contact/normal force, palm contact/force, one-hot finger roles,
one-hot high-level phase, and previous action. A separate 50-value privileged
diagnostic dictionary provides object pose relative to palm, object velocity,
contact counts, support/load, penetration, and fingertip/palm geometric
relations. Reward components are individually measurable, but every weight is
zero and remains PI-owned.

## Artifacts

Machine-readable results:

- `outputs/phase3A/acquisition_and_recruitment.json`
- `outputs/phase3A/contact_handoff.json`
- `outputs/phase3A/shadow_hand_audit.json`

Figures are under `docs/figures/phase3A/` using the ten requested filenames.
Videos are `outputs/phase3A/videos/minimal_acquisition.mp4` and
`outputs/phase3A/videos/contact_handoff.mp4`. The six-frame handoff montage was
visually inspected and clearly shows the anthropomorphic robotic thumb, four
fingers, palm/wrist, moving ellipsoid, and released thumb. Poppler was not
available on the runner, so PDF rasterization itself was not used; the PDFs and
videos were generated from the same verified MuJoCo frames.

## Phase 3B recommendation

Do not start RL until the PI chooses the unresolved acquisition, secure-support,
support-transfer, loss/slip, meaningful-motion, penetration, and reward-weight
definitions. After those decisions, Phase 3B should train only the single-object
fingertip-to-palm handoff in `Phase3ShadowHandEnv`, using the 26-D bounded
variable-impedance action, the 127-D deployable actor observation, and the 50-D
privileged critic/diagnostic state. Validate release/hold behavior and recovered
finger motion out of sample before introducing a second object.
