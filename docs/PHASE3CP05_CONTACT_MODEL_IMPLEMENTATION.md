# Phase 3C-P0.5: implementation and interpretation

## Decision

No candidate was selected and `PHYSICS_V1_NEAR_RIGID` was **not created**.
The isolated regression passed exactly, but all six actual-hand starts hit
the frozen low-force engineering guards after 3-4 ms. They do not provide
the requested sustained simultaneous M/R/L evidence. Do not interpret this
as proof that both contact models are intrinsically unstable.

## Versions and isolation

`configs/contact_physics_registry.yaml` contains exactly these versions:

| Version | solref | solimp |
|---|---|---|
| LEGACY_PHASE3C_CONTACT_PHYSICS | [0.02, 1] | [0.9, 0.95, 0.001, 0.5, 2] |
| PHYSICS_CANDIDATE_IMP99 | [0.02, 1] | [0.99, 0.99, 0.001, 0.5, 2] |
| PHYSICS_CANDIDATE_TC10_IMP99 | [0.01, 1] | [0.99, 0.99, 0.001, 0.5, 2] |

`seqgrasp/contact_physics.py::pair_transform` adds 31 explicit sphere-hand
contact pairs after the existing forearm transform. Collision geom names
come from the existing semantic naming function. Coverage: palm 8, thumb 6,
index 4, middle 4, ring 4, little 5. No existing asset/MJCF is overwritten.
The inherited base configuration is `configs/phase3A_shadow_hand.yaml` via
`phase3c07_scene_config`; the sphere is 25 mm in diameter, mass
0.00818123086872342 kg. The existing forearm actuator is unchanged.

Each pair explicitly has effective dimension 6, friction
[0.5, 0.5, 0.01, 0.003, 0.003], margin 0 and gap 0. Compiled
`pair_solreffriction` remains [0, 0], using the normal reference defaults;
it is not separately tuned. Native geom friction remains sphere
[0.5, 0.01, 0.003] and hand [1, 0.005, 0.0001]. The sphere's priority 1
versus hand priority 0 already made the Legacy dynamic hand-object contact
6D with this effective friction. Actual candidate runtime contacts confirm
the specified solref, solimp, friction and dim.

Unrelated self/environment interactions are not explicit-pair targets.
The native compiled-property fingerprint is identical for Legacy and both
candidates: geometry, mass/inertia, geom parameters, joint ranges, damping,
actuator gains/dynamics/force limits and production solver/timestep.
Analytic native joint/fixed-tendon transmission matrices are identical.
M/R/L distal actuators remain coupled J2+J1 tendons, not independent servos.

Production options remain dt 0.002 s, Newton, 100 iterations, tolerance
1e-8, Euler, elliptic cone and impratio 10. Diagnostic dt values are
0.001/0.002/0.004 s, explicitly labeled in every trial. No candidate
changed friction, geometry, condim, gains, force ranges, joint limits or skin.

MuJoCo documentation explains explicit-pair precedence and the numerical
solver parameterization. These are numerical contact settings, **not** a
claim of material Young's modulus or Poisson ratio:
[contact pair reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair),
[solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).

## Frozen protocol and reproducibility

`configs/phase3CP05_near_rigid_selection.yaml` was captured in
`outputs/phase3CP05/frozen_protocol.json` before any candidate dynamics.
The snapshot contains the entire registry, diagnostic config, geometry and
command hashes, base commit and hashes of every existing P0 output.
`validate_frozen` rejects changed experimental inputs. The runner reuses
existing per-trial results, rather than silently rerunning them.

Regression reproduces only the four requested loads for each candidate,
with the original P0 0.4 s ramp, 4 s hold and final 200-sample statistics.
All eight penetration differences from P0 were exactly 0 m. NPZ traces
identify active physics without requiring pickle. Summary JSON identifies
every candidate, timestep and trace.

Dynamic drops are gravity-only, with no weight cancellation: 0.5, 1, 2 and
5 mm heights, for one physical second each at 1/2/4 ms, for 24 trials.
The original sphere/plane parameters are copied from P0 and only the
candidate sphere solref/solimp changes the effective isolated pair.
No TC10-alone trial was run. Full-second impulse includes static weight
support; a separate first-100-ms-after-impact impulse is reported.

Contact events use positive normal force > 1e-9 N as a numerical-zero
convention, not a success criterion. A 20-ms bin describes rapid event
intervals; it is not a publication threshold. Each drop had one make and
zero breaks. The final contact episode is right-censored, not assumed to
end at logging completion. Positive upward recovery velocity under ongoing
contact is distinguished from a detached rebound.

Energy logs contain translational/rotational kinetic energy, gravitational
potential, contact work from trapezoidal sampled power, and
`E(t)-E(0)-W_contact`. Contact work and residual are numerical diagnostics,
not exact energy conservation statements. Mechanical energy never exceeded
initial energy in these runs, but it need not be monotonic while numerical
deformation relaxes. Residuals are appreciable and timestep-dependent;
they are not concealed as zero or calibrated material damping.

## Actual-hand construction and failed coverage

One bounded local kinematic least-squares fit constructed the three distal
clearances. It was not a receiver search and used no candidate dynamic
outcomes. An initial implementation omitted finger-finger clearance;
its self-colliding static result was rejected and retained as
`geometry_rejected_self_collision.json`. No dynamics were run from it.
The corrected fit explicitly penalized enabled self-collisions and passed
zero-contact validation before being frozen as `geometry.json`.

Final sphere center in palm coordinates (m):
[-0.021394437024792462, -0.052937729198662964, 0.1567752619249067].
M/R/L distal gaps: 40.010509, 40.008747, 39.996138 micrometres.
Thumb/index, wrist and forearm start at zero joint position. All initial
hand-object gaps are positive and there are no enabled self contacts.
Complete joint names, poses, mocap transforms, closest points, Jacobian
directions and actuator targets are in `geometry.json`.

The sphere stays welded throughout. Temporary external support uses solref
[0.008, 1], solimp [0.9999, 0.9999, 0.001, 0.5, 2], identically for all
trials. The time constant is at least twice the largest diagnostic timestep.
This diagnostic fixture is not production hand-object physics and was
explicitly declared before outcomes. No weld release was executed.

Native position servos start at their current actuator lengths; the target
is a simultaneous 0.015 maximum actuator-coordinate increment along the
normal-Jacobian/transmission pseudoinverse directions. Tendon coordinates
are summed angles, not physical tendon lengths in metres. The cubic ramp
is 1 s and hold 2 s, identical in physical time across candidates/dt.
This is a gentle **requested** schedule, not a validated low-force regime.
No old 0.4 virtual offset or deep ROLE_MRL_05 state is used.

Predeclared engineering stops are total normal force >1 N, individual
contact force >0.5 N, object overlap >1 mm, fixture displacement >2 um,
generalized speed >25 (hand rad/s; free-body translation remains near zero),
nonfinite state, or actuator saturation. These stop escalation; they are
not scientific admissibility/publication cutoffs. All six trials stopped
on force, not on changed penetration criteria. The commanded offset was
only 4.0419e-7 at 3 ms or 7.1808e-7 at 4 ms when stopped.

At 1 ms, little was the only loaded finger. At 2/4 ms, ring and little
appeared together in the final snapshot. Middle never loaded. Native zero
servo error is not static equilibrium under gravity/passive/constraint
dynamics; the no-step `startup_audit.json` records initial acceleration,
bias, passive, actuator and constraint forces. Attribution to contact
instability alone would therefore be unjustified.

Every hand force stop occurred at the first positive-contact snapshot,
after `mj_forward` calculated the contact response. There was no subsequent
loaded integration step. Candidate trajectories through that point are
therefore identical; the different first-contact force estimates are not
a sustained actuated-contact dynamics comparison. Initial maximum hand
acceleration was 81.61685063369048 rad/s^2 for both candidates, despite
zero initial servo forces. The prescribed ramp had barely begun.

No trial reached the planned 3 s duration. Steady penetration and steady
force are **null**. Observed-window variance and mean describe only the
censored startup. The first analysis incorrectly labeled startup means
as steady; this reporting error was corrected by reanalyzing the unchanged
raw traces, without rerunning dynamics. Each loaded pair has one active
snapshot, so computed migration of zero has no continuity interpretation.
Contact durations have right-censoring rather than an invented extra step.

`native.record` retains all object contacts, 6D local wrenches, world
forces/moments, normals, distances, identities, positions, qpos/qvel/ctrl,
actuator force/error/saturation, tendon state and fixture wrench each step.
`solver_stats` records warnings and maximum iteration count across islands;
its gradient/improvement fields are first-island diagnostics, not a global
convergence certificate. Native hand/environment contacts are not altered.

## Selection and remaining PI decisions

The softer-adequate preference supplied by the PI remains the selection
principle; smaller static overlap alone is not a selection rule. IMP99
static overlap is 0.03924-0.244462 mm over the four loads, but its impact
peak reaches 2.143187 mm at dt 2 ms in the 5-mm drop. TC10_IMP99 reduces
that to 0.978979 mm while raising peak force from 0.343166 to 0.620647 N.
These are tradeoffs, not proof of task-independent dominance.

Dynamic penetration is timestep-sensitive, particularly when a small drop
has only a few integration steps before contact. Peak overlap (mm), in
height order 0.5/1/2/5 mm:

| Candidate | dt 1 ms | dt 2 ms | dt 4 ms |
|---|---|---|---|
| IMP99 | 0.694868 / 0.950009 / 1.365449 / 2.240272 | 0.664400 / 0.907326 / 1.304573 / 2.143187 | 0.889063 / 1.167117 / 1.180922 / 1.945937 |
| TC10_IMP99 | 0.326143 / 0.436806 / 0.635660 / 1.077486 | 0.295230 / 0.393410 / 0.573783 / 0.978979 | 0.466824 / 0.605602 / 0.456226 / 0.799176 |

Thus static timestep reproducibility from P0 must not be generalized to
impact overlap. No impact conditions were retuned to suppress this effect.

`freeze_version` refuses incomplete or failed selection evidence, refuses
overwriting an existing V1, and would store all required settings plus the
source candidate and content hash if authorized by a passed comparison.
`require_production_alias` rejects diagnostic candidates for future
production use. `assert_locked_model` checks the compiled contact and
numerical settings. Tests exercise hypothetical locks only in temporary
test directories; no workspace production V1 exists.

TODO(PI): approve the next diagnostic initialization/control protocol after
reviewing these censored startup results. Do not silently raise force guards,
increase preloads, settle/reposition the object or retune candidates.

TODO(PI): if a specific compliant-material interpretation is later required,
supply material properties and use a separately versioned ablation; the
present numerical near-rigid assumption does not supply E/nu.

Optional tangential/sliding/spin tests were gated out because Parts C-D
did not pass. P0.6 bounded-force control was not implemented. Receiver
reconstruction, B03, ROLE_MRL_05 release, settling, handoff, shape, skin,
RL and object B remain gated. The old 6.598903-mm settled / 8.333245-mm
peak deep state remains invalid regression evidence, never a receiver.

## Preservation and outputs

P0 commit: `6449e1a725247a461a38c3ab11d1bede28199aea`, pushed to
`codex/phase3CP0-contact-physics-validation`. P0.5 branch
`codex/phase3CP05-near-rigid-contact-selection` was created/pushed from it.
All P0.5 source, reports and figures remain intentionally uncommitted.
No merge to main occurred. All P0 output hashes are checked again by tests.

Local raw artifacts remain under `outputs/phase3CP05/`, which `.gitignore`
excludes, along with videos, caches and environments. Twenty vector PDFs
are under `docs/figures/phase3CP05/`. Two saved-state impact videos show
the 5-mm drops at explicitly labeled 10x slow motion. No misleading hand
video is made from 1-3 integration steps. Rendering takes zero physics steps.

Run commands (all use only the repository virtual environment):

```powershell
.\.venv\Scripts\python.exe scripts/run_phase3cp05.py
.\.venv\Scripts\python.exe scripts/analyze_phase3cp05.py
.\.venv\Scripts\python.exe scripts/plot_phase3cp05.py
.\.venv\Scripts\python.exe scripts/generate_phase3cp05_videos.py
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

The existing-result runner does not retune or rerun completed trials.
This phase is a documented **no-freeze stop result**, not completion of
the intended sustained multi-contact validation objective.

## Final validation

`.\.venv\Scripts\python.exe -m pytest -v`: **452 passed, 7 warnings in
40.50s**, exit 0. This includes 47 new P0.5 tests. Warnings: six existing
Gymnasium infinite Box-bound notices and one existing pytest cache
permission warning (WinError 5). No test failed or was skipped.

`git diff --check`: exit 0, no output. New untracked text files also received
individual no-index whitespace checks; Git's status 1 there denotes the
new-file differences, with no whitespace diagnostics. Source and figure
files are intentionally new/untracked, not staged. Twenty PDFs were
rendered and visually checked; the topology plot uses sampled step states.
Two diagnostic videos were inspected at beginning/contact/end frames.
All 307 preserved P0 output hashes remain identical.
