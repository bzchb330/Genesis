# Phase 3C-P0.5R: dynamic-equilibrium reset audit

## Decision

The production-2-ms dynamic reset is repaired and independently confirmed.
The requested physics-selection evidence is **not complete**: both candidates
have sustained ring/little contact, but no middle contact, at 1 and 2 ms. Both
4-ms runs are rejected by the unchanged pre-hold speed gate before contact.
No PHYSICS_V1_NEAR_RIGID was created. No scientific criterion was relaxed.
This is a numerical contact diagnostic, not a grasp-success or material claim.

Branch: `codex/phase3CP05R-dynamic-equilibrium-reset`.
Preserved P0.5 base: `5ec8582a97b2e0ee1866bccc3e983f2522d61ae0`.
The P0.5 commit and new branch were pushed; P0.5R remains uncommitted.
All 88 pre-existing P0.5 output files are SHA-256 checked against the manifest
frozen before this phase. No main merge occurred.

## Frozen physics and scope

| Parameter | IMP99 | TC10_IMP99 |
|---|---|---|
| solref | [0.02, 1] | [0.01, 1] |
| solimp | [0.99, 0.99, 0.001, 0.5, 2] | same |
| Effective explicit-pair friction | [0.5, 0.5, 0.01, 0.003, 0.003] | same |
| Effective condim | 6 | 6 |
| Production timestep | 0.002 s | same |
| Solver / integration | Newton / Euler | same |
| Iterations / tolerance | 100 / 1e-8 | same |
| Cone / impratio | elliptic / 10 | same |

Hand geom friction remains [1, 0.005, 0.0001]; sphere geom friction remains
[0.5, 0.01, 0.003]. Actuator gains, force ranges, tendon definitions, joint
limits, gravity and passive damping were unchanged. Legacy remains historical,
not an admissible production candidate. Runtime: MuJoCo 3.11.0.

No receiver, B03, release, handoff, natural sphere settling, object B, RL,
shape, skin, storage search or bounded-force primitive was run. Sphere-position
fitting was a local collision-clearance construction with fixed hand FK, not
trajectory optimization. No candidate-specific command tuning occurred.

## Reproduced defect and force balance

`seqgrasp/phase3cp05.py::setup_hand` supplies the exact nominal state.
`seqgrasp/phase3cp05r.py::prepare_equilibrium` collision-disables the sphere
without changing hand qpos/qvel/ctrl/act or hand state indices. At the first
`mj_forward`, both candidates reproduce:

- max actuated-hand |qacc| = 81.61685063369048 rad/s^2;
- max |qvel| = 0; all 21 actuator forces and servo errors = 0;
- native fixed-tendon lengths (I/M/R/L) =
  [0, 0.6677638338798874, 0.6025770575227613, 0.7580477586316303] rad;
- tendon velocities and actuator-transmitted tendon forces = 0;
- max hand net generalized torque = 0.4963411960835348 Nm;
- max hand bias torque = 0.5063411960835348 Nm. At zero velocity, the
  velocity-dependent bias terms vanish; gravity is not balanced by zero servo
  torque. Passive and constraint terms are separately recorded.

`dynamic_reset.raw_diagnostic` reports qfrc_passive, qfrc_bias, qfrc_actuator,
qfrc_constraint, qfrc_smooth + qfrc_constraint, M*qacc, and their residual.
The dynamic-equation residual is tiny while the net torque is nonzero:
equation consistency is not static equilibrium. Native J0 tendons constrain
the summed J2+J1 transmission, not each distal angle individually.

Sphere disabling requires handling explicit pairs as well as geom masks.
Within the diagnostic context only, sphere contype/conaffinity are zero and
related pair margin/gap are -1/+1. Active object constraints are rejected at
every settling step; harmless geometric candidates with efc_address=-1 are
distinguished from active constraints. Parameters are restored exactly on exit.
The fixed sphere support and state slots remain, decoupled from hand contact.
This uses the installed 3.11 margin/gap semantics, documented in the
[MuJoCo computation reference](https://mujoco.readthedocs.io/en/latest/computation/).
No such disabling enters a production candidate.

## Natural settling and complete-state cache

`seqgrasp/dynamic_reset.py::settle_hand_to_dynamic_equilibrium` takes model,
data, complete nominal integration state, original ctrl, physics identifier,
maximum duration and explicit engineering gates. It performs natural steps
under constant original targets and damping. It returns complete integration
state, qpos/qvel/ctrl/act/qacc, tendon/actuator diagnostics, history and hashes.
`restore_equilibrium` checks convergence, state/model hashes and redundant state
metadata; `guard_dynamic_startup` exposes and checks raw speed/acceleration.
No qvel overwrites, temporary damping, gain changes or ctrl recentering were used.

Frozen ENGINEERING_DIAGNOSTIC_ONLY gates: max actuated-joint speed <=0.001
rad/s and acceleration <=0.5 rad/s^2 continuously for 0.5 s. These are not
publication thresholds. The first bounded 12-s scout did not converge (speed
0.00823674 rad/s): slow opposite-sign distal-joint drift remained despite nearly
settled tendon sums. Its failed cache was preserved, not relabeled.

Before continuation, `natural_continuation_protocol.json` froze a maximum
additional 108 s (120 s total), unchanged gates/targets/damping. Both candidates
converged after **77.784 s total natural settling**. The complete integration
states are bitwise identical. Each restored cache then passed a separate 0.5-s
no-object confirmation under original production dynamics.

- Cached and confirmation maximum speed: 0.0003677197442013351 rad/s.
- Cached and confirmation maximum acceleration: 0.0010935982776605572 rad/s^2.
- Maximum nominal-to-equilibrium joint displacement: 0.4998960959097237 rad.
- Maximum final hand net generalized torque: 2.212121667637823e-7 Nm.
- Maximum actuator holding force: 0.5152313860793998 Nm; not saturated.
- Native tendon lengths (I/M/R/L): [-0.0009521941886824081,
  0.6656113240942627, 0.6012727143249809, 0.7569812464233154] rad.
- Tendon velocities: [-1.3252456102893944e-11, -2.578586985277328e-5,
  0.00034731914970676214, -1.052591142589815e-9] rad/s.
- Actuator-transmitted tendon forces: [0.00047609709434120404,
  0.0010762548928123228, 0.0006521715988901944, 0.0005332561041574424].
  These are generalized fixed-tendon forces, not measured cable tensions.

The original 21 ctrl coordinates are preserved exactly; act is empty for this
native actuator model. Full ctrl, realized coordinates, errors and forces are in
each confirmed cache, not rounded/reconstructed from this document. Maximum
absolute ctrl-minus-realized error is 0.06440392325992497 rad. A no-integration
counterfactual setting ctrl=actuator_length produces max qacc=81.8172439857407
rad/s^2. This was diagnostic evaluation only and never used for an experiment.

Cache key:
`6043065f8360d5f5ef38542d93f303d61d2a27007a3aca8f50dacb1832eedd5b`.
Integration-state SHA-256:
`ba5cf8a3a9e18ec9f8eb69ebd32a33f79c0a27b9e103efe49fa73cab51d07ca1`.
Metadata includes the continuation-start state hash and original nominal pose
hash, compiled hand/controller fingerprint, gravity and numerical settings.
This fingerprint is an explicit array/option schema, not a serialized MJB hash.

Offline sensitivity on the saved continuation: doubled speed/acceleration gates
confirm at 77.436 s; baseline confirms at 77.784 s. Half-sized gates do not
accumulate a full confirmation window before the record ends (right-censored,
not proof that tighter convergence is impossible). No sensitivity simulations.

## Settled geometry and common protocol

Exact cached-state FK gives maximum M/R/L collision-geom displacements of
22.1346447993 / 23.5292341666 / 25.0387497242 mm. Geom positions/orientations
and palm transforms are stored alongside joint/tendon changes. The historical
approximately 40-micrometre distal gaps are preserved only for reproduction.

The first equal-three-gap static fit was infeasible and is retained separately.
The accepted local sphere-translation fit instead constrains minimum clearance
to **every hand geom >=0.4 mm**, with the settled hand unchanged. No dynamic
results were used to choose the placement. No 0.3/0.5-mm optional runs were made.
Accepted sphere center in the palm frame:
[-0.02347385963419607, -0.05105673634264201, 0.16856914292230854] m.
M/R/L distal signed gaps: 0.832937175558 / 0.400000000000 / 0.865136633087 mm.
Thus this is not an assertion of three equal 0.4-mm fingertip gaps.

The sphere begins at zero velocity, without hand contact or penetration. Weld
solref=[0.008,1], solimp=[0.9999,0.9999,0.001,0.5,2] are common to all trials.
Prehold fixture force max=0.08025909661239745 N, approximately the sphere weight
0.08025787482217676 N, not a first-contact impulse.

Before contact outcomes, the approximately 0.25-s prehold was rounded to 0.252 s
on the common 4-ms grid. All planned trials therefore use **0.252 s prehold +
1 s cubic ramp + 2 s hold = 3.252 s**, with constant nominal dt throughout.
Frozen maximum native virtual-direction command offset=0.015; settled-FK
directions and final targets are identical across both candidates and all dt.
No command was increased after observing missing middle contact.

Common initial-state SHA:
`226525e9793ee94d4fd41cd7234f9291f61c5644ad099b8c1fadaef73f3cf209`.
Common command SHA:
`fc93cb595df8b54dca7be5daf6b15f4d252b141e3ea8647998f374e1325f72e8`.

Soft warnings retain 0.5-N single / 1-N total. Frozen catastrophic guards are
2-N single, 3-N total, 3-mm penetration, 10-micrometre fixture translation,
25-rad/s hand speed, actuator utilization >=1 and nonfinite state. The 3-N cap
is about 37 sphere weights, not a target/admissibility load; it allows the prior
isolated impact (<0.647 N) while remaining below the invalid 6-N receiver
regime. Actual actuator saturation is independently guarded. None of the full
runs reached a force warning or catastrophic limit.

## Results and censoring

All 1/2-ms runs complete the ramp and hold with subsequent loaded integration.
At production 2 ms, onset is 1.120 s; little joins ring at 1.132 s. Each run
has 1,626 integration steps, including 1,066 after first contact. Ring/little
remain active for the final 500 ms; middle never engages. The following are
valid descriptive **R/L** statistics, not validated simultaneous-M/R/L scores.
Settled penetration is the final-500-ms mean of per-sample maximum penetration.

| Production 2-ms descriptor | IMP99 | TC10_IMP99 |
|---|---:|---:|
| Peak penetration, mm | 0.054553187279 | 0.017492306606 |
| Settled penetration, mm | 0.051756497026 | 0.015366329548 |
| Settled delta/R | 0.004140519762 | 0.001229306364 |
| Peak total normal load, N | 0.197803824935 | 0.256666815542 |
| Peak single normal force, N | 0.100479607004 | 0.130597737014 |
| Final-500-ms total force mean, N | 0.174982203996 | 0.201312395410 |
| Final-500-ms total force variance, N^2 | 5.96308100226e-6 | 3.19176572079e-6 |
| Makes / breaks | 2 / 0 | 2 / 0 |
| Tail R/L persistence; M persistence | 1/1; 0 | 1/1; 0 |
| Tail contact-count variance | 0 | 0 |
| Tail R/L position migration, micrometres | 18.863982301 / 1.311790163 | 14.070483109 / 1.507747170 |
| Tail R/L normal migration, degrees | 0.086343893 / 0.004630600 | 0.064245846 / 0.006562076 |
| Maximum actuator utilization | 0.103068840827 | 0.103068840827 |
| Maximum solver iterations; warnings | 5; 0 | 6; 0 |
| Maximum fixture displacement, micrometres | 0.310310695 | 0.376917031 |

Contact migration is displacement/angle from the first matching pair sample
within the final-500-ms window, not accumulated path length. Contact episodes
are right-censored at trial end, not artificial break events. Raw force traces
show variability despite no make/break chatter; absence of contact breaks does
not imply zero force variation. All per-actuator variances and fixture vectors
are in summary.json and the complete timestep logs.

Mean production fixture wrench, world-frame force N / torque Nm:

- IMP99: [-0.1420978030665, -0.0244357214961, 0.1000411606115] /
  [0.0002095043413, -0.0002827182121, 0.0002419347714].
- TC10: [-0.1655628739966, -0.0298599622904, 0.0951419929044] /
  [-0.0000733538597, -0.0004548598104, 0.0004404351652].

| Candidate | dt, ms | Complete | Peak penetration, mm | Tail mean force, N | Tail force variance, N^2 |
|---|---:|---|---:|---:|---:|
| IMP99 | 1 | yes, R/L | 0.055271951351 | 0.175795934354 | 6.37810538270e-6 |
| IMP99 | 2 | yes, R/L | 0.054553187279 | 0.174982203996 | 5.96308100226e-6 |
| IMP99 | 4 | no, prehold | unavailable loaded value | unavailable | unavailable |
| TC10 | 1 | yes, R/L | 0.017609437640 | 0.209744878748 | 1.33099643859e-5 |
| TC10 | 2 | yes, R/L | 0.017492306606 | 0.201312395410 | 3.19176572079e-6 |
| TC10 | 4 | no, prehold | unavailable loaded value | unavailable | unavailable |

Both 4-ms runs stop at 0.008 s: max speed=0.001201184493293574 rad/s,
max acceleration=0.2797162134245838 rad/s^2. There is no sphere contact, ramp,
loaded integration or steady descriptor. This is common-state timestep/startup
incompatibility, not evidence that either candidate is unstable under load.
The runner reports each candidate/dt rejection individually; spontaneous
prehold contact would stop the overall comparison. No prehold contact occurred.

Across 1/2 ms the same R/L topology persists with two makes and no breaks.
TC10 produces smaller overlap and higher total load, but its force variance is
more timestep-dependent. The unavailable 4-ms load response prevents a complete
timestep-robustness conclusion. No near-rigid acceptance threshold is invented.

## Interpretation, historical impact and next authorization

The startup defect explains why zero nominal servo error was not an acceptable
dynamic reset. Old P0.5 first-contact force peaks were startup-censored; the
repaired comparison also changes sphere geometry and approach directions, so
the old/new force difference is not a one-factor causal ablation.

Historical dynamic trials using the same unconfirmed nominal/reset convention
are potentially contaminated, not automatically disproven: B03 direct placement,
C08 fly-by, preload/receiver/release/transport rollouts that share that convention
need targeted provenance checks and eventual regressions. Old isolated contact
benchmarks do not become invalid merely because the actual-hand reset was wrong.
Geometric C-space connectivity, workspace/orientation reachability and
fixed-network static wrench analysis remain valid within their static assumptions.
Frozen resource fractions remain thumb=0.9559782183972225, index=1.0,
opposition=0.9665998246424643; they were not recomputed.

No candidate is selected and no production V1 alias is unlocked. Candidate
parameter locks remain in force. The next step is **PI review of missing middle
contact and 4-ms common-state prehold compatibility**, followed only by an
authorized bounded diagnostic protocol. No new gap, ramp amplitude or gate is
chosen here. If and only if V1 is later selected, plan the 12-state B03 direct-
placement and C08 4.225637-mm fly-by regressions with equilibrium-based resets
and unchanged historical outcome criteria. They have not run in P0.5R.
Bounded-force primitives, receivers, handoff, shape, skin, object B and RL remain
gated. The reset is solved for this nominal pose at production dt, not for every
possible hand pose or numerical timestep.

## Artifacts, reproduction and validation

All Python executions use `.\.venv\Scripts\python.exe`.
Entry points: `scripts/run_phase3cp05r_reset.py`,
`scripts/continue_phase3cp05r_settling.py`,
`scripts/run_phase3cp05r_comparison.py`, `scripts/analyze_phase3cp05r.py`,
`scripts/plot_phase3cp05r.py`, `scripts/generate_phase3cp05r_videos.py`.
Existing results are cached: do not delete caches to launch a new experiment
without PI authorization. Plotting and videos use saved states and mj_forward,
not integration. Four videos explicitly label hand-only or welded-sphere R/L
diagnostics; none depicts a successful receiver or handoff.

The 20 vector PDFs are in `docs/figures/phase3CP05R/`; each was rendered and
visually checked. Videos and preview frames are in `outputs/phase3CP05R/videos/`.
`outputs/phase3CP05R/summary.json` holds the full audit and descriptor vectors.
`protocol.json`, `natural_continuation_protocol.json`, `realized_schedule.json`,
`settled_geometry.json`, `comparison.json`, `selection.json`, and the complete
`equilibrium_states/` cache supply exact provenance. Failed scout/static-fit
artifacts remain available and explicitly superseded, not erased.

Final exact pytest/diff outcomes are recorded in `outputs/phase3CP05R/validation.json`
and `pytest.log`. `artifact_manifest.json` enumerates paths and hashes. Tests
cover the reset distinction, cache integrity, unchanged physics, complete/censored
exposure, outcome gates, static-result preservation and scope restrictions.
Outputs, videos and caches remain git-ignored; source, tests, this audit and
requested PDFs remain intentionally uncommitted for PI review.
