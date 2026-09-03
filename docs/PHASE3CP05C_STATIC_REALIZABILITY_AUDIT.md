# Phase 3C-P0.5C nominal-pose static realizability audit

## Decision

The exact nominal cup pose is **CASE C - transmission limited** under the
compiled existing Shadow Hand actuator/tendon structure. The 21 x 25 actuator
moment matrix has rank 21. An unconstrained least-norm actuator allocation leaves
a 0.00506283828227 generalized-force residual, 0.986908418655% of the required
static generalized-force norm. This is far above the explicitly recorded
roundoff-only diagnostic (5.23e-11), but no publication or task-success threshold
is inferred. The PI should interpret physical materiality.

The bounded and unbounded solutions coincide; no actuator is saturated and the
maximum utilization is 10.1268239%. Thus the evidence supports CASE C, not CASE B.
The protocol mandates STOP for CASE B/C: no preload control was constructed, no
direct or perturbation dynamics ran, and no PHYSICS_V1 alias was created.

Branch: `codex/phase3CP05C-static-realizability-audit`. Base P0.5R commit:
`213fbe2f965b346e3f0b89c33576f2089b4de19b`. P0.5R was validated (494 passed,
7 existing warnings), committed, and pushed before this branch was created.
P0.5C is intentionally uncommitted and is not merged into main.

## Frozen scope and provenance

The audit uses the exact nominal state from
`outputs/phase3CP05/geometry.json` and verifies the confirmed P0.5R cache hash
`ba5cf8a3a9e18ec9f8eb69ebd32a33f79c0a27b9e103efe49fa73cab51d07ca1`.
The hand is compiled directly from the same vendored MJCF and forearm transform,
without adding the sphere, fixture or floor. Gravity remains active. At the
nominal state qvel=0; qfrc_applied and xfrc_applied are identically zero; there
is no contact. The compiled hand has nq=nv=25 and nu=21.

All actuator kp/gain/bias/dynamics, control/force ranges, gear/transmission,
tendon definitions/coefficients/stiffness, joint limits/stiffness, damping,
armature, friction loss and body gravcomp arrays are copied into the frozen
protocol record and verified unchanged after the audit. No model compensation,
controller retuning, object experiment, receiver, regression, B, RL, shape or
skin work occurred. P0.5R outputs are preserved by SHA-256.

The PI decision that sustained M/R/L is no longer a prerequisite for contact-
physics selection is recorded verbatim as protocol metadata. Conditional V1
freeze nevertheless does not execute because this static audit found a new
foundational reason to stop. The future first naturally occurring sustained
M/R/L state is recorded for a single identical-state IMP99/TC10 cross-check,
with no optimization loop.

## Joint drift attribution

The complete 25-row joint table is in `joint_drift_audit.csv`, both native order
and absolute-drift-sorted order are in `summary.json`. Classification comes from
compiled actuator moment coefficients and transmission enums, not joint names.

The largest drift is `rh_RFJ2`: nominal 0.499415999094 rad, settled
-0.000480096816 rad, signed delta -0.499896095910 rad. It is not directly
actuated or independently commandable. It has moment coefficient 1 in the
fixed-tendon actuator `rh_A_RFJ0`; that actuator applies the same coordinate
force to the coupled distal pair. It is therefore tendon-coupled/underactuated,
not a zero-column unactuated DOF. There are no fully unactuated hand DOFs.

Top ten absolute drifts:

| Joint | Signed delta, rad | Direct | Tendon-coupled | Independently commandable |
|---|---:|---|---|---|
| rh_RFJ2 | -0.499896096 | no | yes | no |
| rh_RFJ1 | +0.498591753 | no | yes | no |
| rh_LFJ2 | -0.382522376 | no | yes | no |
| rh_LFJ1 | +0.381455863 | no | yes | no |
| rh_MFJ2 | -0.368746562 | no | yes | no |
| rh_MFJ1 | +0.366594052 | no | yes | no |
| rh_WRJ1 | -0.064403923 | yes | no | yes |
| rh_LFJ5 | -0.045069647 | yes | no | yes |
| rh_FFJ3 | -0.026693968 | yes | no | yes |
| rh_MFJ3 | -0.026479725 | yes | no | yes |

Across 17 direct coordinates, maximum/summed/mean absolute drift is
0.0644039233 / 0.237280399 / 0.0139576705 rad. Across eight tendon-coupled
distal coordinates, it is 0.499896096 / 2.498758897 / 0.312344862 rad.

## Compiled actuator architecture

The 21 actuators comprise 17 `mjTRN_JOINT` transmissions and four
`mjTRN_TENDON` transmissions (`rh_A_FFJ0`, `rh_A_MFJ0`, `rh_A_RFJ0`,
`rh_A_LFJ0`). Each fixed tendon has compiled unit coefficients on its J2 and J1
coordinates. Full targets, ranges, gainprm, biasprm, gear, nominal length and
25-element moment rows are in `actuator_transmission_audit.csv`.

The dense matrix reconstructed from MuJoCo 3.11's sparse
`data.actuator_moment` and moment row/column metadata is verified as (21,25), so
`qfrc_actuator = actuator_moment.T @ actuator_force`. Twenty-one isolated
1e-4 actuator-coordinate force probes yield maximum generalized mapping error
0 and targeted force-generation error 1.10182046e-17. This establishes matrix
orientation empirically rather than assuming it.

Singular values are four sqrt(2), fifteen 1, and two values equal to 1 within
floating precision. Rank is 21; nonzero condition number is sqrt(2), so the
represented subspace is well-conditioned. The four missing generalized
directions are the antisymmetric distal-pair modes.

Force ranges by actuator order are [-10,10], [-5,5], [-3,3], [-2,2], sixteen
[-1,1] ranges through the finger actuators, and [-10,10] for forearm pronation/
supination. These are native actuator-force coordinates; their mechanical unit
depends on transmission, so the CSV does not relabel every entry as Nm.

## Required static holding force

At q=q_nominal, qvel=qacc=0, `mujoco.mj_inverse` computes qfrc_inverse. MuJoCo
defines this as the net external/actuation generalized force required for the
specified motion after subtracting internal forces. In this no-contact/no-
external-applied-force state, it equals the generalized actuator force required
for static holding. This follows the [official MuJoCo computation model](https://mujoco.readthedocs.io/en/latest/computation/index.html),
including passive-force treatment and actuator moment arms.

The required 25-vector is stored without rounding in `summary.json`; its maximum
component is 0.506341196084 and L2 norm is 0.512999806929. The maximum initial
qacc without preload reproduces 81.61685063369046 rad/s^2. For a consistency
cross-check, inverse dynamics is evaluated at the original forward acceleration;
qfrc_inverse - qfrc_actuator - qfrc_applied has maximum error 2.08166817e-17.
The static identity qfrc_inverse = qfrc_bias - qfrc_passive - qfrc_constraint
has error 0. No hand-derived gravity approximation is used.

## Allocation and residual attribution

Unbounded allocation uses `numpy.linalg.lstsq` on A.T without regularization or
weights. Bounded allocation uses SciPy `lsq_linear` with the compiled native
force ranges and exact dense least-squares solver. Both return the same force
vector to 1e-14. Maximum absolute actuator force is 0.506341196084, maximum
utilization 0.101268239217, and no actuator is saturated.

The unbounded/bounded residual norm is 0.00506283828227 and relative norm is
0.00986908418655. Dominant residuals are paired and opposite-signed:

| Distal pair | J2 residual, Nm | J1 residual, Nm |
|---|---:|---:|
| index | -0.0026364375 | +0.0026364375 |
| middle | -0.0017036923 | +0.0017036923 |
| little | -0.0014409522 | +0.0014409522 |
| ring | -0.0009415160 | +0.0009415160 |

Nominal joint-limit margins for the residual-dominant J2/J1 coordinates are:
index 0/0, middle 0.368084508/0.299679326, ring
0.499415999/0.103161058, and little 0.381928016/0.376119743 rad. The index
pair is exactly at its configured lower limits; the M/R/L residuals are not
explained by proximity to a limit. At the specified zero-acceleration state,
MuJoCo reports zero inverse passive and constraint generalized force. Thus
tau_required equals qfrc_bias here as a verified result, not an assumption.
The CASE C conclusion concerns this exact nominal pose and the compiled
MuJoCo inverse/soft-constraint formulation; it does not claim that every nearby
pose or every possible physical tendon hand is unrealizable.

Each pair receives only one summed tendon force, while the nominal static state
requires unequal generalized forces on its two distal coordinates. The residual
is orthogonal to the actuator span. All independently commandable-coordinate
residuals are at numerical roundoff. This is the direct structural basis for
CASE C; it is not attributed to force limits, singular conditioning, wrist,
forearm, contact physics or sphere initialization.

## Mandatory stop and prior-work implications

No `ctrl_equilibrium` is produced. Although the compiled actuators are affine
fixed-gain position actuators (`gainprm[0]`, affine length/velocity bias), force
inversion is prohibited after CASE C. Consequently requested-versus-realized
preload, initial/0.5-s/2-s dynamics, nominal drift, M/R/L FK drift, tendon
stability and local perturbation results are **not available**, rather than
fabricated. The corresponding requested PDFs explicitly show protocol-gated
unavailability.

Natural settling's 20-25-mm M/R/L geometry changes are strongly associated with
underactuated distal redistribution, but CASE C means they cannot be called
merely a zero-preload artifact. The geometric C-space/workspace calculations
remain geometrically valid for their sampled configurations; this audit adds a
realizability qualifier to dynamic use of the nominal pose. It does not authorize
a storage-manifold or 1.56-million-state rescan.

Historical resource fractions remain exactly thumb 0.9559782183972225, index
1.0, opposition 0.9665998246424643. They are not yet supported as dynamically
maintainable at this exact nominal pose and require future realizability-aware
confirmation. They were not overwritten or rerun.

FOUNDATIONAL SIMULATION INFRASTRUCTURE READY is **not declared**. B03 and fly-by
regressions cannot run next; neither can the bounded-force primitive. The exact
next phase is PI interpretation of the CASE C distal-transmission mismatch and
a PI-level decision on whether/how the model or pose assumptions may change.
No such scientific/model decision is made here.

## Reproduction and artifacts

All Python invocations use `.\\.venv\\Scripts\\python.exe`. Run the static audit
with `scripts/run_phase3cp05c_audit.py`; it executes no physics steps. Figures
use saved static arrays only. The complete machine-readable record is
`outputs/phase3CP05C/summary.json`; immutable inputs/scope/hashes are in
`protocol.json`. The two CSVs are RFC-style flat tables validated through the
workspace artifact tooling. P0.5C tests verify exact sources, compiled mappings,
inverse/forward identities, allocations, classification, mandatory stop,
parameter immutability, scope and P0.5R preservation.
