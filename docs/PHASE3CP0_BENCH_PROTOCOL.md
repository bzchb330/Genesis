# Phase 3C-P0 frozen independent contact bench

Primary geometry: one free 25-mm sphere against a horizontal infinite rigid
plane. No hand, tendon, servo, weld or receiver. Mass, inertia, friction and
contact parameters are copied from the compiled active model. The sphere starts
at exact tangency with zero velocity and no overlap.

World gravity stays [0,0,-9.81]. The external COM load is [0,0,mg-F_target].
Therefore each requested target is the TOTAL normal load, not an extra force
added to weight. No force is applied through a controller or contact constraint.

Loads (N): 0.01, 0.02, 0.05, 0.08025787482217676, 0.10, 0.134311598,
0.20, 0.30, 0.50, 1.00. Two deterministic initial-state repeats per load.

Smooth cubic ramp: 0.4 s. Hold: 4.0 s. Final 200 steps provide mean and
variance. At the largest diagnostic timestep (0.004 s), the post-ramp hold still
contains 1000 steps. Sensitivity comparisons preserve physical durations rather
than reducing settling time with smaller timestep. Final-200 windows span
different physical durations and are explicitly labeled.

Load-unload cycles at 0.08025787482217676, 0.134311598, 0.30 and 0.50 N:
0.4-s ramp up, 4-s hold, matched 0.4-s ramp down, 4-s hold at zero TOTAL load.

Diagnostic subset: weight, 0.134311598, 0.50 and 1.00 N. Timesteps 0.001,
0.002, 0.004 s. Solver comparison: 400 Newton iterations / 1e-12 tolerance,
otherwise unchanged. Baseline current-timestep results are reused, not rerun.

Three small candidate options are frozen before their outcomes: half time
constant; constant 0.99 impedance; both changes. They run only after a CP-B/C/D
classification and only on this bench. They never change friction, production
physics or any grasp/receiver outcome. Each option receives the identical 36-run
suite. No option is automatically approved as physically unique.

## Instrumentation and interpretation

Signed overlap is R-z and may be negative after separation; nonnegative overlap
is also recorded. Contact normal force is obtained from mj_contactForce.
Position, velocity, qfrc_constraint, solver gradient/improvement/iterations,
kinetic/potential energy and external/gravity/contact work are recorded each step.
Contact work uses trapezoidal endpoint forces over actual displacement. The work
balance remainder is an integration/accounting diagnostic, not proof of hidden
energy creation. MuJoCo contact elastic potential is not exposed as a measured
continuum strain energy. A zero-load separation is not spontaneous loss under
positive load.

Settling time uses an explicitly engineering-only 1% force/position band,
1e-8-m position floor and 1e-5-m/s speed bound. It is measured after the load
ramp. No universal deformation/radius threshold is declared. Contact force sign,
contact transitions, force increments, residual kinetic energy, monotonicity,
repeatability and sensitivity are reported separately.

## Stage gates

No force-limited hand experiment unless current physics is CP-A or PI approves
a versioned candidate. Missing material properties do not authorize inventing
elastic constants. If options require review, stop with all hand outcomes N/A.
Do not resume ROLE_MRL_05, shape studies, handoff, RL, skin or object B.

The old C12B failure is read from saved states only. Its deep-overlap contact
topology, normals and friction ratios are not physically interpretable evidence.
An explicitly labeled inherited 3-mm engineering overlap gate rejects it; force
and external-wrench scales are reported without silently adopting new limits.
