# Phase 3B-0.5 Reset Feasibility Map

**ENGINEERING FEASIBILITY LEVELS - NOT PI-FROZEN TRAINING RANGES.**
The same unchanged contact-aware thumb-index acquisition controller is used
at every level. Position is sampled in an L1 ball; Euler and wrist components
are deterministic low-discrepancy probes. Level 3 alternates 15 and 20 degree
object-orientation envelopes so both requested larger probes are represented.

| Level | Tested | Position L1 radius | Object orientation | Wrist | Valid release | Gross collision | Immediate slip | Retained 250 / valid | Penetration median / p95 / max (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.004 m | +/-0 deg | +/-0 deg | 200 (1.000) | 0 | 0 | 135 (0.675) | 0.000655689 / 0.00069703 / 0.00109988 |
| 1 | 200 | 0.006 m | +/-5 deg | +/-5 deg | 200 (1.000) | 0 | 0 | 104 (0.520) | 0.00070767 / 0.000887486 / 0.00104337 |
| 2 | 200 | 0.008 m | +/-10 deg | +/-10 deg | 179 (0.895) | 0 | 0 | 78 (0.436) | 0.000737401 / 0.000980739 / 0.00126549 |
| 3 | 200 | 0.010 m | +/-20 deg | +/-15 deg | 126 (0.630) | 0 | 0 | 45 (0.357) | 0.000760329 / 0.00112348 / 0.00135328 |

## Candidate broader reset distributions

### CONSERVATIVE

**RECOMMENDATION ONLY - PI NOT YET FROZEN.** Position L1 radius 0.006 m; object Euler range [-5.0, 5.0] deg; wrist perturbation [-5.0, 5.0] deg. Observed acquisition=1.000, 250-step retention=0.520. Effective-N sensitivity: `{"0.0": 200, "0.01": 200, "0.025": 199, "0.05": 192, "0.1": 135, "0.15": 86, "0.2": 46}`.

### MODERATE

**RECOMMENDATION ONLY - PI NOT YET FROZEN.** Position L1 radius 0.008 m; object Euler range [-10.0, 10.0] deg; wrist perturbation [-10.0, 10.0] deg. Observed acquisition=0.895, 250-step retention=0.436. Effective-N sensitivity: `{"0.0": 179, "0.01": 179, "0.025": 179, "0.05": 176, "0.1": 167, "0.15": 137, "0.2": 101}`.

### CHALLENGING

**RECOMMENDATION ONLY - PI NOT YET FROZEN.** Position L1 radius 0.010 m; object Euler range [-20.0, 20.0] deg; wrist perturbation [-15.0, 15.0] deg. Observed acquisition=0.630, 250-step retention=0.357. Effective-N sensitivity: `{"0.0": 126, "0.01": 126, "0.025": 126, "0.05": 126, "0.1": 126, "0.15": 113, "0.2": 99}`.

No proposal replaces Phase 3B-0. The plots retain position, roll, pitch,
yaw, WRJ1, and WRJ2 separately rather than collapsing feasibility to a scalar.
