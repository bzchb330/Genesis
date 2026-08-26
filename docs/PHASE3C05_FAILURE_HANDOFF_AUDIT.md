# Phase 3C-0.5 Failure-Handoff Audit

## Scope

This audit reconstructs the six existing Phase 3C-0 open-corridor failures
from `outputs/phase3C0/phase3c0_results.json`. It does not rerun or change the
Phase 3C-0 protocol. The machine-readable reconstruction, including every
recorded sample, is `outputs/phase3C05/failure_handoff_audit.json`.

## Exact timing

| Trial | Storage entry / storage motion | First thumb+index support loss | First storage contact | First palm contact | A floor loss | Commanded release start |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 386 | 386 | none | none | 462 | 987 |
| 2 | 387 | 393 | none | none | 488 | 988 |
| 3 | 385 | 386 | none | none | 446 | 986 |
| 4 | 387 | 393 | none | none | 448 | 988 |
| 5 | 388 | 389 | none | none | 499 | 989 |
| 6 | 405 | 436 | none | none | 511 | 1006 |

At storage entry the recorded WRJ2/WRJ1 coordinates ranged from approximately
`[-0.01851, -0.59800]` to `[-0.01422, -0.59740]` rad. The audit also preserves
thumb, index, middle, ring, little and palm normal forces; A pose and linear and
angular velocity; palm-frame gravity; and support topology at each timestep.

## Diagnosis

A was not lost because the scripted controller began its acquisition-finger
release too early. In all six trials, thumb/index support disappeared after
storage entry, no storage-finger or palm support ever formed, and A reached the
floor 476-544 steps before the commanded release stage began. Thus the actual
handoff failure was an uncommanded loss of acquisition contact before alternate
support became effective. The Phase 3C-0 sequential controller later issued a
release command, but that command operated on an object already lost.

This supports testing coordinated capture while thumb/index remain actively
engaged. It does not by itself select a final support force, persistence,
stability, or release threshold.

