# Phase 3B-0 Deduplication

- raw candidate count: 500
- exact duplicate count: 0
- exact-state retained count: 500
- descriptor: object palm-relative position divided by the 4 mm exercised radius; object palm-relative rotation vector divided by pi; thumb and index object-local contact positions divided by ellipsoid semi-axes; thumb and index joint positions centered and divided by compiled joint widths; wrist joint positions centered and divided by compiled joint widths
- distance: root-mean-square Euclidean distance across dimensionless descriptor coordinates
- threshold source: No PI-approved near-duplicate threshold exists; 0 retains exact unique states and nonzero values are sensitivity analyses only.

## Near-duplicate sensitivity (not frozen)

| Dimensionless RMS threshold | Retained | Flagged duplicate |
|---:|---:|---:|
| 0.0 | 500 | 0 |
| 0.01 | 468 | 32 |
| 0.025 | 243 | 257 |
| 0.05 | 70 | 430 |
| 0.1 | 17 | 483 |

The primary manifest removes exact serialized-state duplicates only. No
nonzero near-duplicate threshold is adopted without PI approval, and future
retention outcome is never used in this calculation.
