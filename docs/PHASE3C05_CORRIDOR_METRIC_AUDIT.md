# Phase 3C-0.5 Corridor-Metric Audit

## Methods

The Phase 3C-0 corridor metric compares an object bounding sphere against
bounding spheres around collision geoms along a straight candidate path. The
audit reconstructed every recorded transfer sample and compared that metric
with exact MuJoCo `mj_geomDistance` queries between the oriented object
ellipsoid and the compiled middle/ring/little collision geoms.

Coordinate transforms were checked against the compiled palm and world poses.
No frame mismatch was found. The machine-readable result is
`outputs/phase3C05/corridor_metric_audit.json`.

## Actual-path comparison

| Trial | Bounding-sphere clearance on actual path | Exact geom clearance on actual path | Unused-finger contact steps |
|---:|---:|---:|---:|
| 1 | 18.512 mm | 44.968 mm | 0 |
| 2 | 17.988 mm | 43.117 mm | 0 |
| 3 | 21.374 mm | 48.959 mm | 0 |
| 4 | 20.797 mm | 47.628 mm | 0 |
| 5 | 17.134 mm | 42.815 mm | 0 |
| 6 | 19.264 mm | 43.896 mm | 0 |

## Conclusion

The previously reported negative values were conservative candidate-path
predictions, not penetration observed on the actual dynamic path. Conservatism
comes from both bounding-sphere geometry and the straight candidate path being
different from the realized dynamic trajectory. The actual path had positive
clearance under both metrics and zero middle/ring/little contact samples in all
six trials.

There is no implementation bug to repair. Both measurements are retained:
candidate-path clearance remains a conservative planning diagnostic, while
exact actual-path geom distance and observed contacts report realized physics.
The prior Phase 3C-0 conclusion is narrowed transparently rather than silently
rewritten.

