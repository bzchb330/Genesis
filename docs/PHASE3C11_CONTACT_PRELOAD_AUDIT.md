# Phase 3C-1.1 compiled contact and preload audit

The audit reads the instantiated MuJoCo model and measures actual contact response. It does **not** assume that zero geometric penetration always means zero normal force.

## Frozen compiled model

- Object geom: `mjGEOM_SPHERE`, condim `6`, friction `[0.5, 0.01, 0.003]`, margin `0.0` m, gap `0.0` m, solref `[0.02, 1.0]`, solimp `[0.9, 0.95, 0.001, 0.5, 2.0]`.
- Representative hand geoms: condim 3, friction `[1.0, 0.005, 0.0001]`, margin/gap 0 m, solref `[0.005, 1.0]`, solimp `[0.5, 0.99, 0.0001, 0.5, 2.0]`.
- Runtime object-hand contacts: dim 6, friction `[0.5, 0.5, 0.01, 0.003, 0.003]`, solref `[0.02, 1.0]`, solimp `[0.9, 0.95, 0.001, 0.5, 2.0]`.
- Solver: timestep 0.002 s, iterations 100, tolerance 1e-08, gravity [0.0, 0.0, -9.81] m/s².

## Calibration and frozen selection

The principal signed-approach sweep is `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]` mm. The explicitly documented extension is `[0.75, 1.0]` mm. The representative pair is chosen from compiled fingertip collision geometry (or nearest compiled palm geom), with other fingers placed in the existing zero-flexion configuration. The inward approach direction is selected by a two-sided signed-distance probe.

The frozen rule is: pair-specific force measured at the 0.20-mm center of the PI-proposed 0.10-0.30-mm engineering region; fall back to the smallest point in that region with force above numerical zero, then to the smallest explicitly documented extension point only if the entire region is force-inactive The selection was hashed before B03 outcomes (`723c7c9023346b96d438dd13fa2e750c1921e7f3b6bec2a8d532202dc4089be7`). Targets are:

| Surface | Approach (mm) | Measured target force (N) |
|---|---:|---:|
| middle | 0.30 | 0.000309341 |
| ring | 1.00 | 0.002303557 |
| little | 0.20 | 0.057360289 |
| palm | 0.20 | 0.043387776 |
