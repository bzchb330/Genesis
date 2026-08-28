# Phase 3C-0.7 pocket geometry audit

This audit was constructed before dynamic outcomes from exact compiled Shadow Hand geometry.

- Sphere: 25 mm diameter, 12.5 mm radius, density 1000 kg/m³.
- Analytic and compiled mass: `0.0081812308687234207 kg`.
- Grid: `12180` palm-frame candidate centers; `344` feasible voxels.
- RING_LITTLE_POCKET_VOLUME: union of feasible voxels, not a single Cartesian target.
- Bounds: `[-0.03125, -0.037000000000000005, 0.08375]` to `[-0.00875, -0.019, 0.11125]` m.
- Voxel-union volume: `4.3e-06 m³`.
- Construction: union of outcome-independent grid voxels with nonnegative open-hand clearance, exact ring/little flexion reach, palm proximity within one sphere radius, and a nonintersecting palm-normal incoming segment.
- Actual references: ring/little MCP roots, proximal links, and compiled palm/root collision geometry.
- The map records palm/middle/ring/little clearance, thumb/index and storage-finger reach gaps, incoming-path clearance, local opening, and escape directions for every voxel.

No dynamic result was used to draw this volume.
