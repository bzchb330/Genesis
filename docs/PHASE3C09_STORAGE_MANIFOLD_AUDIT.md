# Phase 3C-0.9 morphology-aware storage manifold audit

The reduced search varied five quantities: middle, ring, and little flexion; WRJ2 offset; and forearm_PS. Thumb/index remained in the stored acquisition envelope. It evaluated `243` hand configurations and `6426` center voxels per configuration. The 5-mm center grid and 5-mm near-surface neighborhood are numerical diagnostic resolutions, not success thresholds.

- Valid configuration-center pairs: `33111`.
- Unique valid centers: `560`.
- Connected basins: `6`.

- `BASIN_01`: centroid `[-0.05, -0.020000000000000025, 0.04]`, `1` voxels / `1.25e-07 m3`, supports `['palm', 'little']`, `PARTIAL_CAGE`, aperture `0.025156578544094352`, reachability `CONNECTED_ONLY_WITH_HAND_RECONFIGURATION`, prior-pocket voxels `0`.
- `BASIN_02`: centroid `[-0.046250000000000006, 0.013749999999999957, 0.042499999999999996]`, `4` voxels / `5e-07 m3`, supports `['palm', 'little']`, `PARTIAL_CAGE`, aperture `0.025013138524397967`, reachability `NOT_CONNECTED_UNDER_TESTED_GEOMETRY`, prior-pocket voxels `0`.
- `BASIN_03`: centroid `[-0.003777573529411757, -0.029237132352941203, 0.0919025735294117]`, `544` voxels / `6.8e-05 m3`, supports `['middle', 'ring', 'little']`, `GEOMETRIC_CAGE_CANDIDATE`, aperture `0.02525619836344042`, reachability `DIRECTLY_GEOMETRICALLY_CONNECTED`, prior-pocket voxels `125`.
- `BASIN_04`: centroid `[-0.03500000000000001, -0.065, 0.12499999999999997]`, `1` voxels / `1.25e-07 m3`, supports `['ring', 'little']`, `PARTIAL_CAGE`, aperture `0.0638293059362582`, reachability `DIRECTLY_GEOMETRICALLY_CONNECTED`, prior-pocket voxels `0`.
- `BASIN_05`: centroid `[-0.029285714285714304, -0.06071428571428572, 0.058571428571428566]`, `7` voxels / `8.75e-07 m3`, supports `['ring', 'little']`, `PARTIAL_CAGE`, aperture `0.06275463872926614`, reachability `DIRECTLY_GEOMETRICALLY_CONNECTED`, prior-pocket voxels `0`.
- `BASIN_06`: centroid `[0.03999999999999995, 0.013333333333333289, 0.11333333333333329]`, `3` voxels / `3.75e-07 m3`, supports `['index', 'middle']`, `PARTIAL_CAGE`, aperture `0.025038951690156373`, reachability `DIRECTLY_GEOMETRICALLY_CONNECTED`, prior-pocket voxels `0`.

`BASIN_03` is the best-supported morphology-native target for the next design phase: it contains `544` of 560 unique valid centers, is directly geometrically connected, preserves thumb/index, has middle/ring/little support, and contains 125 of the former 344 ulnar-pocket voxels. The human-inspired pocket is therefore not wholly wrong, but it is an overly narrow subset of a broader Shadow-native middle/ring/little manifold. Geometry alone does not establish dynamic retention.
