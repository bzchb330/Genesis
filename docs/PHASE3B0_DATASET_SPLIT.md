# Phase 3B-0 Dataset Split

The split uses only the initial-state geometry descriptor. Ten deterministic,
equal-population principal-axis slabs are formed before any RL outcome exists.

- TRAIN: 300 states, regions [0, 2, 3, 5, 7, 9]
- VALIDATION: 100 states, regions [1, 6]
- TEST: 100 states, regions [4, 8]
- zero ID overlap: True
- zero serialized-state hash overlap: True
- nearest TEST-to-TRAIN descriptor distance: {'minimum': 0.003055291148625628, 'median': 0.023009442859665517, 'p95': 0.032751768807802964}

TEST regions are disjoint from TRAIN. Distance at region boundaries is reported
because categorical region separation does not imply a large geometric gap.
