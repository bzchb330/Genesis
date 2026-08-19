# Phase 2 PI inputs status

All working parameters needed by the Phase 2 resource-correlation experiment were supplied by the PI on 2026-08-08 and are recorded in `configs/phase2_physics_validation.yaml`. They are scoped to this experiment and are not a final definition of scalar resource metric J.

The supplied values cover:

- Part A stability gates, the 81-condition contact sweep, and the a-priori rule to retain passing baseline physics;
- Part B sampling, 500-step screening, and Ferrari–Canny numerical construction;
- Part C occupied-finger threshold, Monte Carlo budgets, collision tolerance, and palm-frame volume box;
- Part D A-retention, B-acquisition, invalidity, B-placement, timing, and repeated-trial definitions;
- Part E the three tactile features and thresholds;
- Part F binning, confidence intervals, repeated-trial robustness, and greedy-baseline reporting.

No Phase 2 execution input remains blocked. `compute_J` deliberately remains PI-blocked: no component weights, normalization, aggregation, or units for J have been supplied. The Phase 2 implementation therefore stores and analyses the three raw components separately.
