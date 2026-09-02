# Phase 3C-1.2A calibration autopsy

All 66 original instantaneous contact-force values reconstruct exactly (maximum error 0 N). No old settling trajectories were rerun. The old hand surface and joints never moved; the sphere moved along a geom-centroid radial ray. Measured force came before the following 25 free-object steps, not after a controlled quasi-static normal command.

No active geom switching was found. Tangential contact migration is real but small; it does not prove the entire cause of force nonmonotonicity. Do not relabel reproducible instantaneous forces as corrupt data.

| Surface | Old normal fraction range | Maximum consecutive tangential migration (mm) | Corrected 0.2-mm IK error (mm) | Command normal fraction | Settled normal fraction | Settled drift (mm) |
|---|---:|---:|---:|---:|---:|---:|
| middle | 0.990181-0.991258 | 0.017482 | 0.083942 | 0.907660 | 0.926468 | 1.559086 |
| ring | 0.960302-0.961680 | 0.065078 | 0.123648 | 0.785990 | 0.111897 | 12.278255 |
| little | 0.980839-0.980842 | 0.048685 | 0.000000 | 1.000000 | 0.135331 | 3.757350 |
| palm | 0.966261-0.966261 | 0.064391 | 0.053350 | 0.963765 | 0.952951 | 0.792710 |
| thumb | 0.986927-0.988834 | 0.001270 | 0.037869 | 0.997666 | 0.175793 | 7.887264 |
| index | 0.999171-0.999194 | 0.009183 | 0.000000 | 1.000000 | 0.722341 | 5.931702 |

The local normal is measured using the frozen 0.025-mm runtime contact probe. A sphere tangent material point is targeted via bounded analytic Jacobian IK. Palm/root uses existing forearm/wrist joints. Targets that cannot be reached are retained and explicitly report IK residuals; they are not called exact normal motion.

Every sample then uses existing actuator position commands and the existing object fixture for 50 steps (100 ms). Final ten-step means are zero for all six surfaces because contact is lost. Large material-point drift shows that kinematic qpos realization does not imply actuator-held configuration. This failed settling trial is not a valid quasi-static force-capacity measurement. Further debugging must characterize joint-target realization/coupling and geometry/control drift without changing contact physics.

Geom identity, contact normals, contact distance, geometric closest-point distance, forces and per-step contact lists are stored. Branches terminate on a different contact geom. Absent contact distance is absent/null, not fabricated as a penetration. Nominal sustainable force capacity cannot be inferred.

Source paths: old seqgrasp/phase3c11.py:_tangent_setup and force_approach_calibration; new seqgrasp/phase3c12a.py:105, seqgrasp/phase3c12a.py:158, seqgrasp/phase3c12a.py:141, seqgrasp/phase3c12a.py:70.

Machine evidence: outputs/phase3C12A/calibration_autopsy.json and corrected_calibration.json. Preload means **normal command offset + actual resulting force**, not geometric penetration alone.
