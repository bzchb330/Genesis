# Phase 3C-0.9 trajectory failure diagnosis

Selection was deterministic: the five smallest recorded minimum pocket distances, with trial ID as a tie-breaker.

- `C08_C07_STATE_00050_F0_STATIC_OPTIMUM`: F0_STATIC, OPTIMUM, minimum `4.225637 mm` at step `592`, outcome `DROP_OR_ESCAPE`.
- `C08_C07_STATE_00035_F0_STATIC_OPTIMUM`: F0_STATIC, OPTIMUM, minimum `11.140107 mm` at step `557`, outcome `DROP_OR_ESCAPE`.
- `C08_C07_STATE_00050_F0_STATIC_PS_MINUS_5_DEG`: F0_STATIC, PS_MINUS_5_DEG, minimum `11.653060 mm` at step `545`, outcome `DROP_OR_ESCAPE`.
- `C08_C07_STATE_00035_F0_STATIC_PS_MINUS_5_DEG`: F0_STATIC, PS_MINUS_5_DEG, minimum `14.524637 mm` at step `506`, outcome `DROP_OR_ESCAPE`.
- `C08_C07_STATE_00035_F1_COORDINATED_PS_MINUS_5_DEG`: F1_COORDINATED, PS_MINUS_5_DEG, minimum `64.891452 mm` at step `359`, outcome `DROP_OR_ESCAPE`.

For the overall best trajectory, the 21-sample numerical window around the minimum had median distance rate `-0.0982929027 m/s`, sphere speed `1.23691306 m/s`, and zero median thumb/index normal force. The minimum occurred during drop/escape after acquisition-contact loss, not during a stationary loaded plateau. It is therefore **not jamming-consistent**.

Sphere position, finite-difference linear/angular motion, stored normal forces, contact points, forearm/wrist state, and palm-frame gravity were reconstructed. Tangential force, friction utilization, and contact slip velocity are unavailable because Phase 3C-0.8 did not log qvel/tangential force/slip and no dynamics rerun is authorized.
