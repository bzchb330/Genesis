# Phase 3C-1.2B compiled actuation audit

Compiled fields, not actuator names alone, establish position-servo semantics. All gains are fixed, bias affine, activation dynamics absent, gear=1. Scalar actuator force is clipped kp*(ctrl-actuator_length). Actuator velocity bias is zero. Native passive damping is 0.05 for finger/forearm DOFs and 0.5 for wrist DOFs. Joint limits and actuator bounds are not changed.

| Actuator | Transmission target | kp | ctrl range | force range |
|---|---|---:|---|---|
| rh_A_WRJ2 | joint: rh_WRJ2 | 10.0 | [-0.523599, 0.174533] | [-10.0, 10.0] |
| rh_A_WRJ1 | joint: rh_WRJ1 | 8.0 | [-0.698132, 0.488692] | [-5.0, 5.0] |
| rh_A_THJ5 | joint: rh_THJ5 | 0.4 | [-1.0472, 1.0472] | [-3.0, 3.0] |
| rh_A_THJ4 | joint: rh_THJ4 | 1.0 | [0.0, 1.22173] | [-2.0, 2.0] |
| rh_A_THJ3 | joint: rh_THJ3 | 0.5 | [-0.20944, 0.20944] | [-1.0, 1.0] |
| rh_A_THJ2 | joint: rh_THJ2 | 1.5 | [-0.698132, 0.698132] | [-1.0, 1.0] |
| rh_A_THJ1 | joint: rh_THJ1 | 1.0 | [-0.261799, 1.5708] | [-1.0, 1.0] |
| rh_A_FFJ4 | joint: rh_FFJ4 | 1.0 | [-0.349066, 0.349066] | [-1.0, 1.0] |
| rh_A_FFJ3 | joint: rh_FFJ3 | 1.0 | [-0.261799, 1.5708] | [-1.0, 1.0] |
| rh_A_FFJ0 | fixed tendon: rh_FFJ0 | 0.5 | [0.0, 3.1415] | [-1.0, 1.0] |
| rh_A_MFJ4 | joint: rh_MFJ4 | 1.0 | [-0.349066, 0.349066] | [-1.0, 1.0] |
| rh_A_MFJ3 | joint: rh_MFJ3 | 1.0 | [-0.261799, 1.5708] | [-1.0, 1.0] |
| rh_A_MFJ0 | fixed tendon: rh_MFJ0 | 0.5 | [0.0, 3.1415] | [-1.0, 1.0] |
| rh_A_RFJ4 | joint: rh_RFJ4 | 1.0 | [-0.349066, 0.349066] | [-1.0, 1.0] |
| rh_A_RFJ3 | joint: rh_RFJ3 | 1.0 | [-0.261799, 1.5708] | [-1.0, 1.0] |
| rh_A_RFJ0 | fixed tendon: rh_RFJ0 | 0.5 | [0.0, 3.1415] | [-1.0, 1.0] |
| rh_A_LFJ5 | joint: rh_LFJ5 | 1.0 | [0.0, 0.785398] | [-1.0, 1.0] |
| rh_A_LFJ4 | joint: rh_LFJ4 | 1.0 | [-0.349066, 0.349066] | [-1.0, 1.0] |
| rh_A_LFJ3 | joint: rh_LFJ3 | 1.0 | [-0.261799, 1.5708] | [-1.0, 1.0] |
| rh_A_LFJ0 | fixed tendon: rh_LFJ0 | 0.5 | [0.0, 3.1415] | [-1.0, 1.0] |
| phase3c08_A_forearm_PS | joint: forearm_PS | 10.0 | [-1.5707963267948966, 1.5707963267948966] | [-10.0, 10.0] |

## Coupling

FF/MF/RF/LF J0 each controls J2+J1; no equality fixing their relative split. Pose-to-target mapping preserves sums, not independent distal positions.

All four fixed tendons have two unit joint terms. There is no equality tying J1 to J2. Passive dynamics, limits and contact redistribute the tendon-controlled total between them. The current model has one equality: the temporary sphere weld. Its presence does not indicate a digit coupling.

## Complete compiled parameters

The machine audit includes every gear component, ctrllimited/forcelimited flag, gainprm/biasprm/dynprm array, transmission enum and target, joint range/stiffness/damping, tendon springlength/stiffness/damping, and native-hand/contact fingerprint: outputs/phase3C12B/fixed_support/actuation_audit.json.

All 500 debug samples per primitive and 700 construction samples record ctrl, actuator force/limit fraction, actual saturation, joint qpos/qvel, tendon lengths/velocities and target error. Ctrl clipping and force saturation are independent measurements.

The original tangent-position command mapping was dimensionally correct but did not create an intentional persistent servo error. The virtual target now creates such error in actuator coordinates. Its Jacobian-based direction is a local engineering approximation; it cannot guarantee the passive distal split or an unchanged contact network.

Implementation: seqgrasp/phase3c12b.py: actuation_audit, transmission_matrix, normal_virtual_direction, saturation and record. Official [MuJoCo actuation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html).
