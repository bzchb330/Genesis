# Phase 3B-0.5 Actuator-Saturation Audit

Phase 3B-0's `actuator_saturation` field tests whether a desired position
command is exactly at a compiled `ctrlrange` endpoint. It does not test
measured actuator force against `forcerange`. This audit reports both.

| Actuator | Joint/tendon | Transmission | ctrlrange | forcerange | Release command-limit frac | Post-release command-limit frac | Release force-limit frac | Post-release force-limit frac | Max command-limit run |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| rh_A_WRJ2 | rh_WRJ2 | joint | [-0.523599, 0.174533] | [-10.0, 10.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_WRJ1 | rh_WRJ1 | joint | [-0.698132, 0.488692] | [-5.0, 5.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_THJ5 | rh_THJ5 | joint | [-1.0472, 1.0472] | [-3.0, 3.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_THJ4 | rh_THJ4 | joint | [0.0, 1.22173] | [-2.0, 2.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_THJ3 | rh_THJ3 | joint | [-0.20944, 0.20944] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_THJ2 | rh_THJ2 | joint | [-0.698132, 0.698132] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_THJ1 | rh_THJ1 | joint | [-0.261799, 1.5708] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_FFJ4 | rh_FFJ4 | joint | [-0.349066, 0.349066] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_FFJ3 | rh_FFJ3 | joint | [-0.261799, 1.5708] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_FFJ0 | rh_FFJ0 | fixed tendon (J2 + J1) | [0.0, 3.1415] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_MFJ4 | rh_MFJ4 | joint | [-0.349066, 0.349066] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_MFJ3 | rh_MFJ3 | joint | [-0.261799, 1.5708] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_MFJ0 | rh_MFJ0 | fixed tendon (J2 + J1) | [0.0, 3.1415] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_RFJ4 | rh_RFJ4 | joint | [-0.349066, 0.349066] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_RFJ3 | rh_RFJ3 | joint | [-0.261799, 1.5708] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_RFJ0 | rh_RFJ0 | fixed tendon (J2 + J1) | [0.0, 3.1415] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_LFJ5 | rh_LFJ5 | joint | [0.0, 0.785398] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_LFJ4 | rh_LFJ4 | joint | [-0.349066, 0.349066] | [-1.0, 1.0] | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1000 |
| rh_A_LFJ3 | rh_LFJ3 | joint | [-0.261799, 1.5708] | [-1.0, 1.0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| rh_A_LFJ0 | rh_LFJ0 | fixed tendon (J2 + J1) | [0.0, 3.1415] | [-1.0, 1.0] | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1000 |

| Actuator | Release command median | Release actual force min / median / max (N) | Post-release actual force min / max (N) |
|---|---:|---:|---:|
| rh_A_WRJ2 | -0.03896 | -0.0136088 / -0.0001891462 / 0.01246056 | -0.01523327 / 0.06115462 |
| rh_A_WRJ1 | -0.5694 | 0.388834 / 0.4140817 / 0.43917 | 0.388834 / 0.5880912 |
| rh_A_THJ5 | -0.1413 | 0.03254077 / 0.04284941 / 0.05307809 | 0.006585058 / 0.0560919 |
| rh_A_THJ4 | 0.8383 | -0.08383564 / -0.07071565 / -0.0563406 | -0.08383564 / 0.01845261 |
| rh_A_THJ3 | 0.199 | -0.007535552 / -0.007215083 / -0.006860603 | -0.007535552 / -0.001244638 |
| rh_A_THJ2 | 0.4945 | 0.01797578 / 0.02291552 / 0.02855322 | 0.002121144 / 0.03683891 |
| rh_A_THJ1 | 0.1291 | 0.005100819 / 0.006518719 / 0.008861136 | 0.002695292 / 0.01089206 |
| rh_A_FFJ4 | -0.3044844 | -0.006651323 / -0.004247128 / -0.002518726 | -0.0250975 / -0.002518726 |
| rh_A_FFJ3 | 0.9194 | 0.1361586 / 0.1407654 / 0.1436078 | 0.03629957 / 0.1506737 |
| rh_A_FFJ0 | 1.139674 | 0.05826051 / 0.0615268 / 0.06497629 | 0.002005204 / 0.06497629 |
| rh_A_MFJ4 | -0.05406 | -0.0006230208 / -0.0006186592 / -0.0006146042 | -0.005479647 / -0.0005710572 |
| rh_A_MFJ3 | -0.07509 | 0.01851572 / 0.01852469 / 0.01856551 | 0.01507857 / 0.02032507 |
| rh_A_MFJ0 | 0.47613 | 0.003573554 / 0.003575208 / 0.003576876 | 0.002371471 / 0.003905415 |
| rh_A_RFJ4 | -0.1062 | -0.001126341 / -0.001123736 / -0.001118518 | -0.001657092 / -0.001118518 |
| rh_A_RFJ3 | 0.2738 | 0.02158673 / 0.02160412 / 0.02165371 | 0.02110029 / 0.0247859 |
| rh_A_RFJ0 | 0.027111 | 0.001375629 / 0.001377642 / 0.001379588 | 0.001375629 / 0.001558001 |
| rh_A_LFJ5 | 0.1551 | 0.04307044 / 0.04311211 / 0.04316006 | 0.04119391 / 0.04563514 |
| rh_A_LFJ4 | -0.349066 | -0.004196932 / -0.004189428 / -0.004181275 | -0.00672691 / -0.004181275 |
| rh_A_LFJ3 | 0.1445 | 0.02153193 / 0.0215674 / 0.02160058 | 0.02153193 / 0.02386022 |
| rh_A_LFJ0 | 0 | 0.0004446504 / 0.0004450381 / 0.0004454761 | 0.000423296 / 0.0004606046 |

## Cause and interpretation

The exact two command-limit actuators are `rh_A_LFJ4` and `rh_A_LFJ0`.
`rh_A_LFJ4` clips because the official pre-grasp LFJ4 target lies below
its ctrlrange. `rh_A_LFJ0` is a fixed-tendon position servo for LFJ2+LFJ1;
the source keyframe sum is negative while the tendon ctrlrange starts at zero.
The controller holds free digits at that unchanged clipped pre-grasp target,
so the commands remain on their boundaries. Neither actuator reaches its
force range in any audited release or replayed post-release sample. Controller
gain and semantic actuator mapping are not responsible. This is natural
command clipping for the inherited keyframe/tendon representation, not actual
force saturation and not a blocker by itself. Renaming the historical field
would change stored schema, so the audit corrects the interpretation without
rewriting Phase 3B-0 results.

No actuator limit or gain was altered.
