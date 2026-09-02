# Phase 3C-P0 final report

**CP-C across the full requested rigid-contact load range. Candidate options await PI approval. No hand or receiver experiment executed.**

## Current physics

1. codex/phase3CP0-contact-physics-validation

2. 07a61ca7731b47f5b7bb03263cd956ad5c588b3e

3. 0.002 s.

4. Newton, 100 iterations, tolerance 1e-8; Euler; elliptic cone, impratio=10.

5. Sphere condim=6, hand condim=3; actual sphere-hand/bench contacts dim=6 because sphere priority=1.

6. Sphere [0.5,0.01,0.003]; hand [1,0.005,0.0001]; runtime pair [0.5,0.5,0.01,0.003,0.003]. Unchanged.

7. Sphere/runtime [0.02,1]; hand [0.005,1].

8. Sphere/runtime [0.9,0.95,0.001,0.5,2]; hand [0.5,0.99,0.0001,0.5,2].

9. 0 m margin and gap.

10. 0.0125 m.

11. 0.00818123086872342 kg.

12. 0.08025787482217676 N.

13. No: MATERIAL ELASTIC CONSTANTS NOT SPECIFIED BY PI. 459 baseline tracked text files searched; no justified E/nu or restitution found.

## Isolated contact bench

14. One free 25-mm sphere, horizontal infinite fixed plane, no hand/tendon/servo/weld. Exact compiled mass/inertia and contact-pair parameters. Starts tangent without overlap. Gravity retained; applied COM force z=mg-F_target.

15. [0.01, 0.02, 0.05, 0.08025787482217676, 0.1, 0.134311598, 0.2, 0.3, 0.5, 1.0] N; two repeats each. 0.4-s ramp + 4-s hold. Items 16-20 follow this ascending load order.

16. 5.42797439e-05, 0.000107418792, 0.000251587198, 0.000367181842, 0.000427601818, 0.000510863028, 0.000636057146, 0.000799258757, 0.00122230996, 0.00244461993 m.

17. 0.00434237951, 0.00859350336, 0.0201269759, 0.0293745474, 0.0342081455, 0.0408690422, 0.0508845717, 0.0639407006, 0.097784797, 0.195569594.

18. 0.01, 0.02, 0.05, 0.0802578748, 0.1, 0.134311598, 0.2, 0.3, 0.5, 1 N.

19. 1.38478854e-20, 1.36326272e-20, 1.44216314e-20, 1.44500114e-20, 1.40724727e-20, 1.33733724e-20, 1.40983465e-20, 1.3507335e-20, 1.54324275e-20, 1.53332897e-20 N^2.

20. 0.062, 0.078, 0.078, 0.06, 0.048, 0.036, 0.036, 0.044, 0.14, 0.16 s after ramp. Engineering-only 1% force/position band and 1e-5-m/s speed bound; not publication success.

21. True: steady normal force and compression increase monotonically.

22. Exact repeated saved trajectories; maximum absolute difference 0.

23. All four unload cycles dissipate net contact work. No residual overlap: sphere separates at zero total load and continues small free drift because external force cancels gravity. This is not positive-load contact loss.

24. Loaded final-200 mean kinetic energy <= 3.85072864e-25 J. Cycle contact work is negative. External/gravity/contact work and integration remainder are logged; no exact continuum energy or global passivity proof is claimed.

## Numerical robustness

25. dt=0.001/0.002/0.004 s. Max relative steady-overlap change 1.02082299e-10; complete transient/settling/energy differences stored.

26. 400 iterations / 1e-12 vs 100 / 1e-8: max relative steady-overlap change 2.72661881e-11.

27. No. Steady force-deformation behavior is contact-parameter dominated in this isolated geometry.

28. CP-C for rigid-contact interpretation across the complete tested range; numerically stable, not CP-D.

29. Response is monotonic, repeatable and convergent, but the legacy compliance produces 0.367182 mm at object weight, 0.510863 mm at the idealized receiver load, 1.22231 mm at 0.5 N and 2.44462 mm at 1 N. At the upper tested load the overlap is about 19.6 percent of the 12.5-mm radius, incompatible with interpreting this entire range as small rigid-contact deformation. This is not a CP-D numerical-instability diagnosis.

## Physical reference

30. Symbolic Hertz reference only; no numerical material reference fitted.

31. No E, nu, restitution or material identity invented. Plane is kinematically rigid but solver contact is compliant.

32. Local finite-difference dF/d(delta): 188.185531, 193.546216, 237.87596, 304.43945, 362.638157, 457.069146, 562.913917, 573.770439, 456.379046, 409.061543 N/m; secant stiffness also stored.

33. Not over the entire requested force range: upper-load deformation is not small compared with radius. This does not claim all low-load uses fail.

## Physics calibration

34. Yes: independent contact-parameter study required by CP-C.

35. OPTION_TC10: solref=[0.01, 1.0], solimp=[0.9, 0.95, 0.001, 0.5, 2.0]; OPTION_IMP99: solref=[0.02, 1.0], solimp=[0.99, 0.99, 0.001, 0.5, 2.0]; OPTION_TC10_IMP99: solref=[0.01, 1.0], solimp=[0.99, 0.99, 0.001, 0.5, 2.0]

36. LEGACY_PHASE3C_CONTACT_PHYSICS: delta(weight)=0.367181842 mm, delta(0.1343N)=0.510863028 mm, delta(1N)=2.44461993 mm; OPTION_TC10: delta(weight)=0.10775542 mm, delta(0.1343N)=0.176035566 mm, delta(1N)=0.719326237 mm; OPTION_IMP99: delta(weight)=0.03924 mm, delta(0.1343N)=0.0656681618 mm, delta(1N)=0.488923985 mm; OPTION_TC10_IMP99: delta(weight)=0.00981 mm, delta(0.1343N)=0.0164170404 mm, delta(1N)=0.122230996 mm

37. LEGACY_PHASE3C_CONTACT_PHYSICS: maximum relative steady-overlap change=1.02082299e-10; OPTION_TC10: maximum relative steady-overlap change=1.52857788e-10; OPTION_IMP99: maximum relative steady-overlap change=2.84544702e-10; OPTION_TC10_IMP99: maximum relative steady-overlap change=1.69216806e-10

38. None selected or approved.

39. All three options are numerical candidates only. Missing material data/scope require PI review; smallest penetration alone is not a selection rule.

40. LEGACY_PHASE3C_CONTACT_PHYSICS remains production. PHYSICS_V1_RIGID_CONTACT has NOT been created/approved.

41. Legacy parameters are frozen in current_physics.json; candidate option definitions in config/registry. No production settings were modified.

## Force-control primitive

42. NOT EXECUTED: hand force-control stage is blocked by protocol pending CP-A or PI-approved revised physics.

43. Existing Shadow actuator semantics and transmissions were preserved; no hand controller modification.

44. None dynamically tested in P0; planned middle/ring/little.

45. [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2] N, predeclared but not executed.

46. N/A: hand stage not executed.

47. N/A: hand stage not executed.

48. N/A: hand stage not executed.

49. N/A: hand stage not executed.

50. N/A: hand stage not executed.

51. N/A: hand stage not executed.

52. N/A: hand stage not executed.

53. N/A: hand stage not executed.

54. No validated bounded-force hand primitive is claimed.

## Regression

55. Yes, rejected using saved evidence only; no C12B dynamics rerun.

56. Settled max 6.598903 mm, peak 8.333245 mm violate inherited 3-mm engineering reference. Last summed normal load 6.187809 N (~77 weights), mean unsupported residual 2.648620 N (~33 weights) and cancelling weld wrench are separately flagged. Do not interpret deep-contact normals/topology/rho as valid receiver mechanics.

57. Reusable physical_admissibility.py logs penetration/radius/diameter ratios, total normal force/weight, residual force/torque, actual actuator saturation fractions, external-support wrench, kinetic energy and environment support. Gates require explicit labels; no default scientific acceptance threshold.

## Final decision

58. PI-approved contact-physics interpretation/parameters, followed by genuinely bounded contact-force control.

59. Numerical normal-contact response characterized; physical/material validation and production candidate approval remain pending.

60. Candidate changes were needed for the calibration study, but no revised production parameters are selected.

61. No: blocked and not executed.

62. Not yet. Phase 3C-P1 requires approved contact physics and validated bounded-force primitives first.

63. Yes.

64. Yes.

65. Yes.

66. Yes.

67. PI review of P0 material assumptions and candidate options, then the gated P0 fixed-sphere force-control study. Only after validation consider Phase 3C-P1; do not start it automatically.

68. 405 passed, 7 warnings in 30.32s; exit 0. Command: .\.venv\Scripts\python.exe -m pytest -v

69. git diff --check: exit 0, no output. All 11 new text files also passed individual whitespace checks.

70. 16 PDFs: requested 1-15 plus physical-admissibility summary. Selected-physics figure explicitly says no selection; four hand-primitive figures omitted because the hand stage did not run.

71. {'generated': ['outputs/phase3CP0/videos/sphere_plane_contact_bench.mp4'], 'frames': 440, 'fps': 25, 'duration_s': 17.6, 'physics_steps_during_rendering': 0, 'reconstruction': 'Recorded z trajectory in exactly axial sphere-plane geometry; no trajectory extrapolation.', 'fixed_sphere_force_control_video': None, 'reason': 'Hand force-control stage not approved/executed'}; no hand-force, receiver or grasp-success video.

72. docs/PHASE3CP0_RESULTS.md; docs/PHASE3CP0_CURRENT_PHYSICS_AUDIT.md; docs/PHASE3CP0_BENCH_PROTOCOL.md; outputs/phase3CP0/phase3cp0_summary.json, current_physics.json, material_audit.json, frozen_protocol.json, physics_registry.json, legacy_classification.json, legacy_regression.json, preserved_phase3C12B_hashes.json, each version/results.json and per-trial JSON/NPZ traces.

## Measured steady load-deformation table

| Applied load N | Measured Fn N | Overlap mm | delta/R | Fn variance N^2 | Settling s | Secant N/m | Local gradient N/m |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.01 | 0.0542797439 | 0.00434237951 | 1.38e-20 | 0.062 | 184.231 | 188.186 |
| 0.02 | 0.02 | 0.107418792 | 0.00859350336 | 1.36e-20 | 0.078 | 186.187 | 193.546 |
| 0.05 | 0.05 | 0.251587198 | 0.0201269759 | 1.44e-20 | 0.078 | 198.738 | 237.876 |
| 0.0802578748 | 0.0802578748 | 0.367181842 | 0.0293745474 | 1.45e-20 | 0.06 | 218.578 | 304.439 |
| 0.1 | 0.1 | 0.427601818 | 0.0342081455 | 1.41e-20 | 0.048 | 233.862 | 362.638 |
| 0.134311598 | 0.134311598 | 0.510863028 | 0.0408690422 | 1.34e-20 | 0.036 | 262.911 | 457.069 |
| 0.2 | 0.2 | 0.636057146 | 0.0508845717 | 1.41e-20 | 0.036 | 314.437 | 562.914 |
| 0.3 | 0.3 | 0.799258757 | 0.0639407006 | 1.35e-20 | 0.044 | 375.348 | 573.77 |
| 0.5 | 0.5 | 1.22230996 | 0.097784797 | 1.54e-20 | 0.14 | 409.062 | 456.379 |
| 1 | 1 | 2.44461993 | 0.195569594 | 1.53e-20 | 0.16 | 409.062 | 409.062 |

## Options for PI review, not a selected physical model

| Model | Overlap at weight mm | At 0.134311598 N mm | At 1 N mm | Settling range s | Max timestep relative delta change |
|---|---:|---:|---:|---|---:|
| LEGACY_PHASE3C_CONTACT_PHYSICS | 0.367181842 | 0.510863028 | 2.44461993 | [0.03600000000000003, 0.1600000000000001] | 1.02e-10 |
| OPTION_TC10 | 0.10775542 | 0.176035566 | 0.719326237 | [0.0020000000000000018, 0.03600000000000003] | 1.53e-10 |
| OPTION_IMP99 | 0.03924 | 0.0656681618 | 0.488923985 | [0.038000000000000034, 0.11800000000000005] | 2.85e-10 |
| OPTION_TC10_IMP99 | 0.00981 | 0.0164170404 | 0.122230996 | [0.0020000000000000018, 0.03400000000000003] | 1.69e-10 |

## Energy and scope limitations

The zero-total-load unload endpoint is force-balanced away from contact: mg is cancelled by the external bench load. A small upward velocity is therefore not damped after contact vanishes. Separation/drift is expected in that neutral free state; it is not equilibrium-position settling. All observed cycle contact work is net dissipative, but this limited diagnostic does not prove global passivity. Contact elastic energy is not directly exposed as a continuum strain-energy state. Work-balance residuals retain finite-step quadrature and integrator error.

The normal sphere-plane response is independent of hand geometry and tendon control; that separation is the purpose and also a limit. This phase does not validate tangential friction, rolling/spinning, multi-contact geometry or receiver force allocation. The C12B state is rejected, not recycled to infer final normals, topology, friction utilization or morphology.

Frozen resource fractions were not recomputed: thumb 0.9559782183972225, index 1.0, opposition 0.9665998246424643. The idealized static-network result remains conditional, not a realized receiver.

TODO(PI): material elastic constants, physically justified deformation/rate scope, and contact-option approval. No physics version is promoted and no scientific threshold is silently resolved.

## Reproduction

Use only .\.venv\Scripts\python.exe. scripts/run_phase3cp0.py has explicit audit/legacy/candidates/regression/hand-gate stages. Existing results are protected from overwrite; candidates require the saved CP-B/C/D decision. scripts/analyze_phase3cp0.py consumes saved arrays only. Do not bypass the hand-stage approval gate.

References: [MuJoCo constraint/contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters), [constraint computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-model).
