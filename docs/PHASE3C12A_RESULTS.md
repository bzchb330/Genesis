# Phase 3C-1.2A final report

Primary outcome: **CASE E**. Static feasibility does not establish an actuator-realizable receiver. Phase 3C-1.1 was committed and both branches pushed; Phase 3C-1.2A remains uncommitted.

## Calibration audit

1. codex/phase3C12a-contact-gravity-wrench-audit

2. 6ca46034a743ba265b7ef58be452decdcb138f33

3. Sphere geom phase3c07_sphere_geom paired with middle/ring/little/thumb/index distal collision_0 meshes and phase3_palm_rh_palm_collision_1 box; exact IDs, bodies, positions and normals in calibration_autopsy.json.

4. 0 switches in all 66 reconstructed old samples; 0 in 60 corrected branches.

5. Yes: nonzero tangential contact-position migration on each surface. This alone does not explain all force nonmonotonicity.

6. No: geom-centroid radial sphere translation, stationary hand joints/surfaces. Mostly normal (0.9603-0.9992 fraction), but not measured local-normal displacement; force was sampled before free settling.

7. Material-point Cartesian target along runtime inward normal; analytic mj_jac bounded joint IK. Existing actuator targets and existing object weld, 50 steps x 0.002 s, last 10 samples summarized. No kinematic clamping. Some IK targets are unreachable and settling drifts.

8. [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5] mm, frozen before sweep; command offset is not penetration.

9. All six settled normal-force curves are [0,0,0,0,0,0,0,0,0,0] N. Contacts disappear, not usable zero-force preload curves. Initial contact forces are recorded separately.

10. 0.080257874822176764 N; compiled mass 0.0081812308687234207 kg.

11. Settled force/weight = 0 at all 60 samples; initial transients are not calibrated capacity. Weight is a scale, not an automatic sum-force success threshold.

12. Parameterization/measurement error confirmed; no geom-switch bug. Corrected calibration also fails stable control, so physical conformity cannot yet be inferred.

## MRL mechanics

13. middle [-0.95056131, -0.227278652, -0.211607207]; little [0.648348786, 0.620215096, 0.441562097]

14. Positive hull of two inward normals; rank 2, planar span 148.287303367 deg; zero 3D solid angle (not a broad frictionless volume).

15. False

16. 93.881758047 deg.

17. Compiled elliptic condim-6 minimum-load solution rho_max=0.997291332907, mean/median=0.996405650904. Restricted point-force model infeasible; rho there is undefined.

18. Original: translational-only point-force F/F/F/F/F/F. Full compiled wrench F/F/F/F/T/T at scales [0,.1,.25,.5,.75,1]; minimum tested .75, not exact continuous onset. Optimized normal-only solution feasible at every scale, minimum 0. Full curve keeps spin/rolling coefficients fixed; scale=0 there is not the all-frictionless model.

19. [0.991987939, 0.0629724356, 0.109518958]

20. 1.56880565869 rad (89.8859431 deg)

21. 0.488691717608 rad (27.9999729 deg)

22. 0.173369421689 rad (9.93333616 deg)

23. 0 deg

24. True

25. 0.0; exact normal-only witness, not the minimum-load frictional solution.

26. [0, 4.36470911e-12, 5.20417043e-18] N; norm=4.36470911e-12 N

27. [1.49077799e-19, -1.81603864e-18, 6.23416249e-19] N*m; norm=1.92584224e-18 N*m

28. Not necessarily at the identified static orientation; physical preload capacity and dynamic retention are not established.

## True thumb-assisted storage

29. Old ROLE-T allowed thumb but required only any two nearby surfaces and closed two nearest digits. Actual ring+little topology reporting was correct. It did not test mandatory thumb support.

30. 18

31. 3 geometrically; 2 with thumb+opposing positive initial normal force; none settled as receivers.

32. Two preloaded static candidates: thumb+ring and thumb+little. Frozen geometry-selected representative: ROLE_T_TRUE_07.

33. little [0.437241179, -0.898211843, -0.0451180371]; thumb [-0.742382658, -0.0364117544, -0.668985929]

34. Rank 2, planar span 105.171638852 deg; zero 3D normal-cone solid angle.

35. Original false; optimized true.

36. [0.114162445, 0.85382009, 0.507895846]

37. [0.116257151, -0.516684436, 0.174533] rad (forearm_PS, WRJ1, WRJ2).

38. 0.0; normal-only witness.

39. Force [-3.46944695e-18, -7.9487826e-10, 0] N; torque [1.30883531e-17, 4.54890574e-17, -1.81942677e-17] N*m.

40. Both achieve rho=0. MRL normal span 148.287 vs true-T 105.172 deg; loads 0.134311598 vs 0.129963308 N. No demonstrated true-T superiority.

41. Not demonstrated. Both need actuator-realizable preload. Full-wrench sample feasibility favors MRL in this bounded audit, not a morphology-wide ranking.

## Resource workspace

42. 0.9559782183972225

43. 1.0

44. 0.9665998246424643

45. 9.8976612134173708e-05 m^3; baseline 0.00010065479045636668; fraction 0.98332738745384951.

46. 1.6808351625750898e-05 m^3; baseline 9.9185186257933305e-05; fraction 0.16946433494655544.

47. Aperture-paired midpoint hull 6.6997110235251099e-05 m^3; fraction 0.6644796476700775. Independent configurations, not true opposition or joint collision-free acquisition proof.

48. Archived MRL preserves thumb/index resource fractions better than this true-T state preserves index/middle. Different samples and morphologies prevent a direct usefulness ranking; usefulness requires PI definition.

## Final decision

49. CASE E: Cartesian calibration does not maintain stable, interpretable contacts with existing controls. D-type measurement correction and conditional A-type static orientation result are secondary.

50. Invalid as a settled local-normal preload calibration; old instantaneous measurements are reproducible, not fabricated.

51. Yes in the fixed-network static equations; no physical receiver claim. MRL original frictionless false -> optimized true.

52. Conditionally as a static geometry, not validated storage.

53. No superiority shown; actual thumb contact is now demonstrated only in static configurations.

54. No actuator-realizable, calibrated receiver has been established.

55. No receiver authorized for dynamics. Conditional candidate: ROLE_MRL_05 at the reported q, middle/little required normal forces [0.10425577113835052, 0.030055826438699335] N; preload capacity unestablished.

56. No: debug quasi-static calibration first, then obtain PI authorization for Phase 3C-1.2B.

57. No.

58. Yes.

59. Yes.

60. Yes.

61. Bounded Phase 3C-1.2A calibration-control/geometry debugging, with fixed physics. Only after measured sustained preload: propose Phase 3C-1.2B direct-hold validation at the identified storage pose.

62. 338 passed, 7 warnings in 37.91s; exit code 0

63. PASS; exit code 0; no output

64. 18 vector PDFs in docs/figures/phase3C12A/.

65. docs/PHASE3C12A_RESULTS.md; docs/PHASE3C12A_CALIBRATION_AUTOPSY.md; docs/PHASE3C12A_ROLE_T_AUDIT.md; docs/PHASE3C_RESOURCE_RECOVERY_KINEMATIC_RESULT.md; outputs/phase3C12A/*.json; exact manifest below.

## Solver interpretation and limits

Two explicitly separated models are stored. The translational point-force model retains all six force/torque equations, using a 64-ray inscribed Coulomb cone (maximum radial conservatism 0.12046%). An outer polygon checks infeasibility certificates. It excludes spin/rolling moments. For two radial sphere contacts its force-feasible directions can have zero 3D angular volume. Never equate its failure with failure of the compiled condim-6 model.

The compiled-reference model uses the runtime elliptic condim-6 cone. A cutting-plane outer LP imposes exact Euclidean-cone separation until slack >= -1e-9 N; an infeasible outer LP certifies infeasibility. Moment coefficients retain their physical length units. rho_translation is reported separately from the combined elliptic-cone utilization. Full-reference objective is minimum total normal load, NOT minimum rho: its best-pose solution may use friction to reduce normal load even though a separate rho=0 witness exists.

Offline friction scales affect translational coefficients only; spin/rolling coefficients stay compiled. All-frictionless feasibility is separately computed with point normals and no contact moments. No offline coefficient is written to MuJoCo. No force-capacity constraint can be justified from the failed calibration, so static results assume unbounded compressive normal-load availability.

Orientation search: 45 compiled/diagnostic-range coarse poses plus three local refinements. Refinement minimizes normal-cone projection residual; descriptors are ranked lexicographically. This is not a global guarantee or a uniform solid-angle estimate. Sample counts include clustered refinement poses. The transport comparator is the first archived C07_STATE_00000 transport-optimal row, fixed in advance, not a universal transport optimum. All 18 frozen true-T attempts are additionally audited in local_candidate_mechanics.json: 15 lack the mandatory geometric network; all three geometric networks (06, 07, 08) admit optimized rho=0. Only 06 and 07 have positive initial forces. Selection remains frozen at 07; no new geometry search or receiver dynamics.

| Candidate | Normal span (deg) | Point-force feasible / 48 | Full compiled feasible / 48 | Frictionless / 48 | Transport-storage angle (deg) |
|---|---:|---:|---:|---:|---:|
| ROLE_MRL_05 | 148.287303 | 3 | 48 | 3 | 137.606385 |
| ROLE_T_03 | 75.691394 | 3 | 13 | 3 | 102.063125 |
| B03_PREVIOUS_BEST | 60.914739 | 3 | 9 | 3 | 29.652825 |
| ROLE_T_TRUE_07 | 105.171639 | 3 | 13 | 3 | 66.696808 |

Each optimized pose preserves the measured geom pairs, palm-frame contact positions and normals under a rigid sphere/palm transform (static forward only). No world-gravity rotation. These checks do not establish a trajectory into that pose. Worst sampled configurations, all per-pose solutions, contact bodies/geoms/gaps/lever arms and force components are in mechanics_audits.json.

New preload representation is commanded Cartesian offset + resulting measured contact force. Nominal sustainable capacity remains **unknown**, not zero capacity: this experiment lost contact. No publication materiality or receiver-validity threshold was invented.

Contact sign follows the [official MuJoCo contact convention](https://mujoco.readthedocs.io/en/latest/XMLreference.html): normal points geom1 to geom2; flip when the object is geom1. Runtime dim=6 and friction=[0.5,0.5,0.01,0.003,0.003]. Closest-point and Jacobian calls follow the [official API](https://mujoco.readthedocs.io/en/3.2.6/APIreference/APIfunctions.html).

## Artifacts and reproduction

`scripts/run_phase3c12a.py` runs the frozen calibration/search; do not rerun for report generation. `scripts/analyze_phase3c12a.py` consumes existing data only. The machine summary hashes every existing Phase 3C-1.1 JSON. Generated datasets remain ignored under outputs/. PDFs are deliberate report artifacts.

- outputs/phase3C12A/calibration_autopsy.json
- outputs/phase3C12A/corrected_calibration.json
- outputs/phase3C12A/local_candidate_mechanics.json
- outputs/phase3C12A/mechanics_audits.json
- outputs/phase3C12A/old_role_t_audit.json
- outputs/phase3C12A/phase3c12a_results.json
- outputs/phase3C12A/true_role_t_search.json
- outputs/phase3C12A/true_role_t_workspace.json
- outputs/phase3C12A/validation.json
- docs/figures/phase3C12A/old_calibration_contact_identity.pdf
- docs/figures/phase3C12A/local_normal_vs_joint_approach.pdf
- docs/figures/phase3C12A/corrected_normal_force_vs_command_offset.pdf
- docs/figures/phase3C12A/contact_normal_tracking.pdf
- docs/figures/phase3C12A/object_weight_vs_calibrated_contact_force.pdf
- docs/figures/phase3C12A/MRL_contact_normal_cone.pdf
- docs/figures/phase3C12A/MRL_gravity_vs_normal_cone.pdf
- docs/figures/phase3C12A/MRL_friction_dependence.pdf
- docs/figures/phase3C12A/MRL_gravity_orientation_friction_map.pdf
- docs/figures/phase3C12A/transport_vs_storage_optimal_orientation.pdf
- docs/figures/phase3C12A/old_ROLE_T_implementation_audit.pdf
- docs/figures/phase3C12A/true_thumb_assisted_topologies.pdf
- docs/figures/phase3C12A/ROLE_T_true_contact_normal_cone.pdf
- docs/figures/phase3C12A/ROLE_T_true_gravity_orientation_map.pdf
- docs/figures/phase3C12A/MRL_vs_ROLE_T_true_friction_requirement.pdf
- docs/figures/phase3C12A/index_middle_workspace_ROLE_T_true.pdf
- docs/figures/phase3C12A/resource_recovery_kinematic_preliminary.pdf
- docs/figures/phase3C12A/phase3C12A_decision_summary.pdf
