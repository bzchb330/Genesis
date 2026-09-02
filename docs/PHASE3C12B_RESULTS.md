# Phase 3C-1.2B final report

**RECEIVER_CONSTRUCTION_FAILURE. Zero weld releases. No validated receiver.**

Two namespaces distinguish a setup diagnostic using the inherited soft weld from the fixed-support implementation. The soft support moved millimetres and did not satisfy the requested premise; all its evidence is preserved. The new temporary external-support weld uses solref=[0.004,1], solimp=[0.9999,0.9999,0.001,0.5,2]. This is an explicitly disclosed support-constraint implementation change, not a hand/contact-physics or actuator-gain change. Sphere translation is numerically constrained, not mathematically exact. No friction, geometry, solver, timestep, native coupling, joint bounds or gains changed.

## Actuation audit

1. codex/phase3C12b-weld-release-receiver

2. 6f550dede4b94b3e755bb3b1b208a1880a21a562

3. All 21 are position-like compiled general actuators: 17 joint transmissions, four fixed-tendon transmissions.

4. MF/RF: J4 and J3 joint actuators, J0 -> J2+J1 tendon. LF additionally has J5. FF similarly J4/J3/J0. Thumb J5..J1 are joint actuators. Wrist and forearm are joint actuators.

5. ctrl is target actuator length: joint angle for unit-gear hinges, sum of J2+J1 angles for fixed tendons. Fixed-tendon units here are generalized angular length, not metres and not normalized input.

6. MF/RF/LF direct joints kp=1; J0 tendon kp=0.5. Wrist WRJ2=10, WRJ1=8; forearm=10. Thumb J5/J4/J3/J2/J1=0.4/1/0.5/1.5/1. Actuator velocity damping=0; native passive joint damping remains.

7. MRL J4 [-0.349066,0.349066]; J3 [-0.261799,1.5708]; J0 [0,3.1415]; LFJ5 [0,0.785398]. Full wrist/thumb/index ranges in actuation_audit.json.

8. MRL and FF actuators [-1,1] in actuator force coordinates (joint torque for hinges); WRJ2 and forearm [-10,10], WRJ1 [-5,5]. Thumb [-3,3], [-2,2], then three [-1,1].

9. Four fixed tendons, unit coefficients: FFJ2+FFJ1, MFJ2+MFJ1, RFJ2+RFJ1, LFJ2+LFJ1.

10. No equality fixing the J2/J1 split. The shared actuator applies the same generalized torque contribution to both; 21 controls do not independently command 25 hand DOFs.

11. Pose-to-ctrl sum mapping was correct. Interpreting a kinematically assigned pose as actuator-held independent joints, or tangent targets as persistent preload, was not justified.

12. No measured actuator-force saturation in the primary construction: 0 timesteps; peak fraction 0.27552894. Some requested primitive commands clipped at ctrl bounds; that is not force saturation.

## Fixed-sphere preload

13. Little, palm/root (via wrist/forearm), middle, ring. Thumb/index audited, not dynamically swept.

14. [0,0.01,0.025,0.05,0.10,0.20,0.30,0.40] times the precomputed unit-max actuator-coordinate normal direction. Exact signed vectors and clipped targets are stored. Eight values x four surfaces x 500 steps. Items 15-22 describe the selected examples: little=0.40, palm=0.05, middle=0.40, ring=0.05. Actuator vector orders are LFJ5/J4/J3/J0, WRJ2/WRJ1/forearm_PS, MFJ4/J3/J0, RFJ4/J3/J0 respectively; values are final-100 means unless stated otherwise.

15. little [0.03209275085609234, -0.007605383547276717, 0.23955505923958678, -0.007781572238152532]; palm [0.0026861835054257065, 0.0486926010967413, 0.022755741842819095]; middle [-0.009603576703100575, 0.19634873466193412, -0.011859091395776069]; ring [0.021034081886624447, 0.0009967761830176403, 0.0007573483279114925]

16. little [0.03209275085609234, -0.007605383547276717, 0.23955505923958678, -0.003890786119076266]; palm [0.026861835054256798, 0.3895408087739304, 0.2275574184281901]; middle [-0.009603576703100575, 0.19634873466193412, -0.005929545697888034]; ring [0.021034081886624447, 0.0009967761830176403, 0.00037867416395574624]

17. little 0.0; palm 0.21678504239851137; middle 0.04518297834503078; ring 0.05106124079967508

18. little 0.0; palm 2.7011061890042667; middle 0.5629725238195052; ring 0.6362147130460262

19. Longest positive intended-contact run (steps): little 190; palm 415; middle 406; ring 310. Final 100-step persistence: little False; palm True; middle True; ring True.

20. No geom switching in isolated curves; disappearance is logged separately. Receiver adds ring-proximal and middle/little middle-link contacts.

21. First-to-last observed contact tangential migration (m): little 0.009918642300226372; palm 0.00022702853064328522; middle 0.00151697667289499; ring 0.006560398950935257. Lost contact is not interpolated to the end.

22. Final-100 force variance (N^2): little 0.0; palm 9.628473771183645e-07; middle 1.7467412774806516e-08; ring 2.5985371055803853e-06. Full curves and ranges are stored; no publication stability cutoff.

23. Not for all M/R/L: middle and ring demonstrate sustained non-saturated preload; little does not in the frozen isolated sweep. Palm also sustains preload. This is not proof that little cannot preload at another local contact.

## Welded receiver construction

24. Exact source qpos and sphere pose: fixed_support/receiver_protocol.json. ROLE_MRL_05 palm center [-0.01, -0.025, 0.12] m; no new basin search.

25. [1.56880566, 0.488691718, 0.173369422] rad in order forearm_PS, WRJ1, WRJ2. Reused C12A best; no outcome-based retuning.

26. 200 steps (0.4 s), all M/R/L concurrently.

27. 500 further steps (1.0 s).

28. phase3c07_sphere_geom/phase3_middle_rh_mfproximal_collision_0; phase3c07_sphere_geom/phase3_middle_rh_mfmiddle_collision_0; phase3c07_sphere_geom/phase3_ring_rh_rfproximal_collision_0; phase3c07_sphere_geom/phase3_little_rh_lfproximal_collision_0; phase3c07_sphere_geom/phase3_little_rh_lfmiddle_collision_0

29. middle [-0.980870566, 0.0505388273, -0.187986062]; middle [-0.764692147, 0.407459569, -0.499222015]; ring [-0.0716707445, 0.179421034, 0.981158192]; little [0.610355406, -0.716826654, -0.337084301]; little [0.579632292, -0.340520788, -0.74031885] (palm frame, final welded sample).

30. middle 2.2424807311507973 N; middle 1.2023564438459367 N; ring 0.6277734476055828 N; little 0.6747129917860636 N; little 1.4404856082060917 N

31. middle 0.16707931565147327 N; middle 0.17827252993076123 N; ring 0.16432229075123334 N; little 0.3096339497895673 N; little 0.6895106898919173 N

32. middle 0.1490129331597257; middle 0.2965385694786597; ring 0.523508254061977; little 0.9178241817159061; little 0.9573308972529052; spin/rolling moments separately logged, not mixed into rho.

33. [-1.39131431, 2.24908537, -0.0648223897] N (last-100 mean world vector).

34. [0.000370084644, 0.0129594469, 0.00503933463] N*m (about sphere COM).

35. [1.39131456, -2.24908513, 0.145080183] N.

36. [-0.000370089609, -0.0129594482, -0.00503932952] N*m.

37. [-1.39131431, 2.24908537, -0.145080265] N; norm 2.64862017 N, 33.0013744 weights.

38. [0.000370084644, 0.0129594469, 0.00503933463] N*m; norm 0.0139096772.

39. 0 force-saturated steps; peak actuator force/limit 0.27552894.

40. Maximum 0.008333245 m; final-100 maximum 0.00659890274 m, exceeding inherited 0.003-m reference. Fixed-support position error peaks at 1.1039393e-06 m (also just above frozen 1e-6-m numerical target).

41. Persistent M/R/L contacts form, but NOT an admissible stable receiver: excessive overlap, large unsupported wrench and force variation. Do not count weld support as success.

## Weld release

42. 700 was frozen; release cancelled at the construction gate.

43. NOT EXECUTED / N/A: weld remained active.

44. NOT EXECUTED / N/A: weld remained active.

45. NOT EXECUTED / N/A: weld remained active.

46. NOT EXECUTED / N/A: weld remained active.

47. NOT EXECUTED / N/A: weld remained active.

48. NOT EXECUTED / N/A: weld remained active.

49. NOT EXECUTED / N/A: weld remained active.

50. NOT EXECUTED / N/A: weld remained active.

51. NOT EXECUTED / N/A: weld remained active.

52. N/A: no release.

53. N/A: no release.

54. N/A: no release.

55. N/A: no release.

56. N/A: no release.

57. N/A: no release.

58. N/A: no release.

59. RECEIVER_CONSTRUCTION_FAILURE, with GROSS_PENETRATION observed during construction. No post-release failure label is assigned.

## Theory vs reality

60. Two predicted contacts: middle proximal + little proximal; normal-only total load 0.13431159757705 N.

61. Five final contacts across M/R/L: middle proximal+middle link, ring proximal, little proximal+middle link.

62. middle 16.12245037975466 deg; little 101.40291284559108 deg; other contacts have no predicted pair.

63. Predicted 0.134311597577 N vs final realized sum 6.18780922259 N; distinct realized geometry.

64. Predicted normal-only rho=0; realized per-contact values in item 32. Lower force utilization alone is not equilibrium.

65. Predicted [0, 4.36470911e-12, 5.20417043e-18] N vs realized last-100 mean [-1.39131431, 2.24908537, -0.145080265] N.

66. Predicted [1.49077799e-19, -1.81603864e-18, 6.23416249e-19] N*m vs realized last-100 mean [0.000370084644, 0.0129594469, 0.00503933463] N*m.

67. Actuator-coordinate preload sustains some isolated contacts, but distal calibration does not transfer to proximal receiver load. Simultaneous commands change the realized geometry, add contacts and overcompress the sphere; the weld balances a large residual wrench. Force saturation is not the cause. Tendon sum control is correct, but does not independently hold distal joint split. Little isolated contact is not sustained in this frozen sweep. The final realized normal cone still contains the gravity-support direction (angular residual 0 deg). This is force-cone feasibility only, not realized load allocation or full-wrench equilibrium.

## Final decision

68. Contact realization/load allocation remains the blocker. Correct ctrl semantics and increased virtual target establish some primitives but do not realize the assumed equilibrium network.

69. Persistent M/R/L contacts occur simultaneously, but not with acceptable overlap and balanced load. Little isolated preload remains unestablished.

70. It forms contacts, not a valid receiver network.

71. Not tested: gate correctly prevented release.

72. No validated 1000-step receiver.

73. None. Exact failed construction and pre-release integration state are preserved, not labeled validated.

74. Virtual offsets calibrated on isolated distal contacts were not transferable to the proximal multi-contact network. Additional contacts, large overlap and unbalanced wrench emerged without force saturation. Initial soft-weld compliance was a separate setup defect, now exposed and documented.

75. No.

76. No.

77. No.

78. Yes.

79. Yes.

80. Yes.

81. Bounded Phase 3C-1.2B follow-up: contact-specific proximal MRL preload/load-allocation debugging at the same frozen orientation and basin, with fixed-sphere support validated first. Do not resume handoff, shape trials, skin, RL or object B.

82. 368 passed, 7 warnings in 38.17s; exit 0. Command: .\.venv\Scripts\python.exe -m pytest -v

83. git diff --check: exit 0, no output. All eight new text files also passed individual whitespace checks.

84. 20 PDFs under docs/figures/phase3C12B/. Four post-release plots explicitly say NOT EXECUTED.

85. ['outputs/phase3C12B/fixed_support/videos/fixed_sphere_preload_debug.mp4', 'outputs/phase3C12B/fixed_support/videos/welded_MRL_receiver_construction.mp4']; no release/success/failure-of-release video exists because no release executed.

86. docs/PHASE3C12B_RESULTS.md; docs/PHASE3C12B_ACTUATION_AUDIT.md; outputs/phase3C12B/fixed_support/phase3c12b_summary.json, fixed_sphere_primitives.json, receiver_protocol.json, receiver_construction.json, release_results.json, *.npz and primitives/*.npz. Earlier soft-weld setup diagnostics remain under outputs/phase3C12B/.

## Scope and causal limitations

The observed network is not the predicted fixed network. In particular, distal-only primitive calibration does not establish proximal-contact force gains in ROLE_MRL_05. Middle/little virtual offset choices were frozen before each construction; little used an explicitly uncalibrated maximum-sweep diagnostic fallback. No controller was retuned using release outcomes. One inherited-soft-weld setup construction and one fixed-support construction were executed; neither released. No broad search or failure batch occurred.

The receiver readiness gate uses necessary multi-contact persistence, actual force-limit saturation, environment support, the inherited 0.003-m penetration reference and a 1-micrometre fixture position target. It does not silently decide a publication threshold for force variance or acceptable free wrench. Raw wrench/variance values are reported. In this experiment gross overlap independently blocks release; the fixture peak error also marginally exceeds its declared target. This phase does not prove impossibility of MRL storage or justify morphology/skin conclusions.

The frozen 0.003-m overlap reference is inherited from Phase 3C, not newly approved as a publication success criterion. Likewise, the 1e-9-N positive-force test is numerical detection, not weight-bearing adequacy. No post-release retention percentages or peak speeds are invented from the welded trajectory.

Theoretical vs realized comparisons use exact geom pair identity. Angular differences are measured in the actual palm frame; added contacts are not silently paired to old contacts. Weld wrench is recovered from the equality-only generalized force J^T f and mapped through the sphere COM Jacobian. Contact force and torque include solver-reported spin/rolling moments. Counterfactual acceleration is a frozen-contact instantaneous wrench prediction, not a simulated free trajectory.

Official reference: [MuJoCo actuation and constraint force mapping](https://mujoco.readthedocs.io/en/stable/computation/index.html). The compiled gain/bias arrays determine ctrl semantics; a fixed tendon controls a joint-angle sum and not the independent split.

## Unresolved PI decision

configs/phase3C12B_weld_release_receiver.yaml: publication thresholds for force stability, receiver wrench and overlap remain undecided. No such threshold was silently supplied.

## Reproduction

Use only .\.venv\Scripts\python.exe. scripts/run_phase3c12b.py exposes explicit audit, primitives, construction and release stages; existing outcome files prevent accidental reruns. scripts/analyze_phase3c12b.py consumes artifacts only. scripts/generate_phase3c12b_videos.py renders saved qpos without dynamics. Preserve both output namespaces.
