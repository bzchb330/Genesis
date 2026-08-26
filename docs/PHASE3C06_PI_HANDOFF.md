# Phase 3C-0.6 PI handoff

1. **Branch:** `codex/phase3C06-sphere-palmodigital-storage` (left uncommitted; not merged).
2. **Base commit:** `7baac924a14ff863c7d1b0bb9bfc67734390609d`.
3. **Measured finger-link lengths:** index, middle, ring, and little each have a 0.045 m proximal and 0.025 m intermediate joint-to-joint link (eight measurements total).
4. **L_ref:** median of all eight audited corresponding non-thumb proximal/intermediate link lengths, `(0.025 + 0.045) / 2 = 0.035 m`, because the model provides no unique single representative link.
5. **D0 diameter:** 0.035000 m (35 mm).
6. **D0 radius:** 0.017500 m (17.5 mm).
7. **D0 mass:** 0.022449297504 kg at the inherited/default 1000 kg/m3 density; analytic and compiled masses agree.
8. **Candidate pockets:** old palm-center control `(0.020000, -0.025000, 0.075000)` m; middle/ring `(0, -0.017500, 0.092625)` m; ring/little `(-0.022000, -0.017500, 0.086375)` m; adjacent ulnar palmodigital `(-0.024625, -0.021875, 0.082875)` m. All are palm-frame volumes, not exact points.
9. **Palm-center matched result:** 268/500 entries, 4/500 transient stable captures, 4 release attempts, 0/500 valid thumb recoveries, 0/4 at 1000 steps.
10. **Middle/ring result:** 122/500 entries, 0/500 stable captures, 0 thumb recoveries.
11. **Ring/little result:** 0/500 entries, 0 stable captures, 0 thumb recoveries.
12. **Ulnar palmodigital result:** 0/500 entries, 0 stable captures, 0 thumb recoveries.
13. **Best-supported tested region:** old palm-center control for transient capture only; no region supported valid post-release storage.
14. **D0 acquisition:** 50/50 frozen states had valid thumb+index contact and no middle/ring/little contact at freeze.
15. **D0 corridor clearance:** 1,689/2,000 formal condition trials (84.45%).
16. **D0 pocket entry:** 390/2,000 (19.50%).
17. **NO_PRESHAPE capture:** 2/1,000 transient captures (0.20%), 0 valid thumb recoveries.
18. **PRESHAPE capture:** 2/1,000 transient captures (0.20%), 0 valid thumb recoveries.
19. **First storage-finger contact:** 832 trials; median step 234, range 135-298. Geometric preshape trigger median step 170, range 170-299.
20. **Ring contact:** 784/2,000 (39.20%).
21. **Little contact:** 128/2,000 (6.40%).
22. **Palm/root contact:** 406/2,000 (20.30%).
23. **Alternate support:** 832/2,000 (41.60%).
24. **Thumb release attempts:** 4.
25. **Valid thumb recovery:** 0/4 attempts and 0/2,000 formal trials.
26. **Index diagnostic:** not released; the primary thumb-recovery milestone failed, so a second acquisition-resource release was not justified.
27. **Retention:** 100-step 0/4, 500-step 0/4, 1000-step 0/4; all four losses occurred during the thumb-release ramp.
28. **Dominant raw load topologies:** thumb+index 468,584 sample occurrences; ring 22,136; index 20,985; middle 4,681; palm 4,444; middle+ring 3,465; middle+palm 1,238; little 1,013; thumb 703; ring+little 95; index+palm 40. These are sample occurrences, not independent trials.
29. **Maximum penetration by hand-sphere surface:** thumb 0.000990374 m; index 0.002469079 m; middle 0.007293006 m; ring 0.003673872 m; little 0.001303301 m; palm/root 0.003790996 m.
30. **Maximum penetration normalized by R0:** thumb 0.056593; index 0.141090; middle 0.416743; ring 0.209936; little 0.074474; palm/root 0.216628.
31. **Penetration interpretation:** the global raw maximum was 0.416743 R0. Acceptability remains TODO(PI); no new threshold or automatic gross-overlap label was applied. Multi-millimeter solver overlap is reported without interpreting it as biological skin deformation.
32. **Fixed wrist W0:** 78/400 entries, 0/400 transient captures, 0 thumb recoveries.
33. **W1 +/-5 degrees:** 312/1,600 entries, 4/1,600 transient captures, 0 thumb recoveries.
34. **W2 +/-10 degrees:** not run; the D0 physical progression gate failed.
35. **W3 +/-20 degrees:** not run; the D0 physical progression gate failed.
36. **Beneficial wrist direction:** `[+5,+5]` was the only command with transient captures (4/400); actual motion in those trials was WRJ2 4.556-4.688 degrees and WRJ1 3.047-3.093 degrees.
37. **Harmful/non-beneficial directions:** `[-5,-5]`, `[-5,+5]`, `[+5,-5]`, and W0 each produced zero transient captures and zero thumb recoveries. This supports only a direction-specific temporary-settling effect.
38. **Gravity-in-palm relationship:** the four transient captures had approximately `[0, 8.420, 5.034]` m/s2 gravity in palm coordinates. World gravity remained exactly `[0,0,-9.81]` m/s2.
39. **Native wrist sufficiency:** not established. W1 changed transient settling but did not produce resource recovery.
40. **Forearm rotation:** necessity is not established because the transfer controller did not reach ring/little or ulnar targets; `PHASE3C06_FOREARM_ROTATION_DOF_LIMIT` was not raised and no joint limit was enlarged.
41. **Sizes tested:** D0 only.
42. **D1 (1.25 D0):** not run; D0 gate failed.
43. **D2 (1.50 D0):** not run; D0 gate failed.
44. **D3 (1.75 D0):** not run; D0 gate failed.
45. **D4 (2.00 D0):** not run; D0 gate failed.
46. **Largest demonstrated size:** D0 had four transient stored states but no valid post-release storage; therefore no sphere size was demonstrated storable after thumb recovery.
47. **Size vs wrist:** D0/W0 had 0 transient captures; D0/W1 had 4; D0/W2-W3 and all D1-D4 cells were not run and are not imputed.
48. **Failure taxonomy counts:** ACQUISITION_FAILED 0; TRANSFER_CORRIDOR_BLOCKED 311; PRESHAPE_TOO_EARLY 0; PRESHAPE_TOO_LATE 0; POCKET_NOT_REACHED 1,610; POCKET_GEOMETRY_MISALIGNED 0; NO_STORAGE_FINGER_CONTACT 1,192; NO_LOAD_BEARING_SUPPORT 1,168; SPHERE_ROLLED_OUT 0; SPHERE_SLID_OUT 0; WRIST_DIRECTION_UNFAVORABLE 0; EXCESSIVE_PENETRATION 0; JOINT_BOUNDARY_LIMIT 2,000; WRIST_DOF_LIMIT 0; LOSS_DURING_THUMB_RELEASE 4; LOSS_AFTER_THUMB_RELEASE 0; OTHER 0. The joint-boundary label records raw negative compiled margins from official open/target keyframes and is not a new exclusion rule.
49. **Classification:** primary **SP-C** with no qualifier: spheres reached candidate volumes, but the current rigid hand/controller did not secure one through thumb recovery.
50. **Palmodigital hypothesis:** not supported by the tested bounded controller; old center outperformed the proposed pocket targets, and ring/little plus ulnar targets were never reached.
51. **Preshaping:** not beneficial in the matched aggregate; entry and transient-capture counts were identical.
52. **Wrist assistance:** beneficial only for four transient captures in one direction, not for the primary thumb-recovery endpoint.
53. **Rigid-contact geometry:** insufficient for reproducible storage and resource recovery under this protocol.
54. **Future compliant skin:** not yet justified; correct ring/little/ulnar transfer geometry was not reached, so conformity/contact area is not yet isolated as the dominant failure.
55. **Object B next:** no. One-object palmodigital storage remains unsolved.
56. **RL:** remains premature; no RL, reward, reward weights, or scalar J was added.
57. **Pytest:** final run `185 passed, 7 warnings in 34.22s` using only `.venv\\Scripts\\python.exe`.
58. **git diff --check:** passed; Git emitted only its informational LF-to-CRLF working-copy warning for `seqgrasp/phase3/model.py`.
59. **Figures:** all 16 requested one-page vector PDFs were generated, rendered with Poppler, and visually inspected in `docs/figures/phase3C06/`.
60. **Videos:** five truthful actual-MuJoCo MP4 replays were generated: D0 acquisition, open transfer, preshape/wrist transient settling, palm-center release failure, and ring/little pocket failure. No success, long-retention, ring/little-entry, or larger-sphere video was fabricated.
61. **Artifact roots:** reports in `docs/PHASE3C06_FINGER_LINK_SIZE_AUDIT.md`, `docs/PHASE3C06_RESULTS.md`, and this handoff; figures in `docs/figures/phase3C06/`; machine results in `outputs/phase3C06/phase3c06_results.json`, `analysis_summary.json`, `mechanism_evidence.json`, `matched_states/`, and `timeseries/`; videos in `outputs/phase3C06/videos/`; implementation in `seqgrasp/phase3c06.py`, `configs/phase3C06_sphere_palmodigital.yaml`, and the four Phase 3C-0.6 scripts; tests in `tests/test_phase3c06_sphere_palmodigital.py`.

## Unresolved TODO(PI) decisions

- `configs/phase3C06_sphere_palmodigital.yaml:39`: freeze a publication criterion for acceptable sphere-contact penetration.
- `configs/phase3C06_sphere_palmodigital.yaml:40`: freeze a publication success-rate threshold for size progression.
- `configs/phase3C06_sphere_palmodigital.yaml:41`: decide whether a later compliant-skin ablation is scientifically warranted.

None was resolved automatically.
