# Phase 3C-0.5 Coordinated Palmar Capture Results

## Scope and frozen protocol

Branch: `codex/phase3C05-coordinated-palmar-capture`

Base commit: `b20cf473dd9bf524128c3a212626162caee27e7f`

Fifty physically valid states were frozen before formal capture conditions.
Every state reached `A_APPROACHES_STORAGE` through OPEN_HAND, thumb/index
minimal acquisition, fixture release, and open-corridor transfer; had both
thumb and index contact; had no middle/ring/little contact; and used no outcome
selection. The same IDs were used across all capture comparisons.

No MJCF, friction, condim, timestep, solver, mass, object geometry, gravity,
joint limit, or actuator limit was changed. Object B was not instantiated in a
manipulation experiment. No RL, reward, scalar objective, PPO, or training was
introduced.

## 1-18: Capture and support results

1. **Branch:** `codex/phase3C05-coordinated-palmar-capture`.
2. **Base:** `b20cf473dd9bf524128c3a212626162caee27e7f`.
3. **Matched N:** 50 frozen states.
4. **Phase 3C-0 diagnosis:** acquisition contact vanished at steps
   386/393/386/393/389/436; A hit the floor at 462/488/446/448/499/511; no
   alternate support formed; commanded release did not start until
   987/988/986/988/989/1006. All six failures preceded commanded release.
5. **Subsets:** middle; ring; little; middle+ring; middle+little; ring+little;
   middle+ring+little.
6. **Alternate-support rate by subset:** for the engineering 10%/25-step gate,
   serial vs simultaneous counts out of 50 were little 1/1, middle 4/3, ring
   4/4, middle+little 3/3, middle+ring 4/5, ring+little 4/4, and all three 4/5.
7. **Serial capture:** 22/350 (6.286%).
8. **Simultaneous fixed-wrist capture:** 23/350 (6.571%).
9. **Wrist-assisted capture:** W1 pooled 134/1400 (9.571%); conditional W2
   137/176 (77.841%, conditional denominator only); predefined
   wrist-load-transfer capture 3/50 (6%).
10. **Wrist ranges exercised:** W0 0 degrees; W1 four diagonal WRJ2/WRJ1
    commands at +/-5 degrees; W2 corresponding +/-10-degree directions only
    after a W1 endpoint retained A. No larger probe was run.
11. **Gravity in palm:** fixed simultaneous medians were approximately
    `[0, 8.0825, 5.5597] m/s2`; all wrist samples ranged approximately x
    `[-2.18e-15, 2.18e-15]`, y `[7.4582, 8.8938]`, z `[4.1397, 6.3726] m/s2`.
    These are descriptive axes, not an optimized gravity score.
12. **Palm contact:** 141/2326 (6.062%) across all capture trials.
13. **Storage-finger contact:** 345/2326 (14.832%).
14. **Maximum alternate-load fraction:** 1.0.
15. **Persistence:** serial 10% gate counts at 10/25/50 steps were 34/24/22;
    simultaneous 38/25/23; W1 226/171/130 of 1400; conditional W2
    145/144/135 of 176; load-transfer 3/3/3 of 50.
16. **10/25/50% load gates:** serial counts for 10/25/50-step persistence were
    34/24/22, 34/16/13, 30/12/9; simultaneous were 38/25/23, 38/22/17,
    32/11/8; W1 were 226/171/130, 223/135/89, 185/91/40; conditional W2 were
    145/144/135, 145/120/101, 133/78/44; load-transfer were 3/3/3, 2/2/1,
    2/1/1.
17. **Fixed vs wrist-assisted:** W1 `[-5,-5]`, `[-5,+5]`, `[+5,-5]`, and
    `[+5,+5]` produced 17, 10, 62, and 45 captures out of 350, versus 23/350
    fixed. Exact paired discordances (wrist-only/fixed-only) were 2/8, 0/13,
    42/3, and 24/2; exact McNemar p values were 0.1094, 0.000244,
    `8.65e-10`, and `1.05e-5`.
18. **Actual wrist motion in successful wrist trials:** vector magnitude median
    7.976 degrees, p95 12.766, maximum 13.450 across 274 W1, conditional W2,
    and load-transfer successful capture trials. WRJ2 ranged -10.514 to 9.654
    degrees; WRJ1 ranged -6.120 to 8.500 degrees.

Capture here means endpoint A retention plus `lambda_alt >= 0.10` for 25
consecutive steps. That is the predefined engineering diagnostic, not a final
scientific threshold.

## 19-28: Release, survival, topology, and load transfer

19. **Thumb-first recovery:** 8/12 executed candidates (66.667%); the eight
    successes span three matched states.
20. **Index-first recovery:** 0/12 executed candidates.
21. **Overall one-resource recovery:** 8/24 executed release candidates
    (33.333%); 3/50 unique frozen states (6%). The other 376/400 candidate
    state/finger/ramp combinations correctly remained
    `SUPPORT_GATE_NOT_REACHED`.
22. **Both-resource recovery:** 0/0; not attempted. Multiple-state one-finger
    success met the optional-stage precondition, but expansion was reserved for
    PI review.
23. **A retention after one release:** valid recoveries retained A through the
    ramp and every post-release checkpoint. Across all 24 attempts, retained
    and contact-free counts at 10/25/50/100/200/300/500/750/1000 steps were
    11/10/10/8/8/8/8/8/8.
24. **A retention after both releases:** not measured because second release
    was not executed.
25. **Post-release survival:** all 8 valid recovery events remained valid at
    10, 25, 50, 100, 200, 300, 500, 750, and 1000 steps.
26. **Successful support topologies:** recurring sampled topologies included
    `{index}`, `{thumb,index,ring}`, `{index,ring}`, `{thumb,index}`,
    `{thumb,index,middle}`, `{ring}`, `{middle}`, and `{index,middle}`. The
    released thumb can be absent while index plus storage support retains A.
27. **Failed topology:** the empty topology dominated failed-trial samples,
    followed by isolated `{index}`, `{thumb,index}`, and isolated `{thumb}`;
    alternate support generally failed to persist through endpoint retention.
28. **Load-transfer curves:** the stored per-step curves show thumb+index load
    decreasing while ring-dominated alternate load persists in representative
    thumb-first successes. Abrupt transient exchanges remain visible during the
    ramp; no smoothing or scalar score was applied.

## 29-38: Safety measurements and interpretation

29. **Penetration:** per-capture maximum median 0.006490 m, mean 0.005971 m,
    p95 0.007784 m, p99 0.008357 m, maximum 0.008884 m (N=2326). For the three
    recovery-source captures: 0.002363, 0.002677, and 0.003226 m. Thus two were
    below the inherited 0.003 m reference and one was slightly above; the
    reference was not altered or silently promoted to a success criterion.
30. **Joint margins:** per-trial minimum median -0.002077 rad, p95 -0.001065,
    minimum -0.010167 rad. These small compiled-state limit excursions are
    reported, not repaired through physics or limit changes.
31. **Actuator clipping:** per-trial maximum clipping-count median 2, p95 3,
    p99 4, maximum 4 of 20 actuators.
32. **Corridor audit:** coordinate frames were consistent; exact actual-path
    clearance was 42.815-48.959 mm and unused-finger contact count was zero.
    Prior negative candidate-path values were conservative due to bounding
    spheres and straight-path prediction, not an implementation bug.
33. **Did wrist materially improve capture?** Direction-specific yes, generic
    no. `[+5,-5]` added 39 matched successes (+11.143 percentage points) and
    `[+5,+5]` added 22 (+6.286 points), while both negative-WRJ2 probes were
    neutral or worse. The W1 range remains diagnostic.
34. **Did simultaneous control materially improve capture?** No on the frozen
    capture definition: 23 vs 22, treatment-only/control-only 2/1, matched risk
    difference +0.286 percentage points, exact p=1.0. Endpoint A retention did
    rise from 24/350 to 39/350, but that did not translate into a material
    coordinated-capture difference.
35. **Secure A storage demonstrated?** Yes, in three predefined
    wrist-load-transfer states with subsequent valid release trials.
36. **At least one acquisition resource recovered?** Yes: thumb recovery in
    three matched states and eight ramp conditions, all surviving 1000 steps.
37. **Classification:** **CC-A**. Coordinated storage/wrist/load handoff retains
    A while one acquisition finger becomes usable across multiple matched
    states without fixture or floor support.
38. **Remaining blocker:** general, safety-qualified reproducibility. Only 3/50
    states entered the release stage, index-first recovery was absent, one of
    three recovery-source captures slightly exceeded the inherited penetration
    reference, and joint/command boundary excursions remain measurable.

## 39-46: Progression, validation, and artifacts

39. **May object B be introduced?** Not automatically. The CC-A mechanism is
    demonstrated, but the protocol leaves adequate reproducibility and
    "penetration remains sane" as PI decisions without hard thresholds.
40. **May wrist-assisted insertion be studied?** Only after the PI accepts the
    progression gate; no B experiment was opened here.
41. **Is Phase 3C-1 RL justified?** Not yet as an autonomous action. The
    scripted mechanism now exists, but final progression/safety criteria remain
    deliberately unfrozen.
42. **Pytest:** `167 passed, 7 warnings in 25.94s`. The warnings are the six
    existing Gymnasium infinite-bound notices and a non-fatal pytest cache
    permission warning.
43. **`git diff --check`:** passed with no whitespace errors; Git emitted only
    the informational LF-to-CRLF working-copy warning for `phase3c0.py`.
44. **Figures:** 16 vector PDFs in `docs/figures/phase3C05/`, covering all
    requested diagnostics and an actual CC-A sequence.
45. **Videos:** actual-state replays under `outputs/phase3C05/videos/`; the
    manifest identifies the exact condition, state, subset, wrist command,
    release finger, and ramp for each file.
46. **Artifact paths:** machine results are
    `outputs/phase3C05/phase3c05_results.json`,
    `outputs/phase3C05/analysis_summary.json`,
    `outputs/phase3C05/failure_handoff_audit.json`, and
    `outputs/phase3C05/corridor_metric_audit.json`; frozen states are under
    `outputs/phase3C05/matched_states/`; per-step arrays are under
    `outputs/phase3C05/timeseries/`; reports are the four Phase 3C-0.5 audit,
    result, and TODO documents; figures and videos are in their corresponding
    Phase 3C-0.5 directories.

### Exact figure paths

- `docs/figures/phase3C05/phase3C0_failure_timeline.pdf`
- `docs/figures/phase3C05/coordinated_capture_concept.pdf`
- `docs/figures/phase3C05/acquisition_vs_alternate_load.pdf`
- `docs/figures/phase3C05/storage_subset_comparison.pdf`
- `docs/figures/phase3C05/fixed_vs_wrist_assisted_capture.pdf`
- `docs/figures/phase3C05/gravity_in_palm_during_capture.pdf`
- `docs/figures/phase3C05/wrist_pose_vs_support.pdf`
- `docs/figures/phase3C05/support_topology_timeline.pdf`
- `docs/figures/phase3C05/load_share_gate_analysis.pdf`
- `docs/figures/phase3C05/release_ramp_analysis.pdf`
- `docs/figures/phase3C05/thumb_first_vs_index_first.pdf`
- `docs/figures/phase3C05/post_release_survival.pdf`
- `docs/figures/phase3C05/serial_vs_simultaneous_capture.pdf`
- `docs/figures/phase3C05/serial_vs_wrist_coordinated_capture.pdf`
- `docs/figures/phase3C05/corridor_metric_audit.pdf`
- `docs/figures/phase3C05/representative_success_sequence.pdf`

### Exact video paths

- `outputs/phase3C05/videos/original_phase3C0_loss.mp4`
- `outputs/phase3C05/videos/simultaneous_storage_finger_capture.mp4`
- `outputs/phase3C05/videos/wrist_assisted_capture.mp4`
- `outputs/phase3C05/videos/successful_one_finger_recovery.mp4`
- `outputs/phase3C05/videos/failed_early_release.mp4`
- `outputs/phase3C05/videos/failed_wrist_orientation.mp4`
- `outputs/phase3C05/videos/representative_CC-A_success.mp4`
