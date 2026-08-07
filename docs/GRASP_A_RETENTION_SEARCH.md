# Object-A Unsupported-Retention Engineering Search

## Scientific boundary

This work is explicitly `engineering_search_only: true`. `engineering_retention_score` is a reproducible feasibility-search ranking tool built from raw simulated behavior. It is not a publication-level grasp-success metric, reward, probability, resource metric J, or PI-approved threshold. No trial is assigned a scientific success/failure label.

## Baseline and failure mechanism

The baseline score is -1.719923. Its release-time object contacts are middle plus thumb with nearly orthogonal inward normals. Middle contact is lost after 0.032–0.044 s, thumb contact after 0.534–0.892 s, and the object moves downward by 0.11757–0.14070 m. The full mechanical sequence is documented in `GRASP_A_FAILURE_ANALYSIS.md`.

## Search method and variables

The search evaluates 128 deterministic Latin-hypercube candidates using seed 20260807 and screening placement seed 0. Physics, hand model, object geometry/mass/friction, gravity, solver, hand mount, and fixture pose remain fixed.

The YAML-owned variables are:

- target fraction within each configured actuator's joint range;
- one post-release hold-target delta per configured finger group;
- one closing delay per configured finger group, producing different closing sequences;
- closing duration;
- fixture-held contact-establishment duration, which changes support-release time.

Every target fraction is clipped to `[0, 1]`, control is clipped to ±1 N m by the existing controller, and a candidate with more than 1e-6 rad joint-limit excess receives an engineering safety rejection score. No reinforcement learning or controller tuning is used.

## Engineering ranking formula

For the one-second unsupported observation window:

```text
engineering_retention_score =
    2 * terminal_height_retention
  +     mean_height_retention
  +     no_table_contact_time_fraction
  +     object_fingertip_contact_time_fraction
  +     mean_active_object_finger_fraction
  -     maximum_translation / 0.05 m
  - 0.5 * maximum_orientation_change / pi
  -     actuator_saturation_time_fraction
```

Height retention is the object center's remaining fraction of the release-to-table-resting height interval, clipped to `[0, 1]`. The constants and weights are declared in `configs/grasp_search_a.yaml` as engineering search settings. They are not reward weights or scientific thresholds.

## Selected candidates

The top five distinct screening postures are committed as complete, actuator-keyed YAML profiles.

| Config | Source ID | Screening score | Close (s) | Establish (s) | Release (s) |
|---|---|---:|---:|---:|---:|
| `grasp_A_candidate_01.yaml` | candidate_0072 | 5.296691 | 0.7394 | 0.3421 | 1.2815 |
| `grasp_A_candidate_02.yaml` | candidate_0077 | 5.249205 | 0.6548 | 0.2758 | 1.1306 |
| `grasp_A_candidate_03.yaml` | candidate_0097 | 5.221074 | 0.3870 | 0.3876 | 0.9747 |
| `grasp_A_candidate_04.yaml` | candidate_0064 | 5.218734 | 0.7326 | 0.3833 | 1.3159 |
| `grasp_A_candidate_05.yaml` | candidate_0001 | 5.149565 | 0.7193 | 0.3105 | 1.2298 |

## Twenty-seed descriptive validation

Each candidate was rerun for deterministic placement seeds 0–19. The following are raw distributions, not pass/fail rates.

| Candidate | Mean score (min–max) | Final vertical displacement m (min–max; mean) | Max translation m (mean) | Max rotation rad (mean) | Contact duration s (min–max) | Runs without table re-contact / complete contact loss |
|---|---|---|---:|---:|---|---:|
| 01 | 5.2050 (4.6332–5.4321) | -0.02281–-0.00384; -0.00773 | 0.00915 | 0.2691 | 1.000–1.000 | 20 / 20 |
| 02 | 5.2334 (5.1166–5.3571) | -0.01144–-0.00708; -0.00972 | 0.01973 | 0.4684 | 1.000–1.000 | 20 / 20 |
| 03 | 5.1948 (5.1648–5.2234) | -0.01088–-0.01005; -0.01050 | 0.01459 | 0.3508 | 1.000–1.000 | 20 / 20 |
| 04 | 2.4893 (-2.8791–5.2643) | -0.14012–-0.00489; -0.05303 | 0.05694 | 0.5933 | 0.036–1.000 | 13 / 13 |
| 05 | 5.1611 (5.0990–5.2217) | -0.00835–-0.00663; -0.00737 | 0.01891 | 0.4162 | 1.000–1.000 | 20 / 20 |

Candidates 01, 02, 03, and 05 preserve at least one object-fingertip contact and avoid object/table contact throughout every one-second unsupported trace. Candidate 04 is placement-sensitive: thirteen traces show the same full-window pattern, while seven show contact loss and table re-contact. These are finite-window physical observations, not scientific declarations of grasp success.

## Finger participation and object-specific force

- Candidate 01 begins with opposed index/middle contacts; thumb joins during free motion. Mean active object fingers across seeds is 2.357.
- Candidate 02 begins with index, middle, and thumb; ring joins and the mean active count is 3.659. Across seeds, mean object-normal forces averaged 1.345 N index, 0.277 N middle, 1.539 N ring, and 1.359 N thumb.
- Candidate 03 maintains an opposed three-digit index/middle/thumb set with mean active count 2.991. Mean object-normal forces averaged 1.784, 6.551, 0, and 3.504 N respectively.
- Candidate 04 relies mainly on opposed index/middle contacts and is sensitive to placement.
- Candidate 05 maintains index/middle/thumb with mean active count 2.998; ring remains inactive.

The viable candidates differ materially in posture and force distribution, which is useful for later PI analysis. The search does not collapse these raw states into resource metric J.

## Physical plausibility checks

- Object A retains its ordinary six-DoF free joint after support release; no weld is introduced.
- The model contains zero equality constraints (`model.neq == 0`).
- `support_active` is zero on every post-release sample, gravity remains `[0, 0, -9.81]` m/s², and object/table contact is independently logged.
- Object mass, size, friction, hand MJCF, solver, and collision geometry are unchanged.
- All 100 top-candidate validation runs contain finite state/control arrays, zero configured actuator saturation, zero joint-limit excess, and no numerical early termination.
- Fixture-held closing can create transient contact penetration. This is measured rather than ignored. Candidate 02's worst post-release contact distance across seeds is -12.745 mm near release, but by the final sample its active contact depths range from 0.328 to 2.181 mm (mean 0.952 mm). The object remains clear of the table while the initial overlap relaxes, so its full-window behavior is not solely an instantaneous release-interpenetration artifact. Candidates 01/03/05 retain somewhat deeper final contacts and should be treated as less clean physical references.
- The official MuJoCo renderer and encoder produced one MP4 for each selected candidate.

## Generated diagnostics

Gitignored outputs include 128 screening NPZ trajectories, 100 validation NPZ trajectories, five raw top-candidate runs, five MP4 videos, 60 per-candidate plots, and six comparison figures:

- baseline vs top object height;
- baseline vs top vertical velocity;
- baseline vs top object-fingertip active count;
- baseline vs top configured-fingertip force;
- all top-candidate multi-seed height traces;
- top-candidate final vertical-displacement distributions.

## Remaining PI decisions

The PI must still define scientific grasp acquisition, required unsupported hold duration, acceptable translation/orientation, table-clearance semantics, required finger/contact continuity, handling of temporary contact loss, acceptable contact-model penetration for scientific use, and loss/drop criteria. Resource metric J, final tactile features, reward weights, active retention control, sequential behavior, and PPO remain unresolved and unimplemented.
