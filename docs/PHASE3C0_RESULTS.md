# Phase 3C-0: Open Corridor and Palmar Storage Diagnostics

## Scope and provenance

Phase 3C-0 is a scripted, non-learning diagnostic on the official MuJoCo
Menagerie right Shadow Hand E3M5. It adds a separate state/action interface,
time-varying finger roles, palm-frame storage geometry, conservative swept-path
corridors, dynamic aperture geometry, palm-frame gravity, and a two-object
support graph. It does not alter the official MJCF, physics, gravity, historical
environments, PPO code, rewards, or checkpoints.

Branch: `codex/phase3C0-open-corridor-palmar-storage`

Base commit: `6777ca0800520238978088be249ddaef23a61dd2`

The formal run used six matched initial A poses per condition. Every pose began
from the compiled `open hand` keyframe (all 24 hand coordinates equal to zero),
with thumb/index as the attempted minimal acquisition set. The object pose was
chosen within the established hand workspace while rotating the existing
ellipsoid 90 degrees about its x axis; this cleared middle/ring/little in all
12 compiled initial states without changing object geometry or physics.

## Ordered experiment outcome

### C0-A - single-object open-corridor transfer

- Attempts: 6.
- Valid thumb+index contact before fixture release: 6/6.
- Middle/ring/little contact at the initial open-hand state: 0/6.
- Configured storage-region entry: 6/6.
- Secure storage after delayed closing and acquisition-finger release: 0/6.
- Retention-gated thumb recovery: 0/6; raw final thumb unloading was 6/6.
- Retention-gated index recovery: 0/6; raw final index unloading was 6/6.
- Multi-resource recovery: 0/6.

The storage-entry trigger occurred at simulation steps 385, 386, 387, 387,
388, and 405 (median 387, or 0.774 s from the diagnostic clock). The transition
from `CLEARING_CORRIDOR` to `SECURING_STORAGE` was issued at exactly the same
step. This is a logged engineering trigger, not a finalized scientific timing
criterion.

### C0-B - matched old-early-support versus open-corridor ablation

| Raw metric | Old early support | Open corridor |
|---|---:|---:|
| matched attempts | 6 | 6 |
| storage-region entries | 6/6 | 6/6 |
| secure-storage outcomes | 0/6 | 0/6 |
| resource-recovery outcomes | 0/6 | 0/6 |
| gross-collision sample count | 25 | 29 |
| minimum conservative corridor clearance | -40.231 mm | -38.844 mm |
| median per-trial minimum clearance | -34.587 mm | -32.292 mm |

The swept-sphere diagnostic identified the middle-finger link as the
conservative bottleneck in every matched initial corridor. The open condition
improved median geometric clearance by 2.295 mm, but did not reduce the sampled
gross-collision count and did not produce secure retention. Therefore this run
does not demonstrate that delayed closure materially reduces obstruction.
Likewise, the old controller was permitted to recruit middle at step 380, but
no actual non-acquisition contact occurred during the sampled transfer segment;
it is not scientifically valid to label those trials as measured early-contact
obstruction events.

### C0-C through C0-F - prerequisite gate

C0-C (stored-A wrist search), C0-D (aperture relaxation), C0-E (B insertion),
and C0-F (A+B resecure) were not run. No open-corridor trial produced a secure,
dynamically retained A state after resource release. Proceeding would have
violated the required experiment order and the instruction that B is allowed
only after validating the first-object storage mechanism.

Consequently:

- wrist configurations tested with securely stored A: 0;
- retention-preserving insertion-corridor wrist poses: 0 measured;
- B acquisition attempts: 0;
- B insertion attempts/successes: 0/0;
- A+B resecure attempts/successes: 0/0;
- simultaneous two-object retention: not observed;
- aperture-relaxation effect on A: not measured.

## Raw geometric observations

The initial open-hand aperture (middle/ring/little plus palm nodes) had width
168.118 mm, height 44.631 mm, and minimum node spacing 22.361 mm. At the final
failed-A state of the nominal open trial, those values were 158.900 mm,
52.122 mm, and 21.582 mm. These values demonstrate that aperture geometry is
state dependent; they do not demonstrate a controlled relaxation maneuver.

Across the recorded open-corridor trajectories, gravity expressed in the palm
frame ranged as follows: x approximately 0, y 7.952 to 9.810 m/s^2, and z 0 to
5.745 m/s^2. World gravity remained exactly `[0, 0, -9.81]` m/s^2. No insertion
direction was fixed: the interface derives candidate directions from current
aperture geometry and palm orientation.

The separate two-object model compiles two free ellipsoids, one-way acquisition
fixtures, and an identity-preserving bipartite support graph. Its API rejects
any attempt to set an object's pose after that object's fixture is released.
This architecture was tested, but no two-object dynamic experiment was opened
after the failed secure-A gate.

## Failure interpretation and next step

The exact remaining blocker before Phase 3C-1 is acquisition-to-storage support
capture: A enters the diagnostic palm-frame region, but thumb/index support is
lost before delayed middle/ring/little closure establishes a secure retained
state. The current evidence does not justify RL, reward weights, a scalar
objective, a new threshold, or object B.

Recommended next experiment: remain in scripted, single-object diagnostics and
characterize the support-capture interval around storage entry using the same
physics and raw criteria. Compare state-triggered storage-finger approach
profiles while reporting contact onset, palm-frame pose, support load, clearance,
and retention. A PI must select final storage-region bounds, corridor validity,
relaxation limits, and multi-object success thresholds before those become
scientific criteria.

## Artifact inventory

Machine-readable results are in `outputs/phase3C0/phase3c0_results.json`; the
visual manifest is `outputs/phase3C0/visual_manifest.json`. Sixteen vector PDFs
are under `docs/figures/phase3C0/`. Three MP4s show actual trajectories: the
single-object open-corridor attempt, the old-style matched failure, and an A-loss
example. Success-specific, wrist, aperture-relaxation, B-insertion, A+B-resecure,
and B-collision videos were deliberately omitted because those trajectories do
not exist.

## Validation

- `.\.venv\Scripts\python.exe -m pytest -v`: 151 passed, 7 warnings in
  33.22 s. The warnings are the existing Gymnasium infinite-bound notices plus
  a non-fatal pytest cache permission warning.
- `git diff --check`: passed with no output.
