# Phase 3B PI Decisions Required

Status: decision packet only. Phase 3B is not implemented and no RL has been
trained. The task remains single-object Shadow-Hand contact handoff. This packet
does not introduce object B, define scalar resource metric `J`, change physics,
or authorize a merge to `main`.

Source baseline: Phase 3A commit `27bc248` on
`codex/phase3A-shadow-hand-contact-handoff`.

## How to read the tables

Every recommendation below is **RECOMMENDATION ONLY**. It is not an adopted
scientific decision. "Freeze" means the PI must choose before the first RL run
whose results could be interpreted scientifically. "Sensitivity" means the
choice can also be varied later after a primary setting is frozen.

The location column identifies the current code or config whose placeholder
would be replaced or parameterized. The broad TODO at
`configs/phase3A_shadow_hand.yaml:39` intentionally left several thresholds
undefined; those questions are separated below.

## A. Success / secure criteria

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| A1 | What constitutes a valid initial thumb-index acquisition? `configs/phase3A_shadow_hand.yaml:39`; `seqgrasp/phase3/experiments.py:135-152` | Phase 3A labels a state successful when dual contact persists for the 0.5 s diagnostic window without floor contact and release penetration is below the historical reference. | It defines the reset population and denominator for learned handoff success. | Require dual contact only (broad/easy); require dual contact plus force/opposition/retention (more physical); require perturbation survival (strongest but costly). | **RECOMMENDATION ONLY:** require physically sane opposed thumb-index contact plus unsupported retention, with all numerical limits calibrated from a larger Shadow acquisition dataset. | YES | YES |
| A2 | What evidence is sufficient for `PALMAR_SECURE`? `configs/phase3A_shadow_hand.yaml:39`; `seqgrasp/phase3/roles.py:59-61`; `seqgrasp/phase3/events.py:25-58` | The role exists, but Phase 3A makes no final scientific classification. | Release rewards and success must not fire from mere palm proximity or momentary touch. | Palm contact flag (exploitable); palm force/load plus alternate support (stronger); reduced-acquisition-load hold validation (most causal). | **RECOMMENDATION ONLY:** require alternate hand support and a release/hold validation under substantially reduced acquisition-finger load; do not use palm-region entry alone. | YES | YES |
| A3 | How long must post-release retention persist? `configs/phase3A_shadow_hand.yaml:39`; diagnostic `unsupported_steps` at `:27` | Phase 3A uses 250 simulation steps (0.5 s) only as a diagnostic. | Too short rewards transient catches; too long raises cost and may conflate handoff with long-horizon disturbance rejection. | Fixed time; randomized hold time; two-tier short training/long evaluation. | **RECOMMENDATION ONLY:** freeze separate training and evaluation durations, with the evaluation hold longer; select values from survival curves rather than adopting 0.5 s automatically. | YES | YES |
| A4 | What post-release translation is acceptable? `configs/phase3A_shadow_hand.yaml:39`; raw object pose in `seqgrasp/phase3/env.py:84` | Translation is recorded but has no success gate. | Unlimited drift could count a nearly lost or unintended regrasp as success. | No translation gate; global displacement gate; palm-relative displacement gate; task-conditioned envelope. | **RECOMMENDATION ONLY:** use a palm-relative displacement envelope calibrated on retained Phase 3A-style holds; report global displacement separately. | YES | YES |
| A5 | What post-release rotation is acceptable? `configs/phase3A_shadow_hand.yaml:39`; object quaternion in `seqgrasp/phase3/env.py:84` | Rotation is recorded but not gated. | Some rolling is desired, while uncontrolled tumbling indicates loss. | No gate; total rotation cap; angular-speed plus final-orientation envelope; object-symmetry-aware metric. | **RECOMMENDATION ONLY:** permit controlled rotation but gate sustained angular velocity and symmetry-aware final orientation; calibrate numerically from retained trials. | YES | YES |
| A6 | How much transient hand-object contact loss is acceptable? `configs/phase3A_shadow_hand.yaml:39`; `seqgrasp/phase3/events.py:25-58` | Event code accepts explicit booleans but defines no persistence. | Zero-tolerance may prohibit useful gaiting; long gaps can reward ballistic toss/catch behavior. | No loss allowed; bounded gap with continuous near-hand state; only require final retention. | **RECOMMENDATION ONLY:** allow short, explicitly bounded contact gaps only when the object remains within a hand-relative safety envelope and is subsequently retained. | YES | YES |
| A7 | Must secure support include the palm, or can fingers alone qualify? `configs/phase3A_shadow_hand.yaml:39`; support channels `seqgrasp/phase3/contacts.py` | Six raw support channels are available; no required topology is selected. | The stated task targets palmar handoff, while finger-only cages may release a digit without achieving it. | Require palm force; allow palm or multi-finger cage; use graded palm-load criterion. | **RECOMMENDATION ONLY:** primary success should require verified palm participation plus alternate support; retain finger-only recovery as a reported secondary outcome, not primary success. | YES | YES |

## B. Contact / physics criteria

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| B1 | What Shadow-Hand unsafe-penetration criterion replaces or confirms the 3 mm reference? `configs/phase3A_shadow_hand.yaml:22,39,53`; `seqgrasp/phase3/env.py:188-198` | 0.003 m is diagnostic only. The Phase 3A release distribution was 0.615-0.712 mm. | It controls reset validity, failure termination, and reward penalties. | Retain 3 mm; choose a Shadow-specific empirical percentile/margin; use pair-specific limits. | **RECOMMENDATION ONLY:** calibrate a Shadow-specific, pair-aware criterion from valid and visibly invalid contact states; keep 3 mm only as a reported historical reference until then. | YES | YES |
| B2 | Should intended fingertip/object solver overlap be separated from gross hand/table/object interpenetration? Same locations as B1; extraction `seqgrasp/phase3/contacts.py:26-76` | Raw penetration is available per semantic surface, but the maximum is undifferentiated for reward use. | Treating compliant grip overlap as gross collision can recreate the Phase 2CM gate problem. | One global maximum; intended-versus-invalid pair classes; per-pair continuous penalties. | **RECOMMENDATION ONLY:** retain raw pair-level penetration and distinguish intended gripping interfaces from table, palm-through-object, and other gross collisions; PI must approve the categories and limits. | YES | YES |
| B3 | Does fingertip/palm `condim=3` remain the primary contact model? Audit `docs/PHASE3A_SHADOW_HAND_AUDIT.md`; model source `assets/hands/shadow_right/right_hand.xml` | Official Menagerie baseline compiles to `condim=3`; Phase 3A did not override it. | Changing contact dimensionality changes torsional/rolling resistance and policy behavior. | Freeze official CM3 baseline; use CM4/CM6; train multiple physics models. | **RECOMMENDATION ONLY:** freeze official CM3 as the primary baseline for the first feasibility run because Phase 3A succeeded under it; do not infer superiority. | YES | YES |
| B4 | Is a contact-model ablation required before the first RL run? Same audit/model locations | No Shadow contact-model ablation exists. | An ablation may improve interpretation but delays the minimal feasibility experiment. | Require pre-RL CM3/4/6 characterization; run CM3 first and ablate later; domain-randomize condim (not directly meaningful). | **RECOMMENDATION ONLY:** do not block the minimal CM3 feasibility run; require a matched post-baseline CM3-versus-approved-alternative sensitivity before broad scientific claims. | NO | YES |
| B5 | Which mechanical events are immediate failures? `configs/phase3A_shadow_hand.yaml:39`; raw state in `seqgrasp/phase3/env.py`; events in `seqgrasp/phase3/events.py` | Complete loss/table contact/numerical failure are representable, but final gates and persistence are absent. | Failure semantics determine termination, credit assignment, and reported success. | Terminate on first event; permit recoverable near-loss; separate training termination from evaluation failure labels. | **RECOMMENDATION ONLY:** always terminate numerical failure, gross unsafe penetration, workspace exit, and confirmed table drop; separately define bounded recoverable contact gaps. | YES | YES |

## C. Resource-recovery criteria

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| C1 | What conditions make a finger "recovered"? `configs/phase3A_shadow_hand.yaml:39,51`; `seqgrasp/phase3/events.py:45-54`; `seqgrasp/phase3/resource.py:72-94` | Diagnostic logic requires no contact, alternate support, retention, and positive motion; final thresholds are absent. | Contact flag zero alone can reward dropping or a jammed finger. | Contact-free only; contact-free plus role; retention plus alternate support plus available motion. | **RECOMMENDATION ONLY:** require verified object retention, alternate support, zero released-finger contact for a persistence window, and meaningful available motion. | YES | YES |
| C2 | What minimum available motion is meaningful? `configs/phase3A_shadow_hand.yaml:39`; `seqgrasp/phase3/resource.py:34-70` | Phase 3A reports raw joint range and a local Jacobian workspace envelope; any positive range passes the diagnostic. | A nominally free but kinematically trapped digit is not a usable resource. | Joint-margin threshold; actuator-range threshold; task-relevant reachable-workspace threshold; combined criterion. | **RECOMMENDATION ONLY:** use a task-relevant local reachable-workspace criterion with joint/actuator margins as safety checks; choose values from held-out probing data. | YES | YES |
| C3 | Is recovery of one acquisition finger sufficient? `configs/phase3A_shadow_hand.yaml:39`; Phase 3A releases thumb only in `:30` | Proof of concept counts one released acquisition finger. | Requiring both is harder and may be unnecessary for the first single-object objective. | At least one; specifically thumb; specifically index; both. | **RECOMMENDATION ONLY:** primary Phase 3B success should require at least one acquisition finger; report both-finger recovery as a stricter secondary endpoint. | YES | YES |
| C4 | Should thumb and index recovery be valued/evaluated differently? Semantic identities in `seqgrasp/phase3/config.py`; resource state in `resource.py` | Identity is preserved, but no preference or equivalence is defined. | Thumb opposition and index dexterity make the two resources non-equivalent. | Treat equally; separate endpoints; task-conditioned value. | **RECOMMENDATION ONLY:** keep separate thumb-recovered and index-recovered metrics and avoid collapsing them to one count; do not choose a scalar preference yet. | YES | YES |
| C5 | What resource representation should be used without scalar `J`? `seqgrasp/phase3/resource.py` | Structured per-finger state and identity-preserving mask are available; scalar `J` remains undefined. | A count can hide which opposable/manipulative digit is available. | Count only; five-bit identity mask; structured contact/force/workspace state. | **RECOMMENDATION ONLY:** retain the structured identity-preserving representation and `N_free` only as an auxiliary statistic; do not define scalar `J` in Phase 3B. | YES | YES |

## D. RL observation design

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| D1 | Which of the current 127 actor values are deployable? `seqgrasp/phase3/env.py:71-90` | q/qvel, explicit wrist state, contact flags/forces, palm signal, role/phase one-hot, and previous action are actor-visible. | Non-deployable or redundant features can invalidate the asymmetric claim. | Use all 127; remove duplicate wrist; remove semantic roles/phases; use a sensor-realistic subset. | **RECOMMENDATION ONLY:** freeze a sensor-realistic proprioception/contact/previous-action actor vector and document every assumed sensor; audit duplicated wrist state before training. | YES | YES |
| D2 | Should finger roles and high-level phase be actor inputs or internal diagnostic labels? `seqgrasp/phase3/env.py:81-82`; `seqgrasp/phase3/roles.py` | Both are actor-visible one-hot vectors. | Hand-authored phase labels may leak a scripted curriculum or make deployment depend on an oracle. | Actor-visible; critic-only; inferred recurrently; remove roles but keep phase command. | **RECOMMENDATION ONLY:** treat curriculum stage as an explicit task command if needed, but keep inferred finger roles diagnostic/critic-only unless a deployable estimator is defined. | YES | YES |
| D3 | Is short temporal history needed? `docs/PHASE3A_TODO_PI.md`; observation construction `seqgrasp/phase3/env.py` | Current actor is Markov-style current state plus previous action; no history stack. | Contact migration and slip may be partially observable from instantaneous forces. | No history; fixed frame stack; recurrent actor. | **RECOMMENDATION ONLY:** begin with a small fixed history ablation against the no-history actor; use recurrence only if the simpler comparison shows a clear need. | YES | YES |
| D4 | What privileged critic state is allowed? `seqgrasp/phase3/env.py:84-90` | A 50-value diagnostic dictionary includes object-palm pose, velocities, contact counts, support/load, penetration, and geometry relations. | The critic must aid learning without leaking into deployed actor evaluation. | Current 50; reduced dynamics-only state; full simulator state. | **RECOMMENDATION ONLY:** use the current semantically structured 50-D critic state first, with strict actor/critic interface tests and normalization. | YES | YES |
| D5 | Should a privileged-actor diagnostic baseline receive object state? Same location as D4 | No privileged actor implementation exists. | It separates physics/control feasibility from partial-observability difficulty. | Actor gets full 50 privileged state; actor gets only object pose/velocity; omit baseline. | **RECOMMENDATION ONLY:** include one diagnostic actor with object-palm pose and object velocity plus the deployable actor inputs; label it non-deployable and compare matched seeds. | YES | YES |
| D6 | How are observations normalized, clipped, delayed, and noised? `seqgrasp/phase3/env.py:61-90` | Observation bounds are infinite; raw forces and geometry are returned. | Scale imbalance and unrealistically perfect signals can dominate PPO or overstate deployment readiness. | Fixed engineering normalization; running normalization; sensor-calibrated scaling/noise; no noise in feasibility baseline. | **RECOMMENDATION ONLY:** freeze deterministic unit-based scaling for the primary run; add delay/noise only as later robustness tests after a clean feasibility baseline. | YES | YES |

## E. RL action design

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| E1 | Is the current 26-D action the primary interface? `seqgrasp/phase3/control.py:37-92`; `seqgrasp/phase3/env.py:60` | 20 actuator-target displacements plus six wrist/finger-group stiffness scales. | It defines exploration dimension and ability to permit controlled sliding. | 20-D fixed impedance; current 26-D; higher-dimensional per-actuator gains. | **RECOMMENDATION ONLY:** use the current 26-D interface as primary and retain 20-D fixed impedance as an ablation. | YES | YES |
| E2 | What actuator displacement bound is appropriate? `configs/phase3A_shadow_hand.yaml:42`; `seqgrasp/phase3/control.py:74-80` | Per-step normalized action maps to at most 0.04 actuator-coordinate displacement. | Too large causes violent contact; too small prevents timely transfer. | Retain 0.04; smaller/larger fixed bounds; phase-dependent bounds. | **RECOMMENDATION ONLY:** calibrate from stable scripted trajectories and actuator-rate distributions; freeze one global bound before primary training. | YES | YES |
| E3 | What stiffness-scale bounds are appropriate? `configs/phase3A_shadow_hand.yaml:43`; `seqgrasp/phase3/control.py:62-88` | Scale is clipped to `[0.2, 1.0]`. | Lower stiffness enables sliding but can lose support; high stiffness can cause penetration/impacts. | Current range; narrower range; allow above nominal; fixed impedance. | **RECOMMENDATION ONLY:** keep nominal as the upper limit for the first run and calibrate the lower limit using stable controlled-slip diagnostics; do not allow above-nominal gains initially. | YES | YES |
| E4 | Should stiffness remain per semantic finger/wrist group or become per actuator? Same locations as E1-E3 | Six scales: wrist plus five fingers. | Per-actuator control is expressive but greatly expands unsafe exploration and identifiability problems. | Six groups; five fingers plus fixed wrist; 20 per-actuator gains; one global gain. | **RECOMMENDATION ONLY:** retain six semantic groups for the primary experiment. | YES | YES |
| E5 | Is fixed impedance required as a formal baseline? `configs/phase3A_shadow_hand.yaml:44`; `seqgrasp/phase3/env.py:35-50` | Compatibility mode exists. | It tests whether learned stiffness modulation materially contributes. | No baseline; fixed-only; matched fixed-versus-variable comparison. | **RECOMMENDATION ONLY:** run a matched fixed-impedance ablation after establishing the primary pipeline. | NO | YES |
| E6 | What action-rate/violent-motion constraint is used? `configs/phase3A_shadow_hand.yaml:55`; raw term `seqgrasp/phase3/env.py:199` | Action difference norm is measured; no weight, rate limit, or scientific threshold is set. | PPO may exploit impulses or high-frequency gain switching. | Hard rate limit; soft cost; actuator-specific limits; both hard safety and soft smoothness. | **RECOMMENDATION ONLY:** impose hardware-consistent hard rate bounds and retain a separately reported smoothness term; PI must approve normalization and any reward use. | YES | YES |

## F. RL reward design

All current weights in `configs/phase3A_shadow_hand.yaml:46-55` are `0.0`.
No weight is proposed or selected here. `seqgrasp/phase3/rewards.py:23-25`
returns a zero weighted sum until the PI freezes a formulation.

| ID | Component/location | Intended behavior | Main exploit | Density / potential shaping / contact gate | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|---|
| F1 | Object progress toward palm, config `:46` | Encourage controlled palmward migration. | Dropping, throwing, or squeezing the COM toward the palm without support. | Dense; palm-relative distance-difference can be potential-based; **must be gated by valid hand-object contact**. | Ungated distance (unsafe); gated potential difference (less path bias); sparse waypoint event (harder credit). | **RECOMMENDATION ONLY:** use only contact-gated palm-relative potential difference; never reward distance alone. | YES | YES |
| F2 | Valid support, config `:47` | Preserve at least one physically valid hand-object support structure. | Maximizing force/contact count or crushing with all fingers. | Dense or event-based; not naturally potential-based; gate by valid pair classes and safety. | Force sum (exploitable); capped support evidence; topology/event bonus. | **RECOMMENDATION ONLY:** use capped, validity-gated support evidence without increasing credit for redundant contacts or force beyond sufficiency. | YES | YES |
| F3 | Palm contact, config `:48` | Establish the intended palmar interface. | Brief palm tap, object-palm collision, or high-force impact. | Sparse onset plus persistence; potential shaping only through a separate safe proximity potential; gate by retention and safe penetration. | Contact flag; force persistence; load-bearing palm support. | **RECOMMENDATION ONLY:** credit persistent, safe, load-bearing palm contact rather than first touch. | YES | YES |
| F4 | Support transfer, config `:49` | Shift load from acquisition fingers to palm/support fingers. | Artificially unloading thumb/index by dropping or eliminating all support. | Dense load-fraction change or sparse event; potential-based on a validated support potential; gate by total valid support and retention. | Raw non-acquisition fraction; fraction increase; thresholded shifted event. | **RECOMMENDATION ONLY:** use change in support-load allocation only while total valid support and retention conditions hold. | YES | YES |
| F5 | Acquisition-finger release after support, config `:50` | Release a digit only after an alternate structure exists. | Opening early, momentary alternate contact, or dropping after release. | Sparse event; not naturally potential-based; **must be gated by prior alternate support and subsequent hold validation**. | Immediate event; delayed validated event; terminal-only credit. | **RECOMMENDATION ONLY:** delayed validated event credit after secure-support evidence and post-release retention. | YES | YES |
| F6 | Recovered resource, config `:51` | Create a genuinely usable free acquisition finger. | Zero contact with jammed joints, finger flinging, or object loss. | Sparse validated event plus optional dense workspace proxy; potential shaping possible for workspace margin; gate by retention. | Free mask; joint margin; reachable workspace; combined validated event. | **RECOMMENDATION ONLY:** credit the combined retained-object/resource criterion; keep workspace values separately observable. | YES | YES |
| F7 | Complete object loss, config `:52` | Strongly reject loss and table drop. | If delayed or weak, policy may trade drops for progress/release reward. | Sparse terminal failure; not potential-based; no contact gate. | First confirmed loss; table contact; workspace exit; all as distinct labels. | **RECOMMENDATION ONLY:** terminal failure with distinct reported subreasons; PI selects persistence for contact-free loss. | YES | YES |
| F8 | Unsafe penetration, config `:53` | Reject gross invalid overlap. | Penalizing intended solver overlap can prevent grasping; a max-only metric can hide pair identity. | Dense excess plus terminal gross violation; not necessarily potential-based; pair-class gate required. | Global max; pair-specific excess; intended/invalid classes. | **RECOMMENDATION ONLY:** pair-aware excess penalty plus terminal gross-invalid gate after B1/B2 are frozen. | YES | YES |
| F9 | Joint-limit penalty, config `:54` | Preserve controllability and prevent limit impacts. | Over-penalizing necessary opposition or preferring open hand. | Dense smooth barrier; potential-based barrier difference possible; gate not required. | Distance barrier; velocity-at-limit penalty; hard safety margin. | **RECOMMENDATION ONLY:** smooth normalized margin barrier with a separate hard compiled-limit safety check. | YES | YES |
| F10 | Violent-action penalty, config `:55` | Discourage impact-generating target/gain changes. | Excessive smoothing can suppress necessary release or controlled slip. | Dense; potential-based only with augmented previous-action state; gate not required. | Action norm; action difference; actuator-rate/energy proxy. | **RECOMMENDATION ONLY:** use normalized action-rate and gain-rate measurements, paired with E6 hard limits; do not penalize tangential object motion itself. | YES | YES |
| F11 | Combined reward signs, scales, phase gates, and weights, `seqgrasp/phase3/rewards.py:23-25` | Produce learnable credit without changing the scientific endpoint. | One dense term can dominate; phase bonuses can be farmed; contact count/all-finger closure can emerge. | Mixture; some potential-based terms; all event gates explicit. | Flat weighted sum; phase-gated sum; constrained/lexicographic safety plus shaped objective. | **RECOMMENDATION ONLY:** freeze safety failures independently, use only approved gated components, log every raw term, and choose weights through a documented pilot—not by optimizing test success. **No weights are selected here.** | YES | YES |

Reward invariants that must remain true under every option:

- COM-to-palm distance alone earns no reward.
- Palmward progress requires valid hand-object contact.
- Acquisition-finger release earns no reward before alternate support.
- A free finger earns no credit if the object is dropped.
- Tangential motion is not intrinsically penalized; controlled slip/rolling remain allowed.
- Contact count and all-five-finger closure are not objectives.
- Minimal/adaptive recruitment is preferred over unconditional recruitment.
- Resource credit requires verified retention.
- Complete loss, table drop, unsafe penetration, numerical failure, and severe
  joint-limit violation remain failures after their PI-owned definitions are frozen.

## G. Training / curriculum

| ID | Exact scientific question and location | Current placeholder/default | Why it matters | Options and consequences | RECOMMENDATION ONLY | Freeze | Sensitivity |
|---|---|---|---|---|---|---|---|
| G1 | What initial reset distribution is used? Phase 3A states in `outputs/phase3A/acquisition_and_recruitment.json`; reset code `seqgrasp/phase3/env.py:146-167` | Environment rebuilds a fixture-held pre-grasp; no RL reset-state dataset exists. | Reset validity determines whether learning addresses handoff rather than collision recovery. | Seven Phase 3A states; larger controller-generated validated dataset; reset from pre-acquisition. | **RECOMMENDATION ONLY:** use a larger, frozen dataset of physically validated Shadow thumb-index release states; never reset into palmar-secured, Phase 2 Allegro, or high-penetration states. | YES | YES |
| G2 | How is the larger reset dataset generated and split? Same locations as G1; diagnostic controller `seqgrasp/phase3/experiments.py` | Seven deterministic candidates, five diagnostic successes. | Seven states are insufficient for generalization and risk train/test leakage. | Dense grid; seeded random sampling; adaptive boundary sampling; split individual states randomly or by pose regions. | **RECOMMENDATION ONLY:** seeded sampling around the compiled thumb-index workspace, contact-aware closure, fixture-release audit, unsupported validation, hash deduplication, and pose-region-disjoint train/validation/test splits. | YES | YES |
| G3 | Are curriculum stages 0-5 adopted, merged, or reordered? No Phase 3B code exists. | Suggested stages are conceptual only. | Stage definitions can embed the desired answer or block emergent valid strategies. | Fixed sequential stages; performance-triggered stages; goal-conditioned mixture; no curriculum. | **RECOMMENDATION ONLY:** PI-reviewable performance-gated stages 0-5 with raw criteria frozen first; retain a no-curriculum baseline if feasible. | YES | YES |
| G4 | What episode duration and control frequency are used? `configs/phase3A_shadow_hand.yaml:2-4` | 0.002 s physics, frame skip 5, 500 environment steps are scaffold values. | Horizon affects exploration, credit assignment, and survival claims. | Fixed current horizon; longer horizon; stage-specific horizon; randomized evaluation hold. | **RECOMMENDATION ONLY:** preserve physics timestep, freeze a task horizon from scripted stage timing, and use a longer post-release evaluation than training validation. | YES | YES |
| G5 | What terminates training and evaluation episodes? `seqgrasp/phase3/env.py:205-210`; B5/A criteria | Only time truncation is active; `terminated` is always false. | Without failure termination, PPO can exploit invalid trajectories and waste samples. | Terminate all failures; only hard safety failures; separate training/evaluation semantics. | **RECOMMENDATION ONLY:** immediate termination for hard mechanical/numerical failures; keep recoverable contact migration nonterminal; report exact reason. | YES | YES |
| G6 | Which PPO implementation/configuration is the baseline? `pyproject.toml` optional `stable-baselines3`; no Phase 3B trainer exists | PPO is suggested but not implemented. | Library, network, rollout, and reproducibility choices affect the baseline. | Stable-Baselines3; CleanRL-style auditable implementation; custom asymmetric PPO. | **RECOMMENDATION ONLY:** use a well-tested PPO implementation only if it cleanly supports separate actor/critic inputs; otherwise add the smallest auditable asymmetric extension with deterministic config capture. | YES | YES |
| G7 | What matched baselines are mandatory? No Phase 3B code exists. | Phase 3A scripted controller is the only behavior reference. | Learned success alone cannot separate observability, stiffness, and control feasibility. | Deployable actor only; add privileged actor; add fixed impedance; add scripted reference. | **RECOMMENDATION ONLY:** primary asymmetric actor-critic, privileged-actor diagnostic, fixed-impedance ablation, and frozen scripted Phase 3A reference on matched resets. | YES | YES |
| G8 | What seeds, sample budget, checkpoint selection, and stopping rule define a run? No Phase 3B config exists. | Undefined. | Post-hoc stopping or best-seed reporting biases results. | Fixed steps; fixed wall time; convergence rule; select final or validation-best checkpoint. | **RECOMMENDATION ONLY:** preregister multiple seeds, a fixed interaction budget, validation-only checkpoint selection, and report every seed. PI chooses numbers. | YES | YES |
| G9 | What evaluation and OOD split is frozen? No Phase 3B evaluator exists. | Undefined. | Training-distribution success does not establish generalization. | Random held-out states; pose-region-held-out; physics perturbations; all. | **RECOMMENDATION ONLY:** pose-region-disjoint in-distribution test plus prespecified OOD pose/orientation, object mass/friction, and observation-noise suites; do not tune on them. | YES | YES |

## Minimal Phase 3B training plan (proposal only)

### 1. Proposed scientific question

Can a learned policy reliably reproduce and generalize dynamic support handoff
and finger-resource recovery from physically valid Shadow-Hand fingertip grasps?

### 2. Proposed primary hypothesis

**PI-REVIEWABLE HYPOTHESIS:** Given physically valid thumb-index acquisition
states, a contact-aware policy with bounded semantic-group impedance control can
increase palm/alternate support, release at least one acquisition finger, and
retain the object more reliably on held-out valid grasps than matched fixed-
impedance and no-learning baselines.

This is not adopted until the PI freezes the success and comparison definitions.

### 3. Proposed training task

Single Shadow-Hand object only. Reset at the instant after a validated fixture
release into a real thumb-index grasp. The object remains fully dynamic. The
policy may migrate contact, recruit support progressively, establish palm
support, unload and release at least one acquisition finger, and must retain the
object through the validation window. No second object and no direct reset to a
palmar-secured endpoint.

### 4. Proposed curriculum

All stages are **PI-reviewable**, not frozen science:

1. Stage 0: maintain a valid fingertip grasp.
2. Stage 1: produce contact-gated palmward progress without loss or unsafe penetration.
3. Stage 2: establish a new support finger and/or valid palm support.
4. Stage 3: shift measured load away from one acquisition finger while total support remains valid.
5. Stage 4: unload and release one acquisition finger, then pass post-release retention.
6. Stage 5: train/evaluate across the frozen distribution of valid initial fingertip grasps.

Advancement should use PI-approved validation metrics, not training return alone.

### 5. Proposed actor observation

Start from deployable proprioception and contact sensing: hand joint positions
and velocities, wrist state, five finger contact flags and normal forces, palm
contact/force, previous action, and an approved task-stage command if the
curriculum requires it. Move inferred role labels out of the actor unless a
deployable estimator is specified. Compare no-history with a small fixed history.

### 6. Proposed critic observation

Use the existing structured privileged state: object pose relative to palm,
object linear/angular velocity, semantic contact counts, six-channel support
force and load fractions, pair-level penetration, and fingertip/palm-object
relations, in addition to actor observations. The critic state must never enter
the deployed actor evaluation path.

### 7. Proposed action space

Primary candidate: current 26-D bounded interface—20 actuator-coordinate target
increments plus six stiffness scales for wrist and five semantic finger groups.
Retain 20-D fixed-impedance compatibility as an ablation. Bounds remain PI-owned
and must be calibrated before training.

### 8. Proposed reward components without weights

Use only PI-approved, separately logged components F1-F10. No weights are set in
this packet. Required logic includes contact-gated palmward potential change,
valid-support evidence, persistent safe palm support, support-load transfer,
validated post-support finger release, retained-object resource recovery, and
separate failure/safety terms for loss, unsafe penetration, joint limits, and
violent action. Do not reward contact count, universal finger closure, release
alone, or COM-to-palm distance alone.

### 9. Proposed success-criterion structure

A logical conjunction, with all numerical values PI-owned:

1. valid physical reset state;
2. no hard safety/numerical failure;
3. verified alternate support with required palm participation;
4. at least one designated acquisition finger unloaded and contact-free;
5. released finger exceeds the approved usable-motion/workspace criterion;
6. object passes post-release retention duration, translation, rotation, contact-gap, and table/workspace gates.

Report each clause separately so a failed conjunction remains interpretable.

### 10. Proposed reset distribution

Create it before RL with the Phase 3A Shadow controller:

1. deterministically sample object pose/orientation around the compiled
   thumb-index workspace, with seed and proposal recorded;
2. run independent contact-aware thumb/index closing against the temporary
   fixture;
3. log complete release state, semantic contacts/forces, velocities,
   penetration pairs, q/qvel, controls, and fixture status;
4. disable the fixture and run the PI-approved unsupported validation;
5. accept only states passing the frozen acquisition and safety criteria;
6. hash and deduplicate full states;
7. stratify by pose, orientation, contact geometry, force, and penetration;
8. freeze pose-region-disjoint train/validation/test partitions before RL.

Do not use Phase 2 Allegro states, high-penetration states, palmar-secured resets,
or post-release kinematic object placement.

### 11. Proposed evaluation metrics

- conjunction success rate with confidence intervals and every seed shown;
- each success-clause pass rate and first-failure reason;
- post-release survival distribution;
- palm-contact incidence, duration, force, and load fraction;
- semantic support-vector/load trajectories and handoff event times;
- thumb-recovered and index-recovered rates separately;
- released-finger joint/workspace availability;
- object translation, rotation, linear/angular velocity, and table/workspace loss;
- pair-level penetration distribution and unsafe-event count;
- recruitment identity/order and number of active/supporting fingers;
- action rate, stiffness usage, joint-limit margin, and numerical failures;
- performance on train-distribution, held-out, and OOD reset sets;
- sample efficiency and wall-clock cost, without selecting the best seed post hoc.

### 12. Proposed ablations

1. deployable actor versus privileged-object-state actor;
2. asymmetric critic versus actor-observation-only critic;
3. variable versus fixed impedance;
4. current state versus short history;
5. curriculum versus no curriculum, if feasible;
6. adaptive recruitment versus unrestricted finger closure controls;
7. individual approved reward components/gates, without changing the frozen endpoint;
8. official CM3 versus a PI-approved contact-model alternative after the primary baseline;
9. one-finger versus both-acquisition-finger recovery as evaluation endpoints.

### 13. Proposed out-of-distribution tests

- held-out thumb-index object positions and orientations outside training strata;
- held-out contact-force and release-penetration strata that remain physically valid;
- modest prespecified object mass and friction variations;
- observation noise, delay, and contact dropout;
- actuator/gain perturbations within approved safe ranges;
- longer retention windows and small wrist-pose offsets;
- recovery of thumb versus index on identity-specific subsets.

These are evaluation-only until the PI decides whether any become training
randomization. No OOD result should be used to tune the primary run.

### 14. Estimated implementation complexity

Medium-high. The Shadow environment, semantic contacts, variable impedance,
roles, support metrics, and diagnostics already exist. Remaining work is mainly
reset-state serialization/restoration, final criterion/event implementation,
actor/critic observation separation in the trainer, reward plumbing after PI
approval, curriculum control, resumable PPO training, and rigorous evaluation.
The asymmetric actor-critic interface and leakage tests are the main engineering
risk; no estimate should assume the existing generic trainer is sufficient.

### 15. Estimated training complexity

High and uncertain. The task combines contact-rich dynamics, delayed validated
release credit, partial observability, and a sequential curriculum. Multiple
seeds and a privileged-actor feasibility baseline are needed to distinguish
control failure from observability failure. Interaction budget, number of seeds,
checkpoint rule, and hardware allocation are PI decisions; this packet does not
set them.

### 16. Exact PI decisions needed before coding/training

Before implementing the Phase 3B trainer and running RL, freeze:

- A1-A7: acquisition, secure, retention, motion, contact-gap, and palm/support definitions;
- B1-B3 and B5: penetration handling, intended/gross contact classes, primary condim, and hard failures;
- C1-C5: recovered-resource structure and finger identity endpoints;
- D1-D6: actor/critic contents, roles, history, privileged baseline, and normalization;
- E1-E4 and E6: action form, bounds, stiffness grouping, and rate safety;
- F1-F11: which reward components/gates are active, their normalization/signs, and weights;
- G1-G9: reset dataset/splits, curriculum, horizon, termination, PPO implementation, baselines, budget/seeds, checkpoint rule, and evaluation/OOD protocol.

B4 (timing of the contact-model ablation) and E5 (timing of the fixed-impedance
ablation) may safely remain sensitivity-planning choices if the primary CM3 and
26-D settings are frozen. No code or training should treat any recommendation in
this packet as PI authorization.
