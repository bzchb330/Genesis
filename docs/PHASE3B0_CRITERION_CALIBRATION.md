# Phase 3B-0 Criterion Calibration

This report is descriptive evidence for PI decisions. Percentiles are not
automatically converted into scientific thresholds. Every recommendation is
explicitly nonbinding and no value has been written into an RL configuration.

## A3

Evidence: `{"1": 1.0, "10": 1.0, "100": 1.0, "1000": 0.646, "200": 0.706, "25": 1.0, "250": 0.684, "300": 0.676, "5": 1.0, "50": 1.0, "500": 0.674, "750": 0.662}`

Physical interpretation: Longer windows increasingly test passive retention rather than momentary release stability.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 250 steps / 0.5 s | 0.684000 | Matches the Phase 3A diagnostic duration. | May admit transient holds. |
| 500 steps / 1.0 s | 0.674000 | Tests a full second with moderate cost. | Still simulator- and task-specific. |
| 1000 steps / 2.0 s | 0.646000 | Strongest observed passive-retention evidence. | Higher evaluation cost and may exceed the intended manipulation window. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: use a shorter training validation and retain 1000 steps as the first long evaluation endpoint, while reporting the full survival curve.

## A4

Evidence: `{"1": {"count": 500, "maximum": 1.0220217295372863e-05, "mean": 7.3384556477777125e-06, "median": 7.3397174484119375e-06, "p90": 8.932135744985197e-06, "p95": 9.369968523232253e-06, "p99": 9.868565133482342e-06}, "10": {"count": 500, "maximum": 0.0006251250201857022, "mean": 0.0004579028626159808, "median": 0.00045700204075409387, "p90": 0.0005489913453835266, "p95": 0.0005656832606613098, "p99": 0.0005826912662592222}, "100": {"count": 500, "maximum": 0.06900151823342827, "mean": 0.04036847701759249, "median": 0.038628592210444734, "p90": 0.04740618111002773, "p95": 0.051739424742404144, "p99": 0.058099925293252724}, "1000": {"count": 323, "maximum": 0.09150976260416209, "mean": 0.03824652127027265, "median": 0.03788785654232996, "p90": 0.04023301751803776, "p95": 0.040823515182846576, "p99": 0.04195440350873113}, "200": {"count": 353, "maximum": 0.09412355798724759, "mean": 0.038971521555591915, "median": 0.03802010614829983, "p90": 0.040539568990843364, "p95": 0.041303660238830045, "p99": 0.0793146592043501}, "25": {"count": 500, "maximum": 0.003974544668270499, "mean": 0.002265653596837154, "median": 0.0021933604458692487, "p90": 0.002988387380098716, "p95": 0.003225961577341891, "p99": 0.003677761435773836}, "300": {"count": 338, "maximum": 0.041680033510563846, "mean": 0.037851206736774085, "median": 0.03784006810416614, "p90": 0.039989951638082305, "p95": 0.04058136757561089, "p99": 0.04126756134925351}, "5": {"count": 500, "maximum": 0.00017738610171447836, "mean": 0.0001348997184016955, "median": 0.00013451208408798195, "p90": 0.00015915746113705123, "p95": 0.0001654052219779338, "p99": 0.0001710439376654136}, "50": {"count": 500, "maximum": 0.021275966127392056, "mean": 0.011571899069584479, "median": 0.011919305806358133, "p90": 0.017076613158578082, "p95": 0.018261411008435407, "p99": 0.020132095812669348}, "500": {"count": 337, "maximum": 0.053986172978742065, "mean": 0.03789341527075964, "median": 0.03781501537083827, "p90": 0.04003906486528392, "p95": 0.04060549153152808, "p99": 0.04130647590465728}, "750": {"count": 331, "maximum": 0.0634491877276829, "mean": 0.03797730758942913, "median": 0.03784705074170388, "p90": 0.04011548870978114, "p95": 0.04069343387454882, "p99": 0.04149331290508601}}`

Physical interpretation: Palm-relative displacement measures grasp migration without conflating wrist motion.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 10 mm at 1000 steps | 0.000000 | Half the smallest object semi-axis; rejects major migration. | May reject useful rolling/sliding. |
| 20 mm at 1000 steps | 0.000000 | Equals the smallest object semi-axis. | May admit substantial pose change. |
| 40 mm at 1000 steps | 0.564000 | Equals the smallest object diameter. | May describe near-loss rather than retention. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: select a palm-relative envelope from visual failure transitions, not a percentile alone; retain world displacement as a secondary metric.

## A5

Evidence: `{"angular_speed_1000": {"count": 323, "maximum": 7.4367129858809236, "mean": 0.10088395357844995, "median": 0.015073371204307935, "p90": 0.026249139721407033, "p95": 0.04041385401999271, "p99": 4.029949304306406}, "rotation": {"1": {"count": 500, "maximum": 0.0007344209095338197, "mean": 0.0006168960225371285, "median": 0.0006181210308446416, "p90": 0.0006960375505067671, "p95": 0.0007105659415700756, "p99": 0.0007265261145424847}, "10": {"count": 500, "maximum": 0.027011885017143456, "mean": 0.02076485770984096, "median": 0.02075079339081466, "p90": 0.02418283011127868, "p95": 0.024730486914182587, "p99": 0.02527116911039773}, "100": {"count": 500, "maximum": 1.766607295770077, "mean": 1.382381678283769, "median": 1.388850371772833, "p90": 1.4262628113969986, "p95": 1.4324170611756464, "p99": 1.4540355517466854}, "1000": {"count": 323, "maximum": 1.5505332795279625, "mean": 1.3768713431681034, "median": 1.3785780417767504, "p90": 1.416543676460959, "p95": 1.4216310409457724, "p99": 1.4307553170620946}, "200": {"count": 353, "maximum": 1.6298278296156368, "mean": 1.397953900549628, "median": 1.3975059729480481, "p90": 1.4373795098868019, "p95": 1.4449384067976545, "p99": 1.557381233033485}, "25": {"count": 500, "maximum": 0.17631789848296522, "mean": 0.09983043526551295, "median": 0.09758115620077693, "p90": 0.13547657044905323, "p95": 0.1450660286349467, "p99": 0.16415519199211995}, "300": {"count": 338, "maximum": 1.4478599137940364, "mean": 1.3928436734496321, "median": 1.3944409391849617, "p90": 1.4329294503623178, "p95": 1.4386114916497463, "p99": 1.4464622913972847}, "5": {"count": 500, "maximum": 0.008491281957077714, "mean": 0.006961364001183336, "median": 0.007007778365350545, "p90": 0.007872616216226267, "p95": 0.008039161882014102, "p99": 0.008285944642463738}, "50": {"count": 500, "maximum": 0.8166329777618562, "mean": 0.4950642622498554, "median": 0.5215215280826562, "p90": 0.694105238359712, "p95": 0.7280418042302282, "p99": 0.7853648727004143}, "500": {"count": 337, "maximum": 1.4434906011181254, "mean": 1.388002623793438, "median": 1.3901479028381443, "p90": 1.428200918327709, "p95": 1.4331718612337212, "p99": 1.4415160662400108}, "750": {"count": 331, "maximum": 1.438033350445103, "mean": 1.3820324166301292, "median": 1.3845619809299905, "p90": 1.4224771447984972, "p95": 1.4271532666521232, "p99": 1.4349898155544845}}}`

Physical interpretation: Rotation can be controlled rolling; sustained angular speed is stronger evidence of uncontrolled tumbling.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 15 degrees | 0.000000 | Strict pose preservation. | Likely rejects useful rolling. |
| 30 degrees | 0.000000 | Moderate reorientation allowance. | Still ignores object symmetry. |
| 60 degrees plus angular-speed gate | 0.000000 | Allows deliberate rolling. | Needs a separately approved speed criterion. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: combine symmetry-aware orientation change with sustained angular speed; do not use rotation alone.

## A6

Evidence: `{"duration_s": {"count": 2877, "maximum": 0.092, "mean": 0.010599930483142162, "median": 0.006, "p90": 0.018000000000000002, "p95": 0.06, "p99": 0.078}, "gap_count": 2877, "reestablished_count": 2697, "reestablished_fraction": 0.9374348279457768, "trajectories_with_gap": 500, "trajectory_fraction": 1.0}`

Physical interpretation: Recovered short gaps may represent migration; unrecovered gaps lead toward loss.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 10 ms / 5 steps | 0.206000 | Very conservative continuity. | May prohibit useful gaiting. |
| 50 ms / 25 steps | 0.650000 | Allows brief solver/contact transitions. | May admit early ballistic motion. |
| 100 ms / 50 steps | 1.000000 | Allows longer migration. | Requires a hand-relative safety envelope. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: distinguish recovered from unrecovered gaps and gate any allowance by palm-relative displacement and speed.

## B1_B2

Evidence: `{"index_object": {"count": 500, "maximum": 0.0008136204489595071, "mean": 0.000637503087978082, "median": 0.0006370925611506516, "p90": 0.0006538496923634312, "p95": 0.0006583911393304234, "p99": 0.0006685643607441157}, "maximum_gross_non_grip": {"count": 500, "maximum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}, "maximum_intended_grip": {"count": 500, "maximum": 0.0008136204489595071, "mean": 0.0006588986998483177, "median": 0.0006544090176939729, "p90": 0.0006911262904664688, "p95": 0.0006960981678276643, "p99": 0.0007027296152860632}, "other_finger_object": {"count": 500, "maximum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}, "other_object": {"count": 500, "maximum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}, "palm_object": {"count": 500, "maximum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}, "table_object": {"count": 500, "maximum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}, "thumb_object": {"count": 500, "maximum": 0.0007105819476680096, "mean": 0.0006542444947269251, "median": 0.000653198462585096, "p90": 0.0006911262904664688, "p95": 0.0006960981678276643, "p99": 0.0007027296152860632}}`

Physical interpretation: All accepted initial contacts separate intended fingertip overlap from non-grip collision.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 1 mm intended-grip diagnostic | 1.000000 | Close to the observed sub-millimetre scale. | A strict cutoff may reject valid compliant contact. |
| historical 3 mm intended reference | 1.000000 | Maintains continuity with prior diagnostics. | Was not calibrated for Shadow as final science. |
| pair-specific intended limits plus zero gross initial contact | 1.000000 | Avoids conflating gripping overlap with invalid collision. | Requires PI-approved pair limits. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: preserve pair-aware raw values and visually calibrate intended-interface limits separately from gross collision.

## B5

Evidence: `{"numeric_invalidity": 0, "table_contact": 177, "workspace_exit": "UNDEFINED_PI_THRESHOLD"}`

Physical interpretation: Table contact and numerical invalidity are unambiguous; workspace, translation, rotation, and gap persistence remain undefined.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| terminate numerical invalidity and table contact only | 0.646000 | Uses unambiguous events. | May spend samples after permanent loss. |
| also terminate unrecovered complete contact loss | INSUFFICIENT DATA | Improves efficiency after confirmed loss. | Requires a persistence definition. |
| also terminate PI-approved workspace/gross-collision violations | INSUFFICIENT DATA | Adds mechanical safety. | Criteria are not yet frozen. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: always terminate numerical invalidity and table collision; add other events only after their persistence and geometry are approved.

## C1

Evidence: `"INSUFFICIENT DATA: Phase 3B-0 contains acquisition states and no finger-release maneuver."`

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 25-step contact-free persistence | INSUFFICIENT DATA | Fast validation. | May count transient release. |
| 100-step contact-free persistence | INSUFFICIENT DATA | More robust evidence. | May delay credit. |
| 250-step retention plus motion probe | INSUFFICIENT DATA | Matches Phase 3A diagnostic scale. | Expensive and still task-specific. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: collect explicit post-release motion probes before choosing persistence.

## C2

Evidence: `{"count": 1500, "maximum": 1.8655876617712983, "mean": 1.4777631702805638, "median": 1.565895336321624, "p90": 1.8655794904447383, "p95": 1.8655810188909132, "p99": 1.865583502196914}`

Physical interpretation: Initial free-digit range is a precursor, not evidence that a released acquisition finger is usable.

| Candidate option | Fraction retained | Advantage | Risk |
|---|---:|---|---|
| 0.25 rad aggregate available motion | 1.000000 | Low bar for nominal mobility. | May count kinematically unhelpful motion. |
| 0.5 rad | 1.000000 | Moderate range requirement. | Still not task-space reachability. |
| 1.0 rad plus local-workspace requirement | 1.000000 | Stronger precursor to usable motion. | Local Jacobian remains first-order only. |

RECOMMENDATION ONLY — NOT FROZEN BY PI: require both joint/actuator margin and a task-relevant local workspace in a future release probe.

## E2

Evidence: `"INSUFFICIENT DATA: passive retention has zero post-release target displacement."`

RECOMMENDATION ONLY — NOT FROZEN BY PI: calibrate actuator displacement using scripted dynamic-transfer trajectories, not passive holds.

## E3

Evidence: `"INSUFFICIENT DATA: all Phase 3B-0 trajectories use nominal stiffness scale 1.0."`

RECOMMENDATION ONLY — NOT FROZEN BY PI: run an approved controlled-slip stiffness characterization before lowering the bound.

## E6

Evidence: `"INSUFFICIENT DATA: post-release actuator commands and stiffness scales are constant."`

RECOMMENDATION ONLY — NOT FROZEN BY PI: derive rate bounds from safe scripted handoff motion and compiled actuator constraints.

