# Phase 3C-1.0 results

## Outcome

The support-gated metric rejects the Phase 3C-0.8 ballistic fly-by. Direct validation classified B03 as **B03-C**: 0/12 frozen trials survived 100, 500, or 1000 steps. B03 is not approved as a transport target. Consequently, the frozen protocol forbids the B03-stored workspace experiment and the scripted handoff; neither was run or inferred.

## Metric repair

`SUPPORTED_PROGRESS` evaluates raw progress only while hand normal-force support is positive, excludes table/floor/fixture support by construction, and requires palm-relative sphere speed below a configurable engineering diagnostic gate. The predeclared sensitivity gates were `[0.02, 0.05, 0.1]` m/s; none is a publication threshold. Raw minimum distance remains descriptive only.

The Phase 3C-0.8 best trajectory had raw minimum `4.225637 mm`, speed `1.25647952 m/s`, and hand force `0 N` at that sample, so the fly-by is rejected. Across all three gates, the valid supported minimum was `72.734661 mm` and maximum supported progress was `0.038215 mm`.

`TRANSFER_CLEARANCE` is the minimum sphere clearance from middle/ring/little during early transfer. `RECEIVER_READY` is separate: receiver joints near an actual B03 configuration, clear sphere C-space, and at least two storage-side contact opportunities. The joint tolerance is configurable and labeled an engineering geometry diagnostic.

## Frozen B03 direct validation

Manifest SHA-256: `03310c8802269a9b730689bad667a01fa28ef7db556fe79485891e739050ac37`. Candidate selection (medoid then deterministic farthest-point coverage) and four orientations were frozen before dynamic outcomes.

- `B03_CANDIDATE_00`: center `[-0.010000000000000023, -0.025000000000000022, 0.08999999999999997]` m; configuration `{'middle_fraction': 0.0, 'ring_fraction': 0.0, 'little_fraction': 0.0, 'WRJ2_offset_deg': -10.0, 'forearm_PS_deg': -50.0}`; full qpos is frozen in the manifest.
- `B03_CANDIDATE_01`: center `[0.024999999999999967, -0.065, 0.05999999999999999]` m; configuration `{'middle_fraction': 1.0, 'ring_fraction': 0.0, 'little_fraction': 1.0, 'WRJ2_offset_deg': -10.0, 'forearm_PS_deg': -50.0}`; full qpos is frozen in the manifest.
- `B03_CANDIDATE_02`: center `[-0.03500000000000001, 0.014999999999999958, 0.05499999999999999]` m; configuration `{'middle_fraction': 0.0, 'ring_fraction': 0.0, 'little_fraction': 1.0, 'WRJ2_offset_deg': -10.0, 'forearm_PS_deg': -50.0}`; full qpos is frozen in the manifest.

- `NOMINAL`: forearm/WRJ1/WRJ2 = `None`, `None`, `None` rad; basis `associated valid-candidate configuration`.
- `RECEIVER_BIASED`: forearm/WRJ1/WRJ2 = `0.6108652381980151`, `0.488692`, `-0.523599` rad; basis `B03 centroid/support geometry and existing reachable gravity set`.
- `ESCAPE_BIASED`: forearm/WRJ1/WRJ2 = `-0.9599310885968813`, `-0.698132`, `-0.523599` rad; basis `B03 centroid/support geometry and existing reachable gravity set`.
- `TANGENTIAL`: forearm/WRJ1/WRJ2 = `1.308996938995747`, `0.488692`, `-0.523599` rad; basis `B03 centroid/support geometry and existing reachable gravity set`.

All 12 states had zero initial maximum penetration, so B03-D is rejected. Survival counts were `{'10': 9, '25': 4, '50': 1, '100': 0, '200': 0, '500': 0, '1000': 0}`. Median/maximum displacement were `0.172306021` / `1.11846306 m`. Dynamic penetration median/p95/p99/max were `0.000367181842` / `0.00110706292` / `0.00824411775` / `0.0183734507 m`.

Any-contact trial counts were `{'middle': 2, 'ring': 1, 'little': 4, 'palm': 5}`. The dominant load-bearing topology was `[]` in `11` trials. The measured outcome is static/gravity escape from a geometry-only cage, not gross initial overlap.

## Gated downstream experiments

Workspace: **NOT RUN**. There is no dynamically validated retained B03 state in which to fix/store A, so open-versus-stored workspace values, retained fractions, apertures, and collision fractions are unavailable rather than zero.

Scripted handoff: **NOT RUN**. The frozen M2 mapping remains thumb guide + index unload/migration, and the six receiver-first stages are specified in code, but target validation is an upstream prerequisite. Therefore no nominal initial state, gravity decomposition, receiver-ready time, unload time, force history, path trace, B03 entry, post-handoff hold, or handoff video exists.

## Interpretation and next step

The receiver-first hypothesis is untested, not rejected. Old preshape conclusions remain non-discriminative under failed transport and should be reconsidered only after a viable receiving target exists. Trajectory optimization, RL, compliant skin, and object B remain premature. The exact next phase is PI review of the B03-C failure and a PI decision about revising the storage target or contact-conformity hypothesis; no scientific criterion or physics change is selected here.

## Figures

- `docs/figures/phase3C10/old_vs_support_gated_progress.pdf`
- `docs/figures/phase3C10/phase3C08_flyby_metric_failure.pdf`
- `docs/figures/phase3C10/transfer_clearance_vs_receiver_readiness.pdf`
- `docs/figures/phase3C10/B03_actual_validation_states.pdf`
- `docs/figures/phase3C10/B03_gravity_orientation_hold_map.pdf`
- `docs/figures/phase3C10/B03_hold_survival.pdf`
- `docs/figures/phase3C10/B03_support_topology.pdf`
- `docs/figures/phase3C10/thumb_workspace_open_vs_B03.pdf`
- `docs/figures/phase3C10/index_workspace_open_vs_B03.pdf`
- `docs/figures/phase3C10/thumb_index_joint_acquisition_workspace.pdf`
- `docs/figures/phase3C10/six_stage_contact_handoff_sequence.pdf`
- `docs/figures/phase3C10/gravity_transport_receiver_decomposition.pdf`
- `docs/figures/phase3C10/scripted_sphere_speed_profile.pdf`
- `docs/figures/phase3C10/scripted_support_transfer_profile.pdf`
- `docs/figures/phase3C10/scripted_contact_topology_timeline.pdf`
- `docs/figures/phase3C10/Cspace_reference_vs_actual_sphere_path.pdf`
- `docs/figures/phase3C10/B03_storage_handoff_result.pdf`
- `docs/figures/phase3C10/phase3C10_causal_summary.pdf`
