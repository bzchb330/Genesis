"""Analyze saved P0.5 data without taking any physics steps."""
import json
import hashlib
import numpy as np
from seqgrasp import phase3cp05 as p, contact_physics as physics


def main():
    frozen=p.validate_frozen(); primary=p.read(p.OUTPUT/'primary_results.json')
    for h in primary['hand']:
        rows=p.load_trace(h['trace'])
        h.update(p.summarize_hand(rows,h['dt_s'],h['stop_reason']))
        h['maximum_applied_command_offset']=float(np.max(np.abs(np.array([r['ctrl'] for r in rows])-np.array(rows[0]['ctrl']))))
        h['maximum_hand_speed_radps']=float(np.max(np.abs(np.array([r['qvel'][:-6] for r in rows]))))
        h['overlap_geometry_planar_equivalent']=p.overlap_geometry(h['peak_penetration_m'])
        p.save(f"{h['physics_name']}/hand_{h['dt_s']:g}.json",h)
    for x in primary['impact']:
        rows=p.load_trace(x['trace']); x['events']=p.events([r['active'] for r in rows],x['dt_s'])
        p.save(f"{x['physics_name']}/impact_{x['dt_s']:g}_{x['height_m']:g}.json",x)
    p.save('primary_results.json',primary)
    complete=all(h['completed'] for h in primary['hand'])
    simultaneous=all(h['simultaneous_mrl_tail_fraction']==1 for h in primary['hand'])
    selection=dict(physics_names=p.config()['candidates'],selected_candidate=None,selection_gate_passed=False,
        all_hand_trials_complete=complete,simultaneous_mrl_validated=simultaneous,
        status='NO_FREEZE_INCOMPLETE_ACTUAL_HAND_VALIDATION',
        rationale='All six shared-command hand runs terminated on predeclared force guards after 3-4 ms, before simultaneous MRL contact or ramp completion. No steady multi-contact/timestep equivalence evidence exists. This is a diagnostic initialization/control limitation, not proof that either contact model is intrinsically unstable. Do not raise guards or retune after outcomes.',
        optional_tangential_executed=False,optional_tangential_reason='Parts C-D did not pass; Part E remains gated.',
        legacy_rejected=True,legacy_deep_state_valid=False,production_frozen=False,
        next_phase='PI review of P0.5 startup/low-force diagnostic design; no P0.6 implementation or receiver reconstruction authorized.',
        material_properties='No E/nu claim; numerical near-rigid assumption only.')
    p.save('selection.json',selection)
    preserved=all(hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha for path,sha in frozen['preserved_p0_files'].items())
    audits=[]
    base=physics.build_hand(physics.LEGACY)
    for name in p.config()['candidates']:
        s=physics.build_hand(name); physics.assert_locked_model(s,name)
        audits.append(dict(physics_name=name,pairs=s.model.npair,native_fingerprint_unchanged=p.native.physics_fingerprint(s)==p.native.physics_fingerprint(base),
                          transmission_identical=bool(np.array_equal(p.native.transmission_matrix(s),p.native.transmission_matrix(base))),
                          surfaces={k:len(v) for k,v in s.collision_geoms.items()}))
    startup=[]
    for n in p.config()['candidates']:
        # mj_forward only: inspect the exact saved initial state, no dynamics step.
        s=p.setup_hand(n,.002); m,d=s.model,s.data
        startup.append(dict(physics_name=n,initial_contact_count=d.ncon,
            initial_actuator_force=d.actuator_force.tolist(),qfrc_bias=d.qfrc_bias.tolist(),
            qfrc_passive=d.qfrc_passive.tolist(),qfrc_constraint=d.qfrc_constraint.tolist(),
            initial_qacc=d.qacc.tolist(),maximum_initial_hand_acceleration_radps2=float(np.max(abs(d.qacc[:-6]))),
            interpretation='Zero servo error is not static equilibrium under native gravity/passive/constraint dynamics. No gravity-compensating preload or settling was applied.'))
    p.save('startup_audit.json',dict(physics_names=p.config()['candidates'],rows=startup,physics_steps=0))
    summary=dict(physics_names=p.config()['candidates'],base_commit=frozen['base_commit'],
                 protocol=frozen['config'],regression=primary['regression'],impact=primary['impact'],hand=primary['hand'],
                 selection=selection,model_audits=audits,p0_preserved=preserved,p0_file_count=len(frozen['preserved_p0_files']),
                 startup_audit=startup,
                 limitations=['Hand observed-window averages/variances are censored startup statistics, not steady statistics.',
                             'No MRL continuity, migration or chatter inference can be made from 1-2 active snapshots.',
                             'Hand overlap-radius numbers are planar-equivalent descriptors, not actual mesh contact patch radii.',
                             'Contact work uses trapezoidal sampled force/velocity; residual includes integration/quadrature error.',
                             'Recorded solver gradients refer to the first island; maximum iteration count and warnings span islands.',
                             'Startup differences across timesteps are not full-duration robustness comparisons.'])
    p.save('summary.json',summary)
    print(json.dumps(dict(selection=selection,hand=[{k:h[k] for k in ['physics_name','dt_s','duration_s','peak_penetration_m','peak_total_normal_force_n','maximum_applied_command_offset']} for h in summary['hand']]),indent=2))


if __name__=='__main__': main()
