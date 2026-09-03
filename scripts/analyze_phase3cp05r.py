"""Saved-result audit; no dynamics and no automatic scientific promotion."""
import hashlib
import numpy as np
import mujoco
from seqgrasp import phase3cp05r as p, dynamic_reset as r


def main():
    eq=p.effective_equilibrium(); trials=p.read('comparison.json')['trials']; audit=[]
    for row in eq['rows']:
        n=row['physics_name']; snap=p.read(row['cache']); s=p.old.setup_hand(n,.002)
        with p.no_object_contact(s):
            r.restore_equilibrium(s.model,s.data,snap)
            exact_fk=p.fk(s); nominal=row['nominal_fk']; changes=[]
            for key,x in exact_fk['geoms'].items():
                y=nominal['geoms'][key]; delta=np.asarray(x['rotation_world'])@np.asarray(y['rotation_world']).T
                changes.append(dict(name=x['name'],surface=x['surface'],position_displacement_m=float(np.linalg.norm(np.array(x['position_world'])-y['position_world'])),
                                    orientation_displacement_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))))))
            s.data.ctrl[:]=s.data.actuator_length; mujoco.mj_forward(s.model,s.data)
            recentered=r.raw_diagnostic(s.model,s.data,p.hand_dofs(s))
        trace=p.old.load_trace(snap['history']); sensitivity=[]
        for mult in p.config()['equilibrium']['sensitivity_multipliers']:
            g=r.DiagnosticGates('ENGINEERING_DIAGNOSTIC_ONLY',.001*mult,.5*mult,.5); since=None; first=None
            for x in trace:
                since=(x['time_s'] if since is None else since) if r.passes(x,g) else None
                if since is not None and x['time_s']-since>=.5-1e-10: first=x['time_s']; break
            sensitivity.append(dict(multiplier=mult,confirmed_at_absolute_s=first,right_censored=first is None))
        audit.append(dict(physics_name=n,nominal=row['nominal'],snapshot=snap,fk_displacements=changes,
                          cached_fk=exact_fk,ctrl_recenter_counterfactual=recentered,gate_sensitivity=sensitivity,
                          counterfactual_dynamics_steps=0))
    f=p.read('protocol.json'); preserved=all(hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha for path,sha in f['preserved_p05_outputs'].items())
    selection=dict(physics_names=p.config()['candidates'],selected_candidate=None,selection_gate_passed=False,
        all_hand_trials_complete=all(x['completed'] for x in trials),simultaneous_mrl_validated=all(x['simultaneous_mrl_tail'] for x in trials),
        reset_solved_at_production_dt=eq['all_valid'],cross_timestep_reset_valid=not any(x['stop_reason']=='INVALID_PREHOLD_DYNAMICS' for x in trials),
        reason='Production reset repaired; 1/2-ms runs have genuine sustained R/L exposure but no middle contact. 4-ms runs fail the unchanged prehold speed guard. Neither the complete three-finger exposure nor full timestep grid required for selection is satisfied.',
        production_frozen=False,legacy_rejected=True,historical_dynamic_regressions_executed=False,
        next='PI review of middle-contact coverage and common-state 4-ms prehold compatibility. No command or guard retuning after outcomes; no V1 or historical dynamic regression yet.')
    p.save('selection.json',selection)
    p.save('summary.json',dict(physics_names=p.config()['candidates'],base_commit=f['base_commit'],config=p.config(),
        control_schedule=p.control_schedule(),equilibrium=eq,equilibrium_audit=audit,geometry=p.read('settled_geometry.json'),
        trials=trials,selection=selection,p05_outputs_preserved=preserved,p05_preserved_count=len(f['preserved_p05_outputs']),
        resource_fractions=p.config()['resource_fractions'],historical_regression_plan=dict(b03='12-state direct-placement, only after V1 and separate next-substage authorization',
        c08='4.225637-mm fly-by dynamic trajectory, only after V1',executed=False),
        static_results_preserved=['C-space connectivity','workspace','fixed-network static wrench analysis','orientation reachability','resource fractions']))
    print(selection)


if __name__=='__main__': main()
