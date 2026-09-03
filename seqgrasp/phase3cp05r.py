"""P0.5R: hand-only equilibrium, settled geometry, fixed-sphere comparison."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import subprocess

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares, minimize

from . import phase3cp05 as old, contact_physics as physics, dynamic_reset as reset
from .phase3c0 import palm_transform
from .phase3c07 import _set_object_palm

ROOT=old.ROOT
OUTPUT=ROOT/'outputs/phase3CP05R'
MRL=old.MRL


def config(): return yaml.safe_load((ROOT/'configs/phase3CP05R_dynamic_reset.yaml').read_text())
def read(name): return json.loads((OUTPUT/name).read_text())
def save(name,value):
    path=OUTPUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); return value
def save_trace(name,rows,physics_name):
    path=OUTPUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,records_json=np.asarray([json.dumps(r,allow_nan=False) for r in rows]),physics_name=np.asarray(physics_name))
    return path.relative_to(ROOT).as_posix()
def gates():
    c=config()['equilibrium']; return reset.DiagnosticGates(**{k:c[k] for k in ('label','max_speed_radps','max_acceleration_radps2','confirmation_s')})
def hand_dofs(s): return np.flatnonzero(np.any(old.native.transmission_matrix(s)!=0,axis=0))


def control_schedule():
    c=dict(config()['control'])
    # One common grid, not candidate-specific or variable-timestep integration.
    largest=max(config()['timesteps_s']); c['prehold_s']=float(np.ceil(c['prehold_s']/largest)*largest)
    return c


@contextmanager
def no_object_contact(s):
    """Disable ONLY sphere collisions, including explicit pairs, then restore exactly.

    Dynamic geom masks alone do not disable explicit pairs. A diagnostic
    inactive gap prevents their constraints while the sphere is disabled.
    MuJoCo may still report geometric candidates with efc_address=-1.
    Sphere state and fixed support remain, so all hand state indices are exact.
    """
    m=s.model; oid=old.native.a._object_geom_id(s)
    ct=int(m.geom_contype[oid]); ca=int(m.geom_conaffinity[oid]); gaps=m.pair_gap.copy(); margins=m.pair_margin.copy()
    related=(m.pair_geom1==oid)|(m.pair_geom2==oid)
    try:
        m.geom_contype[oid]=0; m.geom_conaffinity[oid]=0
        m.pair_margin[related]=-1.; m.pair_gap[related]=1.
        mujoco.mj_forward(m,s.data); yield oid
    finally:
        m.geom_contype[oid]=ct; m.geom_conaffinity[oid]=ca; m.pair_gap[:]=gaps; m.pair_margin[:]=margins
        mujoco.mj_forward(m,s.data)


def freeze_protocol():
    if (OUTPUT/'protocol.json').exists():
        f=read('protocol.json')
        if f['config']!=config() or f['registry']!=physics.registry(): raise ValueError('Frozen P0.5R protocol changed')
        return f
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in old.OUTPUT.rglob('*') if p.is_file()}
    return save('protocol.json',dict(physics_names=config()['candidates'],config=config(),registry=physics.registry(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),preserved_p05_outputs=hashes,
        scope=dict(receiver=False,b03=False,weld_release=False,handoff=False,object_B=False,rl=False,shape=False,skin=False,resource_recompute=False,storage_search=False),
        frozen_before_settling_and_candidate_contact_outcomes=True))


def fk(s):
    origin,rot=palm_transform(s)
    return dict(palm_origin_world=origin.tolist(),palm_rotation_world=rot.tolist(),
        geoms={str(g):dict(name=old.native.name(s.model,mujoco.mjtObj.mjOBJ_GEOM,g),surface=x,
                    position_world=s.data.geom_xpos[g].tolist(),rotation_world=s.data.geom_xmat[g].reshape(3,3).tolist())
               for x in MRL for g in old.native.a._geom_ids(s,x)})


def prepare_equilibrium():
    freeze_protocol()
    if (OUTPUT/'equilibrium_summary.json').exists(): return read('equilibrium_summary.json')
    outcomes=[]; first=None
    for name in config()['candidates']:
        s=old.setup_hand(name,.002); ids=hand_dofs(s); nominal_fk=fk(s)
        with no_object_contact(s) as oid:
            nominal=reset.state_vector(s.model,s.data); ctrl=s.data.ctrl.copy()
            zero=reset.raw_diagnostic(s.model,s.data,ids)
            if abs(zero['max_acceleration']-81.61685063369048)>1e-8: raise AssertionError('Old startup acceleration did not reproduce')
            result=reset.settle_hand_to_dynamic_equilibrium(s.model,s.data,nominal,ctrl,name,config()['equilibrium']['max_duration_s'],gates(),hand_dofs=ids,object_geom_ids=[oid])
            snap=result['snapshot']; snap['history']=save_trace(f'equilibrium_states/{name}_settling.npz',result['history'],name)
            settled_fk=fk(s); confirmation=[]
            if snap['converged']:
                reset.restore_equilibrium(s.model,s.data,snap)
                for step in range(round(config()['equilibrium']['cache_confirmation_s']/.002)+1):
                    confirmation.append(reset.raw_diagnostic(s.model,s.data,ids))
                    if step<round(config()['equilibrium']['cache_confirmation_s']/.002):
                        mujoco.mj_step(s.model,s.data); mujoco.mj_forward(s.model,s.data)
            snap['restore_confirmation_trace']=save_trace(f'equilibrium_states/{name}_confirmation.npz',confirmation,name)
            snap['restore_confirmation_passed']=bool(confirmation and all(reset.passes(r,gates()) for r in confirmation))
            snap['restore_confirmation_max_speed']=max((r['max_speed'] for r in confirmation),default=None)
            snap['restore_confirmation_max_acceleration']=max((r['max_acceleration'] for r in confirmation),default=None)
            snap['restore_confirmation_max_qpos_drift']=float(np.max(np.abs(s.data.qpos-np.array(snap['qpos_eq']))))
            sens=[]
            for mult in config()['equilibrium']['sensitivity_multipliers']:
                gg=reset.DiagnosticGates('ENGINEERING_DIAGNOSTIC_ONLY',gates().max_speed_radps*mult,gates().max_acceleration_radps2*mult,gates().confirmation_s)
                since=None; at=None
                for row in result['history']:
                    since=(row['time_s'] if since is None else since) if reset.passes(row,gg) else None
                    if since is not None and row['time_s']-since>=gg.confirmation_s-1e-10: at=row['time_s']; break
                sens.append(dict(multiplier=mult,first_confirmed_time_s=at,right_censored=at is None))
            snap['gate_sensitivity']=sens
            path=f'equilibrium_states/{name}.json'; save(path,snap)
            outcome=dict(physics_name=name,cache=path,cache_key=snap['cache_key'],snapshot_hash=snap['state_sha256'],
                nominal=zero,nominal_fk=nominal_fk,settled_fk=settled_fk,converged=snap['converged'],confirmation_passed=snap['restore_confirmation_passed'],
                elapsed_s=snap['elapsed_s'],max_qpos_displacement=float(np.max(abs(np.array(snap['qpos_eq'])-np.array(zero['qpos'])))),
                exact_equilibrium_shared=True if first is None else bool(np.array_equal(first,snap['integration_state'])))
            if first is None: first=snap['integration_state']
            outcomes.append(outcome)
            print('EQUILIBRIUM',name,snap['converged'],snap['elapsed_s'],snap['final_diagnostic']['max_speed'],snap['final_diagnostic']['max_acceleration'],flush=True)
    return save('equilibrium_summary.json',dict(physics_names=config()['candidates'],rows=outcomes,
        all_valid=all(x['converged'] and x['confirmation_passed'] and x['exact_equilibrium_shared'] for x in outcomes)))


def restore_common(name,dt):
    eq=effective_equilibrium(); s=old.setup_hand(name,dt); snap=read(eq['rows'][0]['cache'])
    with no_object_contact(s):
        reset.restore_equilibrium(s.model,s.data,snap,allow_contact_variant=True,allow_diagnostic_timestep=dt!=.002)
    return s


def prepare_geometry():
    if (OUTPUT/'settled_geometry.json').exists(): return read('settled_geometry.json')
    eq=effective_equilibrium()
    if not eq['all_valid']: return dict(physics_names=config()['candidates'],valid=False,reason='No confirmed common equilibrium; no sphere insertion')
    s=restore_common(config()['candidates'][0],.002); m,d=s.model,s.data
    oid=old.native.a._object_geom_id(s); gs=[old.native.a._geom_ids(s,x)[-1] for x in MRL]
    allg=[g for x in s.collision_geoms for g in old.native.a._geom_ids(s,x)]
    # Nominal center is only a local numerical seed, never the accepted geometry.
    seed=np.array(old.read(old.OUTPUT/'geometry.json')['center_palm_m']); gap=config()['geometry']['gap_m']
    hand_before=d.qpos[:m.jnt_qposadr[s.object_joint_id]].copy(); vel_before=d.qvel[:m.jnt_dofadr[s.object_joint_id]].copy()
    def assign(center): _set_object_palm(s,center)
    def residual(center):
        assign(center); dist=np.array([old.native.a.old._pair_distance(s,oid,g)[0] for g in gs])
        others=np.array([old.native.a.old._pair_distance(s,oid,g)[0] for g in allg])
        return np.r_[(dist-gap)*1000,np.minimum(others-gap,0)*1000]
    preliminary=least_squares(residual,seed,bounds=(seed-.03,seed+.03),max_nfev=500,xtol=1e-12,ftol=1e-12,gtol=1e-12,diff_step=1e-5)
    def objective(mm):
        assign(mm/1000); return sum((1000*(old.native.a.old._pair_distance(s,oid,g)[0]-gap))**2 for g in gs)
    def constraints(mm):
        assign(mm/1000); return np.array([1000*(old.native.a.old._pair_distance(s,oid,g)[0]-gap) for g in allg])
    fit=minimize(objective,preliminary.x*1000,method='SLSQP',bounds=list(zip((seed-.03)*1000,(seed+.03)*1000)),
                 constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=1000,ftol=1e-12,eps=1e-5))
    center=fit.x/1000
    assign(center); closest=[old.native.a.old._pair_distance(s,oid,g) for g in gs]; gaps=[old.native.a.old._pair_distance(s,oid,g)[0] for g in allg]
    valid=bool(min(gaps)>=gap-1e-8 and abs(min(gaps)-gap)<=config()['geometry']['gap_fit_tolerance_m'])
    initial=d.ctrl.copy(); direction=np.zeros(m.nu); details=[]
    for surface,g,(dist,points) in zip(MRL,gs,closest):
        normal=(points[:3]-points[3:])/np.linalg.norm(points[:3]-points[3:]); ids,vec,sensitivity=old.native.normal_virtual_direction(s,surface,g,normal,points[3:]); direction[ids]=vec
        details.append(dict(surface=surface,geom_id=int(g),gap_m=dist,normal=normal.tolist(),points=points.tolist(),ids=ids.tolist(),direction=vec.tolist(),sensitivity=sensitivity.tolist()))
    target=initial+config()['control']['maximum_virtual_offset']*direction
    if np.any(target<m.actuator_ctrlrange[:,0]) or np.any(target>m.actuator_ctrlrange[:,1]): valid=False
    assert np.array_equal(hand_before,d.qpos[:len(hand_before)]) and np.array_equal(vel_before,d.qvel[:len(vel_before)])
    return save('settled_geometry.json',dict(physics_names=config()['candidates'],valid=valid,
        method='Local sphere translation only, using settled FK; no hand pose, target, candidate-dependent or storage search',
        nominal_center_seed=seed.tolist(),sphere_center_palm_m=center.tolist(),gap_m=gap,all_hand_gaps_m=gaps,
        gap_semantics='Minimum separation to every hand geom is 0.4 mm; individual M/R/L gaps need not be equal. Fixed hand FK, sphere translation only.',
        closest=details,initial_ctrl=initial.tolist(),target_ctrl=target.tolist(),direction=direction.tolist(),
        qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),act=d.act.tolist(),mocap_pos=d.mocap_pos.tolist(),mocap_quat=d.mocap_quat.tolist(),
        initial_state_sha256=reset.digest(dict(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),act=d.act.tolist(),tendon=d.ten_length.tolist())),
        command_sha256=reset.digest(dict(initial=initial.tolist(),target=target.tolist(),protocol=control_schedule())),
        initial_diagnostic=reset.raw_diagnostic(m,d,hand_dofs(s)),fit_message=fit.message))


def continue_natural_settling():
    """Bounded original-dynamics continuation after the recorded 12-s failure."""
    freeze_protocol()
    if (OUTPUT/'equilibrium_continuation_summary.json').exists(): return read('equilibrium_continuation_summary.json')
    previous=read('equilibrium_summary.json')
    if previous['all_valid']: return previous
    protocol=dict(physics_names=config()['candidates'],maximum_additional_s=108.,maximum_total_s=120.,
        unchanged_gates=config()['equilibrium'],targets_unchanged=True,damping_unchanged=True,
        rationale='12-s natural scout failed on slow opposite-sign distal-joint drift. Continue original dynamics; no controller/gate/contact-outcome tuning.',
        frozen_before_continuation=True)
    save('natural_continuation_protocol.json',protocol); outcomes=[]; first=None
    for n,prior in zip(config()['candidates'],previous['rows']):
        s=old.setup_hand(n,.002); snap=read(prior['cache'])
        with no_object_contact(s) as oid:
            result=reset.settle_hand_to_dynamic_equilibrium(s.model,s.data,np.array(snap['integration_state']),snap['ctrl_eq'],n,
                protocol['maximum_additional_s'],gates(),hand_dofs=hand_dofs(s),object_geom_ids=[oid],compact_history=True)
            out=result['snapshot']; out['total_natural_duration_s']=12+out['elapsed_s']; out['original_nominal_pose_sha256']=snap['nominal_pose_sha256']
            out['history']=save_trace(f'equilibrium_states/{n}_continuation.npz',result['history'],n)
            confirmation=[]
            if out['converged']:
                reset.restore_equilibrium(s.model,s.data,out)
                for i in range(251):
                    confirmation.append(reset.raw_diagnostic(s.model,s.data,hand_dofs(s)))
                    if i<250: mujoco.mj_step(s.model,s.data); mujoco.mj_forward(s.model,s.data)
            out['restore_confirmation_trace']=save_trace(f'equilibrium_states/{n}_continuation_confirmation.npz',confirmation,n)
            out['restore_confirmation_passed']=bool(confirmation and all(reset.passes(x,gates()) for x in confirmation))
            out['restore_confirmation_max_speed']=max((x['max_speed'] for x in confirmation),default=None)
            out['restore_confirmation_max_acceleration']=max((x['max_acceleration'] for x in confirmation),default=None)
            path=f'equilibrium_states/{n}_confirmed.json'; save(path,out)
            row=dict(prior,cache=path,cache_key=out['cache_key'],snapshot_hash=out['state_sha256'],
                converged=out['converged'],confirmation_passed=out['restore_confirmation_passed'],
                elapsed_s=out['total_natural_duration_s'],settled_fk=fk(s),
                max_qpos_displacement=float(np.max(abs(np.array(out['qpos_eq'])-np.array(prior['nominal']['qpos'])))),
                exact_equilibrium_shared=True if first is None else bool(np.array_equal(first,out['integration_state'])))
            if first is None: first=out['integration_state']
            outcomes.append(row); print('CONTINUATION',n,out['converged'],out['total_natural_duration_s'],out['final_diagnostic']['max_speed'],out['final_diagnostic']['max_acceleration'],flush=True)
    return save('equilibrium_continuation_summary.json',dict(physics_names=config()['candidates'],rows=outcomes,
        all_valid=all(x['converged'] and x['confirmation_passed'] and x['exact_equilibrium_shared'] for x in outcomes)))


def effective_equilibrium():
    for name in ('equilibrium_continuation_summary.json','equilibrium_summary.json'):
        if (OUTPUT/name).exists(): return read(name)
    return prepare_equilibrium()


def setup_trial(name,dt):
    g=read('settled_geometry.json')
    if not g['valid']: raise ValueError('Invalid settled sphere geometry')
    s=restore_common(name,dt)
    # Restore complete hand state, then insert only the sphere; no hand reset.
    _set_object_palm(s,np.asarray(g['sphere_center_palm_m']))
    reset.guard_dynamic_startup(s.model,s.data,hand_dofs(s),gates())
    contacts,*_=old.native.contact_wrenches(s)
    if contacts: raise ValueError('Sphere contact before pre-hold')
    return s


def record_trial(s,step,start,stage):
    row=old.native.record(s,step,stage,np.asarray(s.data.mocap_pos[s.fixture_mocap_id]))
    dyn=reset.raw_diagnostic(s.model,s.data,hand_dofs(s))
    row.update({k:dyn[k] for k in ('qacc','act','ctrl_error','max_speed','max_acceleration','tendon_actuator_force','net_generalized_force','force_balance_residual')})
    row['time_s']=float(s.data.time-start); row['solver']=old.solver_stats(s.model,s.data)
    row['total_normal_force_n']=sum(c['normal_force_n'] for c in row['contacts'] if c['surface']!='environment')
    return row


def exposure_summary(rows,dt,stop):
    c=control_schedule(); g=config()['guards']; duration=c['prehold_s']+c['ramp_s']+c['hold_s']
    hits=[i for i,r in enumerate(rows) if r['total_normal_force_n']>g['numerical_positive_force_n']]
    full=bool(stop is None and abs(rows[-1]['time_s']-duration)<1e-8)
    tail=[r for r in rows if r['time_s']>=duration-c['tail_s']-1e-9]
    continued=bool(hits and len(rows)-1>hits[0])
    sustained=bool(full and tail and all(r['total_normal_force_n']>g['numerical_positive_force_n'] for r in tail))
    allmrl=bool(sustained and all(all(x in r['topology'] for x in MRL) for r in tail))
    valid=bool(full and continued and sustained and allmrl)
    pre=[r for r in rows if r['time_s']<=c['prehold_s']+1e-9]
    pairs=sorted({tuple(c['geom_names']) for r in rows for c in r['contacts']})
    events={'|'.join(pair):old.events([any(tuple(c['geom_names'])==pair and c['normal_force_n']>1e-9 for c in r['contacts']) for r in rows],dt) for pair in pairs}
    result=dict(completed=full,stop_reason=stop,elapsed_s=rows[-1]['time_s'],steps=len(rows)-1,
        contact_onset_s=rows[hits[0]]['time_s'] if hits else None,loaded_integration_steps=(len(rows)-1-hits[0]) if hits else 0,
        sustained_loaded_hold=sustained,simultaneous_mrl_tail=allmrl,valid_for_selection=valid,
        final_topology=rows[-1]['topology'],topology_sequence=[list(x) for x in dict.fromkeys(tuple(r['topology']) for r in rows)],
        prehold=dict(contact_absent=not any(r['contacts'] for r in pre),max_speed=max(r['max_speed'] for r in pre),max_acceleration=max(r['max_acceleration'] for r in pre),
            max_fixture_force_n=max(float(np.linalg.norm(r['weld_force_world_n'])) for r in pre)),
        pair_events=events,makes=sum(e['makes'] for e in events.values()),breaks=sum(e['breaks'] for e in events.values()),
        observed_peak_penetration_m=max(r['maximum_penetration_m'] for r in rows),observed_peak_total_force_n=max(r['total_normal_force_n'] for r in rows),
        observed_peak_single_force_n=max([x['normal_force_n'] for r in rows for x in r['contacts']]+[0]),
        actuator_saturated=any(any(r['actuator_saturated']) for r in rows),max_actuator_utilization=max(max(r['actuator_saturation_fraction']) for r in rows),
        maximum_solver_iterations=max(r['solver']['iterations'] for r in rows),solver_warnings=rows[-1]['solver']['warnings'],
        maximum_fixture_displacement_m=max(r['sphere_displacement_from_anchor_m'] for r in rows),
        statistics_gate='Full-duration sustained-loaded run required; separate simultaneous-MRL selection gate',steady=None)
    if sustained:
        force=np.asarray([r['total_normal_force_n'] for r in tail])
        result['steady']=dict(penetration_mean_m=float(np.mean([r['maximum_penetration_m'] for r in tail])),
            delta_over_R=float(np.mean([r['maximum_penetration_m'] for r in tail])/.0125),
            normal_force_mean_n=float(np.mean(force)),normal_force_variance_n2=float(np.var(force)),
            topology_persistence={x:float(np.mean([x in r['topology'] for r in tail])) for x in old.native.SURFACES},
            contact_count_variance=float(np.var([len(r['contacts']) for r in tail])),
            migration=old.contact_migration(tail),actuator_force_variance=np.var([r['actuator_force'] for r in tail],axis=0).tolist(),
            fixture_force_mean_n=np.mean([r['weld_force_world_n'] for r in tail],axis=0).tolist(),
            fixture_torque_mean_nm=np.mean([r['weld_torque_world_nm'] for r in tail],axis=0).tolist(),
            fixture_force_variance_n2=np.var([r['weld_force_world_n'] for r in tail],axis=0).tolist())
    return result


def run_trial(name,dt):
    path=f'{name}/trial_{dt:g}.json'
    if (OUTPUT/path).exists(): return read(path)
    freeze_protocol(); s=setup_trial(name,dt); d,m=s.data,s.model; c=control_schedule(); g=config()['guards']
    geo=read('settled_geometry.json'); initial=np.array(geo['initial_ctrl']); target=np.array(geo['target_ctrl']); start=float(d.time)
    initial_hash=reset.digest(dict(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),act=d.act.tolist(),tendon=d.ten_length.tolist()))
    if initial_hash!=geo['initial_state_sha256']: raise ValueError('Candidate initialization mismatch')
    rows=[]; stop=None; warnings=[]; steps=round((c['prehold_s']+c['ramp_s']+c['hold_s'])/dt)
    # Common 0.252-s prehold is divisible by every nominal timestep.
    endpoints=[c['prehold_s'],c['prehold_s']+c['ramp_s'],c['prehold_s']+c['ramp_s']+c['hold_s']]
    step=0
    while True:
        elapsed=float(d.time-start); stage='PREHOLD' if elapsed<=endpoints[0]+1e-10 else ('RAMP' if elapsed<=endpoints[1]+1e-10 else 'HOLD')
        row=record_trial(s,step,start,stage); row['physics_name']=name; rows.append(row)
        fn=row['total_normal_force_n']; single=max([x['normal_force_n'] for x in row['contacts']]+[0])
        if fn>g['warning_total_normal_force_n'] or single>g['warning_single_normal_force_n']: warnings.append(dict(time_s=row['time_s'],total_n=fn,single_n=single))
        checks=[(not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(),'NONFINITE'),
            (fn>g['hard_total_normal_force_n'],'HARD_TOTAL_FORCE'),(single>g['hard_single_normal_force_n'],'HARD_SINGLE_FORCE'),
            (row['maximum_penetration_m']>g['hard_penetration_m'],'HARD_PENETRATION'),
            (row['sphere_displacement_from_anchor_m']>g['hard_fixture_translation_m'],'FIXTURE_DRIFT'),
            (row['max_speed']>g['hard_hand_speed_radps'],'HARD_SPEED'),(max(row['actuator_saturation_fraction'])>=g['hard_actuator_utilization'],'ACTUATOR_SATURATION')]
        if elapsed<=c['prehold_s']+1e-10:
            checks.extend([(bool(row['contacts']),'INVALID_PREHOLD_CONTACT'),(not reset.passes(row,gates()),'INVALID_PREHOLD_DYNAMICS')])
        stop=next((reason for hit,reason in checks if hit),None)
        if stop or elapsed>=endpoints[-1]-1e-10: break
        fraction=(elapsed+dt-c['prehold_s'])/c['ramp_s']
        d.ctrl[:]=initial+old.bench.smooth_ramp(fraction)*(target-initial)
        mujoco.mj_step(m,d); mujoco.mj_forward(m,d); step+=1
    m.opt.timestep=dt
    result=exposure_summary(rows,dt,stop)
    result.update(physics_name=name,nominal_dt_s=dt,initial_state_sha256=initial_hash,command_sha256=geo['command_sha256'],
        warnings=warnings,boundary_step_policy='Common 0.252-s prehold + 1-s ramp + 2-s hold; nominal timestep unchanged throughout',
        trace=save_trace(f'{name}/trial_{dt:g}.npz',rows,name))
    save(path,result); print('TRIAL',name,dt,result['completed'],result['valid_for_selection'],stop,result['final_topology'],flush=True)
    return result


def run_comparison():
    if not prepare_geometry()['valid']: raise ValueError('No valid settled geometry; comparison gated')
    if not (OUTPUT/'realized_schedule.json').exists():
        save('realized_schedule.json',dict(physics_names=config()['candidates'],control=control_schedule(),
            reason='Round the approximately 0.25-s prehold upward to a common 4-ms grid before any candidate contact outcome; all six trials use exactly 0.252+1+2 s.',
            command_sha256=read('settled_geometry.json')['command_sha256'],frozen_before_contact_trials=True))
    trials=[]
    for name in config()['candidates']:
        for dt in config()['timesteps_s']:
            result=run_trial(name,dt); trials.append(result)
            if result['stop_reason']=='INVALID_PREHOLD_CONTACT':
                return save('comparison.json',dict(physics_names=config()['candidates'],trials=trials,invalid_initialization=True))
    return save('comparison.json',dict(physics_names=config()['candidates'],trials=trials,
        invalid_initialization=any(x['stop_reason'] and x['stop_reason'].startswith('INVALID_PREHOLD') for x in trials)))
