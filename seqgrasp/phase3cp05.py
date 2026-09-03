"""P0.5 diagnostic contact comparison, never a receiver or manipulation test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

from .config import ROOT
from . import contact_physics as physics, contact_bench as bench
from . import phase3c12b as native
from .phase3c07 import _set_object_palm
from .phase3.model import set_fixture

OUTPUT = ROOT / 'outputs/phase3CP05'
MRL = ('middle', 'ring', 'little')


def config():
    return yaml.safe_load((ROOT/'configs/phase3CP05_near_rigid_selection.yaml').read_text())


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    path = OUTPUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def overlap_geometry(delta, radius=.0125):
    a = np.sqrt(np.maximum(0, 2*radius*np.asarray(delta)-np.asarray(delta)**2))
    return dict(a_m=np.asarray(a).tolist(), a_over_R=np.asarray(a/radius).tolist(),
                delta_over_R=np.asarray(delta/radius).tolist())


def prepare_geometry():
    """One local kinematic fit, before dynamics; no task-score or physics tuning."""
    if (OUTPUT/'geometry.json').exists():
        raise FileExistsError('Diagnostic geometry already frozen')
    s = physics.build_hand(physics.LEGACY); m, d = s.model, s.data
    joints = np.concatenate([s.joint_ids[x] for x in MRL]); adr = m.jnt_qposadr[joints]
    seed = np.array([.2,.7,.5,.5, 0,.7,.5,.5, .3,-.2,.7,.5,.5, -.015,-.035,.15])
    geoms = [native.a._geom_ids(s,x)[-1] for x in MRL]
    oid = native.a._object_geom_id(s)
    allg = [g for x in native.SURFACES for g in native.a._geom_ids(s,x)]
    pairs = {(min(g,h),max(g,h)): i for i,(g,h) in enumerate(
        ( (g,h) for k,g in enumerate(allg) for h in allg[k+1:] ))}
    lo = np.r_[m.jnt_range[joints,0]+1e-5,[-.055,-.09,.07]]
    hi = np.r_[m.jnt_range[joints,1]-1e-5,[.025,.01,.20]]
    def assign(v):
        d.qpos[adr] = v[:-3]
        mujoco.mj_forward(m,d); _set_object_palm(s,v[-3:])
    def distances():
        return np.array([native.a.old._pair_distance(s,oid,g)[0] for g in allg])
    clearance = config()['hand']['initial_clearance_m']
    def residual(v):
        assign(v)
        gaps = [native.a.old._pair_distance(s,oid,g)[0] for g in geoms]
        self_overlap = np.zeros(len(pairs))
        for k in range(d.ncon):
            c=d.contact[k]; key=tuple(sorted((int(c.geom1),int(c.geom2))))
            if key in pairs: self_overlap[pairs[key]]=max(self_overlap[pairs[key]],clearance-c.dist,0)
        # Metres scaled to mm; weak posture prior only breaks geometric ties.
        return np.r_[(np.asarray(gaps)-clearance)*1000,
                     np.minimum(distances()-clearance,0)*1000, self_overlap*1000,
                     .0001*(v-seed)/np.r_[np.ones(len(joints)),[.1,.1,.1]]]
    fit = least_squares(residual,seed,bounds=(lo,hi),max_nfev=1000,
                        ftol=1e-12,xtol=1e-12,gtol=1e-12,diff_step=1e-5)
    assign(fit.x); gaps = distances()
    nearest = [native.a.old._pair_distance(s,oid,g) for g in geoms]
    if max(abs(x[0]-clearance) for x in nearest)>config()['hand']['clearance_fit_tolerance_m'] or min(gaps)<0 or any(d.contact[k].dist<0 for k in range(d.ncon)):
        raise RuntimeError(f'Collision-free three-finger geometry fit failed: {fit.message}; gaps={gaps}; fit={fit.x}')
    d.qvel[:] = 0; mujoco.mj_forward(m,d); initial = d.actuator_length.copy()
    direction = np.zeros(m.nu); details=[]
    for surface,g,(gap,points) in zip(MRL,geoms,nearest):
        # mj_geomDistance points: first on sphere, second on finger.
        normal = (points[:3]-points[3:])/np.linalg.norm(points[:3]-points[3:])
        ids,vec,sens = native.normal_virtual_direction(s,surface,g,normal,points[3:])
        direction[ids] = vec
        details.append(dict(surface=surface,geom_id=int(g),gap_m=float(gap),
                            normal_world=normal.tolist(),closest_points=points.tolist(),
                            actuator_ids=ids.tolist(),direction=vec.tolist(),sensitivity=sens.tolist()))
    target = initial+config()['hand']['virtual_actuator_offset']*direction
    if np.any(target<m.actuator_ctrlrange[:,0]) or np.any(target>m.actuator_ctrlrange[:,1]):
        raise RuntimeError('Diagnostic target outside native actuator bounds')
    result = dict(physics_name='ALL_CANDIDATES_SHARED_GEOMETRY',scenario=config()['hand']['scenario'],
                  qpos=d.qpos.tolist(),mocap_pos=d.mocap_pos.tolist(),mocap_quat=d.mocap_quat.tolist(),
                  initial_ctrl=initial.tolist(),target_ctrl=target.tolist(),direction=direction.tolist(),
                  center_palm_m=fit.x[-3:].tolist(),mrl_joint_names=[native.name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints],
                  mrl_joint_qpos=fit.x[:-3].tolist(),distal_contacts=details,
                  all_hand_object_gaps_m=gaps.tolist(),initial_contacts=[dict(geom1=int(d.contact[i].geom1),geom2=int(d.contact[i].geom2),distance=float(d.contact[i].dist)) for i in range(d.ncon)],
                  construction='One bounded local least-squares fit to three distal clearances, not receiver search; no dynamic outcomes used.',
                  seed=seed.tolist(),fit_evaluations=fit.nfev,fit_message=fit.message,
                  native_physics_sha256=native.physics_fingerprint(s))
    result['command_sha256'] = digest([result['initial_ctrl'],result['target_ctrl'],config()['hand']['ramp_s'],config()['hand']['hold_s']])
    return save('geometry.json',result)


def frozen_protocol():
    path=OUTPUT/'frozen_protocol.json'
    if path.exists(): return read(path)
    g=read(OUTPUT/'geometry.json')
    preserved={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/'outputs/phase3CP0').rglob('*') if p.is_file()}
    import subprocess
    return save('frozen_protocol.json',dict(physics_names=config()['candidates'],config=config(),
        registry=physics.registry(),geometry_sha256=digest(g),command_sha256=g['command_sha256'],
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preserved_p0_files=preserved,scope=dict(receiver=False,b03=False,handoff=False,object_B=False,
        rl=False,shape=False,skin=False,weld_release=False),
        modeling_assumption='Numerical near-rigid; not material E/nu calibration',
        setup_debug='Initial static fit rejected for self-collision before any dynamics; retained separately. Final fit explicitly excludes all initial collisions.'))


def validate_frozen():
    f=frozen_protocol()
    if config()!=f['config'] or physics.registry()!=f['registry'] or digest(read(OUTPUT/'geometry.json'))!=f['geometry_sha256']:
        raise ValueError('Frozen diagnostic inputs changed')
    return f


def save_trace(relative,rows,physics_name,**metadata):
    path=OUTPUT/relative; path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,records_json=np.asarray([json.dumps(r,allow_nan=False) for r in rows]),
                        physics_name=np.asarray(physics_name),metadata_json=np.asarray(json.dumps(metadata)))
    return path.relative_to(ROOT).as_posix()


def load_trace(relative):
    with np.load(ROOT/relative,allow_pickle=False) as f: return [json.loads(x) for x in f['records_json']]


def events(active,dt):
    a=np.asarray(active,dtype=bool); edges=np.diff(np.r_[False,a].astype(int))
    times=np.flatnonzero(edges)*dt
    durations=[]; begin=None
    for i,x in enumerate(a):
        if x and begin is None: begin=i
        if not x and begin is not None: durations.append((i-begin)*dt); begin=None
    if begin is not None: durations.append((len(a)-1-begin)*dt)
    intervals=np.diff(times)
    return dict(makes=int(sum(edges==1)),breaks=int(sum(edges==-1)),
                event_times_s=times.tolist(),contact_durations_s=durations,
                last_episode_right_censored=bool(len(a) and a[-1]),
                minimum_event_interval_s=float(min(intervals)) if len(intervals) else None,
                rapid_intervals=int(sum(intervals<config()['impact']['rapid_event_interval_s'])),
                rapid_bin_s=config()['impact']['rapid_event_interval_s'])


def solver_stats(m,d):
    nit=int(max(d.solver_niter)); per=len(d.solver)//len(d.solver_niter)
    st=d.solver[max(0,min(nit-1,per-1))]
    return dict(iterations=nit,gradient=float(st.gradient),improvement=float(st.improvement),
                warnings=[int(w.number) for w in d.warning],nefc=int(d.nefc))


def run_regression():
    validate_frozen(); resultfile=OUTPUT/'regression.json'
    if resultfile.exists(): return read(resultfile)
    snapshot=read(ROOT/'outputs/phase3CP0/current_physics.json')
    protocol=yaml.safe_load((ROOT/'configs/phase3CP0_contact_physics.yaml').read_text())['bench']; rows=[]
    for name in config()['candidates']:
        historical=read(ROOT/'outputs/phase3CP0'/name.replace('PHYSICS_CANDIDATE_','OPTION_')/'results.json')
        for i,load in enumerate(config()['regression_loads_n']):
            b=bench.create_bench(snapshot,physics.version(name)); run=bench.run_bench(b,load,protocol)
            summary=bench.summarize(run,load,protocol)
            # Historical rows contain repeated and sensitivity runs; select production first repeat.
            oldrows=historical['rows']
            old=next(x for x in oldrows if not x['cycle'] and abs(x['load_n']-load)<1e-12 and x.get('timestep_multiplier',1)==1 and not x.get('solver_diagnostic',False))
            difference=summary['mean']['overlap_m']-old['mean']['overlap_m']
            summary.update(physics_name=name,p0_difference_m=difference,
                regression_pass=abs(difference)<=config()['regression_absolute_tolerance_m'],
                overlap_geometry=overlap_geometry(summary['mean']['overlap_m']))
            path=OUTPUT/f'{name}/regression_{i}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,samples=run['samples'],fields=np.asarray(run['fields']),physics_name=np.asarray(name),
                                metadata_json=np.asarray(json.dumps({k:v for k,v in run.items() if k not in ('samples','fields')})))
            summary['trace']=path.relative_to(ROOT).as_posix(); rows.append(summary)
            print('REGRESSION',name,load,summary['mean']['overlap_m'],difference,flush=True)
    result=save('regression.json',dict(physics_names=config()['candidates'],rows=rows,passed=all(x['regression_pass'] for x in rows)))
    if not result['passed']: raise RuntimeError('P0 regression failed; no broader study authorized')
    return result


def run_impact(name,dt,height,*,tangential=False):
    label=f'{name}/'+(f'tangential_{dt:g}' if tangential else f'impact_{dt:g}_{height:g}')
    if (OUTPUT/(label+'.json')).exists(): return read(OUTPUT/(label+'.json'))
    b=bench.create_bench(read(ROOT/'outputs/phase3CP0/current_physics.json'),physics.version(name),dt/.002)
    m,d=b.model,b.data; d.xfrc_applied[:]=0; d.qpos[2]=b.radius+height
    if tangential:
        cfg=config()['optional_tangential']; d.qvel[0]=cfg['initial_vx_mps']; d.qvel[5]=cfg['initial_spin_radps']
    mujoco.mj_forward(m,d); rows=[]; work=0.
    initial_energy=.5*b.mass*np.dot(d.qvel[:3],d.qvel[:3])+.5*np.dot(m.body_inertia[b.body_id]*d.qvel[3:],d.qvel[3:])+b.mass*9.81*d.qpos[2]
    for step in range(int(round(config()['impact']['duration_s']/dt))+1):
        contacts=[]; force=np.zeros(6)
        for k in range(d.ncon):
            c=d.contact[k]; local=np.zeros(6); mujoco.mj_contactForce(m,d,k,local)
            frame=c.frame.reshape(3,3); force[:3]+=frame.T@local[:3]; force[3:]+=frame.T@local[3:]
            contacts.append(dict(fn=float(local[0]),wrench=local.tolist(),dim=int(c.dim),friction=c.friction.tolist(),solref=c.solref.tolist(),solimp=c.solimp.tolist(),distance_m=float(c.dist)))
        ke=.5*b.mass*np.dot(d.qvel[:3],d.qvel[:3])+.5*np.dot(m.body_inertia[b.body_id]*d.qvel[3:],d.qvel[3:]); pe=b.mass*9.81*d.qpos[2]
        power=float(force[:3]@d.qvel[:3]+force[3:]@d.qvel[3:])
        if rows: work+=.5*(rows[-1]['contact_power_w']+power)*dt
        fn=sum(c['fn'] for c in contacts)
        row=dict(physics_name=name,time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),
                 penetration_m=max(0.,float(b.radius-d.qpos[2])),normal_force_n=fn,contact_wrench_world=force.tolist(),
                 kinetic_energy_j=float(ke),potential_energy_j=float(pe),mechanical_energy_j=float(ke+pe),
                 contact_power_w=power,contact_work_j=work,energy_residual_j=float(ke+pe-initial_energy-work),
                 active=bool(fn>1e-9),contacts=contacts,solver=solver_stats(m,d))
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(): raise FloatingPointError(label)
        rows.append(row)
        if step<int(round(config()['impact']['duration_s']/dt)):
            mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
    f=np.array([x['normal_force_n'] for x in rows]); pen=np.array([x['penetration_m'] for x in rows]); hits=np.flatnonzero(f>1e-9)
    first=int(hits[0]) if len(hits) else len(rows)-1; end=min(len(rows),first+int(round(config()['impact']['impulse_window_s']/dt))+1)
    result=dict(physics_name=name,dt_s=dt,height_m=height,tangential=tangential,
        steps=len(rows)-1,duration_s=float(d.time),impact_velocity_mps=float(rows[first]['qvel'][2]),
        peak_penetration_m=float(max(pen)),peak_force_n=float(max(f)),force_impulse_ns=float(np.trapezoid(f,dx=dt)),
        early_impulse_ns=float(np.trapezoid(f[first:end],dx=dt)),early_impulse_duration_s=(end-first-1)*dt,
        rebound_velocity_mps=max(0.,max(x['qvel'][2] for x in rows[first:])),
        force_variance_tail_n2=float(np.var(f[-int(.2/dt):])),maximum_force_step_n=float(max(abs(np.diff(f)))),
        events=events(f>1e-9,dt),contact_work_j=work,energy_residual_j=rows[-1]['energy_residual_j'],
        initial_energy_j=initial_energy,final_energy_j=rows[-1]['mechanical_energy_j'],
        maximum_energy_increase_j=max(x['mechanical_energy_j']-initial_energy for x in rows),
        maximum_solver_iterations=max(x['solver']['iterations'] for x in rows),
        warning_counts=rows[-1]['solver']['warnings'],trace=save_trace(label+'.npz',rows,name,dt_s=dt,height_m=height,tangential=tangential))
    save(label+'.json',result); print('IMPACT',name,dt,height,result['peak_penetration_m'],result['events']['breaks'],flush=True)
    return result


def setup_hand(name,dt):
    s=physics.build_hand(name,diagnostic_timestep=dt); g=read(OUTPUT/'geometry.json'); h=config()['hand']
    s.data.qpos[:]=g['qpos']; s.data.qvel[:]=0; s.data.mocap_pos[:]=g['mocap_pos']; s.data.mocap_quat[:]=g['mocap_quat']
    s.model.eq_solref[s.fixture_eq_id]=h['fixed_support_solref']; s.model.eq_solimp[s.fixture_eq_id]=h['fixed_support_solimp']
    set_fixture(s,True); s.data.ctrl[:]=g['initial_ctrl']; mujoco.mj_forward(s.model,s.data)
    return s


def contact_migration(rows):
    tracks={}
    for r in rows:
        for c in r['contacts']:
            if c['normal_force_n']<=1e-9: continue
            key='|'.join(c['geom_names']); tracks.setdefault(key,[]).append(c)
    result={}
    for key,cs in tracks.items():
        p=np.array([c['position_world_m'] for c in cs]); n=np.array([c['inward_normal_world'] for c in cs])
        result[key]=dict(samples=len(cs),maximum_displacement_from_first_m=float(max(np.linalg.norm(p-p[0],axis=1))),
                        maximum_normal_angle_from_first_deg=float(max(np.degrees(np.arccos(np.clip(n@n[0],-1,1))))))
    return result


def summarize_hand(rows,dt,stop):
    cfg=config()['hand']; tail=[r for r in rows if r['time_s']>=rows[-1]['time_s']-cfg['tail_s']-1e-9]
    fn=np.array([sum(c['normal_force_n'] for c in r['contacts'] if c['surface']!='environment') for r in rows]); tailfn=fn[-len(tail):]
    counts=np.array([sum(c['normal_force_n']>cfg['numerical_positive_force_n'] for c in r['contacts']) for r in rows])
    keys=sorted({tuple(c['geom_names']) for r in rows for c in r['contacts']})
    pair_events={'|'.join(k):events([any(tuple(c['geom_names'])==k and c['normal_force_n']>1e-9 for c in r['contacts']) for r in rows],dt) for k in keys}
    full_duration=cfg['ramp_s']+cfg['hold_s']
    return dict(completed=stop is None and abs(rows[-1]['time_s']-full_duration)<1e-8,stop_reason=stop,
        duration_s=rows[-1]['time_s'],steps=len(rows)-1,peak_penetration_m=max(r['maximum_penetration_m'] for r in rows),
        steady_penetration_m=float(np.mean([r['maximum_penetration_m'] for r in tail])) if stop is None else None,
        steady_total_normal_force_n=float(np.mean(tailfn)) if stop is None else None,
        observed_window_penetration_mean_m=float(np.mean([r['maximum_penetration_m'] for r in tail])),
        observed_window_force_mean_n=float(np.mean(tailfn)),
        window_semantics='steady final 0.4 s' if stop is None else 'censored startup only; not steady state',
        peak_total_normal_force_n=float(max(fn)),
        peak_single_normal_force_n=max([c['normal_force_n'] for r in rows for c in r['contacts']]+[0]),
        total_force_variance_tail_n2=float(np.var(tailfn)),contact_count_variance_all=float(np.var(counts)),
        contact_count_variance_tail=float(np.var(counts[-len(tail):])),
        all_topologies=sorted({tuple(r['topology']) for r in rows}),tail_topologies=sorted({tuple(r['topology']) for r in tail}),
        simultaneous_mrl_tail_fraction=float(np.mean([all(x in r['topology'] for x in MRL) for r in tail])),
        per_surface_tail_force_n={x:float(np.mean([sum(c['normal_force_n'] for c in r['contacts'] if c['surface']==x) for r in tail])) for x in native.SURFACES},
        pair_events=pair_events,makes=sum(x['makes'] for x in pair_events.values()),breaks=sum(x['breaks'] for x in pair_events.values()),
        migration=contact_migration(rows),actuator_force_variance_tail=np.var([r['actuator_force'] for r in tail],axis=0).tolist(),
        any_actuator_saturation=any(any(r['actuator_saturated']) for r in rows),
        maximum_saturation_fraction=max(max(r['actuator_saturation_fraction']) for r in rows),
        fixture_force_mean_n=np.mean([r['weld_force_world_n'] for r in tail],axis=0).tolist(),
        fixture_force_variance_n2=np.var([r['weld_force_world_n'] for r in tail],axis=0).tolist(),
        fixture_torque_mean_nm=np.mean([r['weld_torque_world_nm'] for r in tail],axis=0).tolist(),
        fixture_torque_variance_nm2=np.var([r['weld_torque_world_nm'] for r in tail],axis=0).tolist(),
        maximum_fixture_drift_m=max(r['sphere_displacement_from_anchor_m'] for r in rows),
        maximum_self_penetration_m=max(r['self_penetration_m'] for r in rows),
        maximum_solver_iterations=max(r['solver']['iterations'] for r in rows),
        warning_counts=rows[-1]['solver']['warnings'])


def run_hand(name,dt):
    label=f'{name}/hand_{dt:g}'
    if (OUTPUT/(label+'.json')).exists(): return read(OUTPUT/(label+'.json'))
    s=setup_hand(name,dt); m,d=s.model,s.data; cfg=config()['hand']; g=read(OUTPUT/'geometry.json')
    initial=np.array(g['initial_ctrl']); target=np.array(g['target_ctrl']); anchor=d.xpos[s.object_body_id].copy()
    rows=[]; stop=None; steps=int(round((cfg['ramp_s']+cfg['hold_s'])/dt))
    oid=native.a._object_geom_id(s)
    for step in range(steps+1):
        row=native.record(s,step,'FIXED_SPHERE_DIAGNOSTIC',anchor)
        row.update(physics_name=name,solver=solver_stats(m,d),
            self_penetration_m=max([max(0,-d.contact[k].dist) for k in range(d.ncon) if oid not in [d.contact[k].geom1,d.contact[k].geom2]]+[0]),
            command_clipping=np.zeros(m.nu).tolist())
        rows.append(row); force=sum(x['normal_force_n'] for x in row['contacts']); peak=max([x['normal_force_n'] for x in row['contacts']]+[0])
        checks=[(not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(),'NONFINITE'),
                (force>cfg['max_total_normal_force_n'],'TOTAL_FORCE_GUARD'),(peak>cfg['max_single_normal_force_n'],'SINGLE_FORCE_GUARD'),
                (row['maximum_penetration_m']>cfg['max_penetration_m'],'PENETRATION_GUARD'),
                (row['sphere_displacement_from_anchor_m']>cfg['max_fixture_translation_m'],'FIXTURE_DRIFT_GUARD'),
                (np.max(abs(d.qvel))>cfg['max_joint_speed_radps'],'VELOCITY_GUARD'),
                (any(row['actuator_saturated']),'ACTUATOR_SATURATION')]
        stop=next((reason for hit,reason in checks if hit),None)
        if stop or step==steps: break
        d.ctrl[:]=initial+bench.smooth_ramp((step+1)*dt/cfg['ramp_s'])*(target-initial)
        mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
    summary=summarize_hand(rows,dt,stop)
    summary.update(physics_name=name,dt_s=dt,command_sha256=g['command_sha256'],
        runtime_parameters=[dict(dim=int(d.contact[k].dim),friction=d.contact[k].friction.tolist(),solref=d.contact[k].solref.tolist(),solimp=d.contact[k].solimp.tolist()) for k in range(d.ncon) if oid in [d.contact[k].geom1,d.contact[k].geom2]],
        trace=save_trace(label+'.npz',rows,name,dt_s=dt,command_sha256=g['command_sha256']))
    save(label+'.json',summary); print('HAND',name,dt,stop,summary['tail_topologies'],summary['steady_total_normal_force_n'],flush=True)
    return summary


def run_primary():
    validate_frozen(); regression=run_regression(); impact=[]; hand=[]
    for name in config()['candidates']:
        for dt in config()['timesteps_s']:
            for height in config()['impact']['heights_m']: impact.append(run_impact(name,dt,height))
            hand.append(run_hand(name,dt))
    return save('primary_results.json',dict(physics_names=config()['candidates'],regression=regression,impact=impact,hand=hand))
