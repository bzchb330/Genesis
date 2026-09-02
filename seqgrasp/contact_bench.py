"""Isolated sphere-plane normal-load bench. No hand, controller or task imports.

Gravity remains active. Applied force is +mg - F_target along z, so F_target
is TOTAL compressive load, not extra load on top of weight. This also permits
the user-requested loads below weight and a zero-total-load unload endpoint.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


FIELDS = ('time_s', 'target_load_n', 'applied_force_z_n', 'z_m', 'vz_mps',
          'signed_overlap_m', 'overlap_m', 'delta_over_radius', 'normal_force_n',
          'constraint_force_z_n', 'acceleration_z_mps2', 'kinetic_energy_j',
          'potential_energy_j', 'external_work_j', 'gravity_work_j', 'contact_work_j',
          'work_balance_residual_j', 'solver_iterations', 'solver_gradient',
          'solver_improvement', 'ncon', 'active_contact', 'constraint_velocity_mps')


@dataclass
class Bench:
    model: mujoco.MjModel
    data: mujoco.MjData
    body_id: int
    sphere_id: int
    plane_id: int
    radius: float
    mass: float
    xml: str


def smooth_ramp(fraction):
    x = np.clip(fraction, 0., 1.)
    return float(x*x*(3-2*x))


def create_bench(snapshot, candidate=None, timestep_multiplier=1., solver_diagnostic=False):
    """Copy compiled geom parameters; change only the isolated bench geometry."""
    p = snapshot['sphere']; hand = snapshot['representative_hand_geom']
    root = ET.Element('mujoco', model='phase3CP0_isolated_normal_contact')
    opts = copy.deepcopy(snapshot['options'])
    dt = opts['timestep'] * timestep_multiplier
    option = ET.SubElement(root, 'option', timestep=str(dt), gravity='0 0 -9.81',
                          integrator=snapshot['integrator_name'], solver=snapshot['solver_name'],
                          cone=snapshot['cone_name'], iterations=str(400 if solver_diagnostic else opts['iterations']),
                          tolerance=str(1e-12 if solver_diagnostic else opts['tolerance']),
                          impratio=str(opts['impratio']))
    # No joint damping, springs, armature, servos, weld, equality or hand.
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.15 0.3', dir='0 0 -1')
    def attrs(source):
        result = {k:' '.join(map(str,source[k])) for k in ('friction','solref','solimp')}
        result.update({k:str(source[k]) for k in ('condim','priority','margin','gap','solmix')})
        return result
    ET.SubElement(world, 'geom', name='bench_plane', type='plane', size='.15 .15 .01',
                  rgba='.35 .4 .45 1', **attrs(hand))
    body = ET.SubElement(world, 'body', name='bench_sphere', pos=f"0 0 {p['size'][0]}")
    ET.SubElement(body, 'freejoint', name='sphere_free')
    sphere_attrs = attrs(p)
    if candidate:
        sphere_attrs.update({k:' '.join(map(str,candidate[k])) for k in ('solref','solimp')})
    ET.SubElement(body, 'geom', name='bench_sphere_geom', type='sphere', size=str(p['size'][0]),
                  mass=str(snapshot['sphere_mass_kg']), rgba='.9 .55 .12 1', **sphere_attrs)
    xml = ET.tostring(root, encoding='unicode')
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'bench_sphere')
    # Copy all other scalar options used by the active model, not just XML defaults.
    for key in ('ls_iterations','ls_tolerance','noslip_iterations','noslip_tolerance',
                'ccd_iterations','ccd_tolerance','disableflags','enableflags'):
        if key in opts:
            setattr(m.opt, key, opts[key])
    d.xfrc_applied[bid,2] = snapshot['sphere_mass_kg']*9.81
    mujoco.mj_forward(m,d)
    return Bench(m,d,bid,1,0,float(p['size'][0]),float(m.body_mass[bid]),xml)


def runtime_contact(bench):
    m,d=bench.model,bench.data
    for i in range(d.ncon):
        c=d.contact[i]
        if {int(c.geom1),int(c.geom2)} == {bench.sphere_id,bench.plane_id}:
            f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f)
            return dict(dim=int(c.dim),friction=c.friction.tolist(),solref=c.solref.tolist(),
                        solimp=c.solimp.tolist(),distance_m=float(c.dist),normal_force_n=float(f[0]),
                        included_margin_m=float(c.includemargin),efc_address=int(c.efc_address))
    return None


def contact_force(bench):
    c=runtime_contact(bench)
    return (c['normal_force_n'],c['efc_address']>=0) if c else (0.,False)


def force_schedule(load,dt,ramp_seconds,hold_seconds,cycle=False):
    ramp_steps=int(round(ramp_seconds/dt)); hold_steps=int(round(hold_seconds/dt))
    up=load*np.array([smooth_ramp(i/ramp_steps) for i in range(1,ramp_steps+1)])
    values=np.r_[up,np.full(hold_steps,load)]
    if cycle: values=np.r_[values,load-up,np.zeros(hold_steps)]
    return values,ramp_steps,hold_steps


def run_bench(bench,load,protocol,cycle=False):
    m,d=bench.model,bench.data; dt=m.opt.timestep
    schedule,ramp_steps,hold_steps=force_schedule(load,dt,protocol['ramp_seconds'],protocol['hold_seconds'],cycle)
    rows=np.zeros((len(schedule),len(FIELDS))); ext_work=grav_work=con_work=0.
    initial_ke=.5*bench.mass*np.dot(d.qvel[:3],d.qvel[:3])
    first_contact_parameters=None
    for i,target in enumerate(schedule):
        z0=float(d.qpos[2]); fext=bench.mass*9.81-float(target)
        d.xfrc_applied[bench.body_id,2]=fext
        mujoco.mj_forward(m,d); fn0,_=contact_force(bench)
        mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
        fn,active=contact_force(bench); z=float(d.qpos[2]); dz=z-z0
        if first_contact_parameters is None and active: first_contact_parameters=runtime_contact(bench)
        ext_work+=fext*dz; grav_work+=-bench.mass*9.81*dz; con_work+=.5*(fn0+fn)*dz
        ke=.5*bench.mass*np.dot(d.qvel[:3],d.qvel[:3])+.5*np.dot(m.body_inertia[bench.body_id]*d.qvel[3:],d.qvel[3:])
        nit=int(np.max(d.solver_niter))
        per_island=len(d.solver)//len(d.solver_niter)
        stat=d.solver[max(0,min(nit-1,per_island-1))]
        overlap=bench.radius-z
        rows[i]=[d.time,target,fext,z,d.qvel[2],overlap,max(0.,overlap),overlap/bench.radius,
                 fn,d.qfrc_constraint[2],d.qacc[2],ke,bench.mass*9.81*z,ext_work,grav_work,con_work,
                 ke-initial_ke-ext_work-grav_work-con_work,nit,float(stat.gradient),float(stat.improvement),
                 d.ncon,float(active),d.qvel[2]]
        if not np.isfinite(rows[i]).all(): raise FloatingPointError(f'Nonfinite bench state at step {i}')
    return dict(samples=rows,fields=FIELDS,ramp_steps=ramp_steps,hold_steps=hold_steps,
                runtime_contact_parameters=first_contact_parameters)


def column(run,key): return run['samples'][:,list(run['fields']).index(key)]


def summarize(run,load,protocol,cycle=False):
    n=protocol['tail_steps']; x=run['samples']; fields=list(run['fields'])
    steady=x[-n:] if not cycle else x[run['ramp_steps']+run['hold_steps']-n:run['ramp_steps']+run['hold_steps']]
    means={k:float(np.mean(steady[:,i])) for i,k in enumerate(fields)}
    variances={k:float(np.var(steady[:,i])) for i,k in enumerate(fields)}
    endpoint=run['ramp_steps']+run['hold_steps']
    fn=column(run,'normal_force_n')[:endpoint]; delta=column(run,'signed_overlap_m')[:endpoint]; v=column(run,'vz_mps')[:endpoint]
    band=protocol['settling_relative_band']; posband=max(protocol['settling_position_floor_m'],band*abs(means['signed_overlap_m']))
    good=(abs(fn-load)<=band*load)&(abs(delta-means['signed_overlap_m'])<=posband)&(abs(v)<=protocol['settling_velocity_mps'])
    bad=np.flatnonzero(~good[run['ramp_steps']:]); first=run['ramp_steps']+(int(bad[-1])+1 if len(bad) else 0)
    settle=None if first>=endpoint else float(column(run,'time_s')[first]-column(run,'time_s')[run['ramp_steps']-1])
    contact=column(run,'active_contact')>0; hits=np.flatnonzero(contact)
    losses=int(np.count_nonzero(contact[:-1]&~contact[1:]))
    force=column(run,'normal_force_n'); eps=protocol['numerical_force_epsilon_n']
    return dict(load_n=load,cycle=cycle,steps=len(x),post_ramp_steps=run['hold_steps'],mean=means,variance=variances,
                settling_after_ramp_seconds=settle,contact_loss_transitions=losses,
                first_contact_step=None if not len(hits) else int(hits[0]+1),
                negative_normal_force_samples=int(np.count_nonzero(force < -eps)),
                maximum_force_step_n=float(np.max(abs(np.diff(force)))),
                maximum_hold_force_step_n=float(np.max(abs(np.diff(fn[run['ramp_steps']:])))),
                final_contact_work_j=float(column(run,'contact_work_j')[-1]),
                final_external_work_j=float(column(run,'external_work_j')[-1]),
                final_gravity_work_j=float(column(run,'gravity_work_j')[-1]),
                final_work_balance_residual_j=float(column(run,'work_balance_residual_j')[-1]),
                final_kinetic_energy_j=float(column(run,'kinetic_energy_j')[-1]),
                unload_final_signed_overlap_m=float(column(run,'signed_overlap_m')[-1]) if cycle else None,
                maximum_overlap_m=float(np.max(column(run,'overlap_m'))),
                maximum_normal_force_n=float(np.max(force)),
                runtime_contact_parameters=run['runtime_contact_parameters'])


def save_trace(path,run):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,samples=run['samples'],fields=np.asarray(run['fields']))


def load_trace(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(samples=data['samples'],fields=data['fields'].tolist())


def hertz_force(delta_m,radius_m,*,young_sphere=None,poisson_sphere=None,young_surface=None,poisson_surface=None):
    if any(x is None for x in (young_sphere,poisson_sphere,young_surface,poisson_surface)):
        raise ValueError('MATERIAL ELASTIC CONSTANTS NOT SPECIFIED BY PI')
    if min(young_sphere,young_surface,radius_m)<=0 or min(poisson_sphere,poisson_surface)<=-1 or max(poisson_sphere,poisson_surface)>=.5:
        raise ValueError('Invalid elastic constants')
    effective=1/((1-poisson_sphere**2)/young_sphere+(1-poisson_surface**2)/young_surface)
    return 4/3*effective*np.sqrt(radius_m)*np.maximum(delta_m,0)**1.5
