"""Complete-state, fixed-target dynamic equilibration and guarded restoration.

Gates are explicitly supplied engineering diagnostics, never scientific criteria.
No velocity overwrite, target re-centering, gain change or damping aid occurs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json

import mujoco
import numpy as np

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
VERSION = 'DYNAMIC_EQUILIBRIUM_RESET_V1'


@dataclass(frozen=True)
class DiagnosticGates:
    label: str
    max_speed_radps: float
    max_acceleration_radps2: float
    confirmation_s: float

    def __post_init__(self):
        values=[self.max_speed_radps,self.max_acceleration_radps2,self.confirmation_s]
        if self.label!='ENGINEERING_DIAGNOSTIC_ONLY' or not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Positive, explicitly labeled engineering gates required')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def model_signature(m, *, contact_pairs=True, timestep=True):
    fields=('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
            'geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_condim','geom_solref','geom_solimp',
            'geom_contype','geom_conaffinity','geom_margin','geom_gap','jnt_type','jnt_axis','jnt_pos','jnt_range',
            'qpos0','dof_damping','dof_armature','dof_frictionloss','jnt_stiffness',
            'actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange',
            'actuator_gear','actuator_trnid','actuator_trntype','tendon_stiffness','tendon_damping',
            'tendon_lengthspring','wrap_type','wrap_objid','wrap_prm','eq_data','eq_solref','eq_solimp')
    if contact_pairs: fields+=('pair_geom1','pair_geom2','pair_dim','pair_friction','pair_solref','pair_solimp','pair_margin','pair_gap')
    d={k:getattr(m,k).tolist() for k in fields}
    d['options']={k:np.asarray(getattr(m.opt,k)).tolist() for k in ('gravity','solver','integrator','iterations','tolerance','cone','impratio','disableflags','enableflags')}
    if timestep: d['options']['timestep']=float(m.opt.timestep)
    return digest(d)


def actuator_configuration_hash(m):
    return digest({k:getattr(m,k).tolist() for k in ('actuator_gainprm','actuator_biasprm','actuator_dynprm',
                   'actuator_ctrlrange','actuator_forcerange','actuator_gear','actuator_trnid','actuator_trntype')})


def state_vector(m,d):
    out=np.empty(mujoco.mj_stateSize(m,STATE)); mujoco.mj_getState(m,d,out,STATE)
    return out


def raw_diagnostic(m,d,hand_dofs):
    ids=np.asarray(hand_dofs,int); matrix=np.zeros((m.nv,m.nv)); mujoco.mj_fullM(m,d,matrix)
    net=d.qfrc_smooth+d.qfrc_constraint
    # Fixed tendon servos expose force through the transmitting actuator.
    tendon_actuator_force=np.zeros(m.ntendon)
    for a in range(m.nu):
        if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_TENDON:
            tendon_actuator_force[m.actuator_trnid[a,0]]+=d.actuator_force[a]*m.actuator_gear[a,0]
    util=np.abs(d.actuator_force)/np.maximum(np.where(d.actuator_force>=0,m.actuator_forcerange[:,1],-m.actuator_forcerange[:,0]),1e-12)
    return dict(time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),qacc=d.qacc.tolist(),ctrl=d.ctrl.tolist(),act=d.act.tolist(),
        actuator_length=d.actuator_length.tolist(),actuator_velocity=d.actuator_velocity.tolist(),actuator_force=d.actuator_force.tolist(),
        ctrl_error=(d.ctrl-d.actuator_length).tolist(),actuator_utilization=util.tolist(),
        tendon_length=d.ten_length.tolist(),tendon_velocity=d.ten_velocity.tolist(),tendon_actuator_force=tendon_actuator_force.tolist(),
        tendon_force_semantics='Mapped native actuator force in fixed-tendon coordinates; not a standalone tendon tension sensor. Native passive tendon stiffness/damping recorded in model hash.',
        passive_force=d.qfrc_passive.tolist(),bias_force=d.qfrc_bias.tolist(),actuator_generalized_force=d.qfrc_actuator.tolist(),
        constraint_generalized_force=d.qfrc_constraint.tolist(),net_generalized_force=net.tolist(),
        mass_times_acceleration=(matrix@d.qacc).tolist(),force_balance_residual=(matrix@d.qacc-net).tolist(),
        max_speed=float(np.max(abs(d.qvel[ids]))),max_acceleration=float(np.max(abs(d.qacc[ids]))),
        contacts=[dict(geom1=int(d.contact[k].geom1),geom2=int(d.contact[k].geom2),dist=float(d.contact[k].dist),
                       active_constraint=bool(d.contact[k].efc_address>=0)) for k in range(d.ncon)])


def passes(row,gates):
    return bool(np.isfinite([row['max_speed'],row['max_acceleration']]).all() and
                row['max_speed']<=gates.max_speed_radps and row['max_acceleration']<=gates.max_acceleration_radps2)


def guard_dynamic_startup(m,d,hand_dofs,gates):
    row=raw_diagnostic(m,d,hand_dofs)
    if not passes(row,gates): raise ValueError(f"Non-equilibrated dynamic startup: speed={row['max_speed']}, acceleration={row['max_acceleration']}")
    return row


def settle_hand_to_dynamic_equilibrium(model,data,nominal_state,ctrl_state,physics_identifier,
                                       maximum_settling_duration,diagnostic_settings,*,hand_dofs,object_geom_ids=(),compact_history=False):
    """Natural production dynamics; targets stay exactly fixed throughout."""
    m,d=model,data; gates=diagnostic_settings
    if not np.isfinite(maximum_settling_duration) or maximum_settling_duration<=0 or not len(hand_dofs):
        raise ValueError('Positive finite duration and explicit hand DOFs required')
    mujoco.mj_setState(m,d,np.asarray(nominal_state),STATE); d.ctrl[:]=ctrl_state
    ctrl=d.ctrl.copy(); damping=m.dof_damping.copy(); full_hash=model_signature(m)
    mujoco.mj_forward(m,d); history=[]; since=None; start=float(d.time); converged=False
    for step in range(int(round(maximum_settling_duration/m.opt.timestep))+1):
        row=raw_diagnostic(m,d,hand_dofs)
        history.append({k:row[k] for k in ('time_s','qpos','qvel','qacc','max_speed','max_acceleration')} if compact_history else row)
        if any(c['active_constraint'] and (c['geom1'] in object_geom_ids or c['geom2'] in object_geom_ids) for c in row['contacts']):
            raise ValueError('Object contact in hand-only settling')
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all(): raise FloatingPointError('Settling diverged')
        since=(float(d.time) if since is None else since) if passes(row,gates) else None
        if since is not None and d.time-since>=gates.confirmation_s-1e-10: converged=True; break
        if step<int(round(maximum_settling_duration/m.opt.timestep)):
            mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
        if not np.array_equal(d.ctrl,ctrl) or not np.array_equal(m.dof_damping,damping): raise RuntimeError('Controller or damping changed during natural settling')
    state=state_vector(m,d)
    snapshot=dict(version=VERSION,physics_identifier=physics_identifier,converged=converged,elapsed_s=float(d.time-start),
        gates=asdict(gates),confirmation_s=gates.confirmation_s,integration_state=state.tolist(),state_sha256=digest(state.tolist()),
        model_sha256=full_hash,hand_common_sha256=model_signature(m,contact_pairs=False,timestep=False),
        actuator_sha256=actuator_configuration_hash(m),gravity=m.opt.gravity.tolist(),production_timestep_s=float(m.opt.timestep),
        qpos_eq=d.qpos.tolist(),qvel_eq=d.qvel.tolist(),ctrl_eq=d.ctrl.tolist(),act_eq=d.act.tolist(),qacc_eq=d.qacc.tolist(),
        tendon_state=dict(length=d.ten_length.tolist(),velocity=d.ten_velocity.tolist()),final_diagnostic=row,
        original_damping_preserved=bool(np.array_equal(m.dof_damping,damping)),temporary_damping_used=False,velocity_overwrite_used=False,
        nominal_pose_sha256=digest(np.asarray(nominal_state).tolist()))
    snapshot['cache_key']=digest({k:snapshot[k] for k in ('version','nominal_pose_sha256','hand_common_sha256','actuator_sha256','gravity','production_timestep_s')})
    return dict(snapshot=snapshot,history=history)


def restore_equilibrium(m,d,snapshot,*,allow_contact_variant=False,allow_diagnostic_timestep=False):
    if not snapshot['converged']: raise ValueError('Cannot restore an unconverged equilibrium as valid')
    if digest(snapshot['integration_state'])!=snapshot['state_sha256']: raise ValueError('Equilibrium state hash mismatch')
    if model_signature(m,contact_pairs=False,timestep=False)!=snapshot['hand_common_sha256']: raise ValueError('Hand/model/controller/production physics changed')
    if not allow_diagnostic_timestep and m.opt.timestep!=snapshot['production_timestep_s']: raise ValueError('Production timestep mismatch')
    if not allow_contact_variant and not allow_diagnostic_timestep and model_signature(m)!=snapshot['model_sha256']: raise ValueError('Physics version mismatch')
    mujoco.mj_setState(m,d,np.asarray(snapshot['integration_state']),STATE); mujoco.mj_forward(m,d)
    if any(not np.array_equal(getattr(d,key),snapshot[value]) for key,value in
           [('qpos','qpos_eq'),('qvel','qvel_eq'),('ctrl','ctrl_eq'),('act','act_eq')]):
        raise RuntimeError('Incomplete equilibrium restore or inconsistent cached metadata')
    if not np.allclose(d.ten_length,snapshot['tendon_state']['length'],rtol=0,atol=1e-10):
        raise RuntimeError('Restored tendon state does not match the cache')
    return d
