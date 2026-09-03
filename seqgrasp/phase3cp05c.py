"""Compiled nominal-pose static realizability; no automatic model retuning."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
import yaml

from . import phase3cp05r as prior, dynamic_reset as reset, contact_physics as physics
from .phase3.model import _name_runtime_collision_geoms, _vec
from .phase3c08 import _forearm_transform

ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/phase3CP05C'


@dataclass(frozen=True)
class AuditConfig:
    physics_identifier: str
    nominal_source: str
    settled_source: str
    numerical_diagnostics: dict
    pi_decision: dict
    resource_fractions: dict
    phase: str


def config():
    return AuditConfig(**yaml.safe_load((ROOT/'configs/phase3CP05C_static_audit.yaml').read_text()))


def read(path): return json.loads((ROOT/path).read_text())


def save(name,obj):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    return obj


def name(m,kind,i): return mujoco.mj_id2name(m,kind,int(i))


def moment_matrix(m,d):
    """MuJoCo 3.11 sparse actuator moment storage -> (nu,nv), not transpose."""
    matrix=np.zeros((m.nu,m.nv))
    if d.actuator_moment.ndim==2:
        matrix[:]=d.actuator_moment
    else:
        for i in range(m.nu):
            start=int(d.moment_rowadr[i]); n=int(d.moment_rownnz[i])
            matrix[i,d.moment_colind[start:start+n]]=d.actuator_moment[start:start+n]
    return matrix


def build_nominal():
    """Same hand composition, without adding sphere, fixture, or floor."""
    cfg=prior.old.physics.phase3c07_scene_config()
    modelpath=ROOT/cfg.hand.model_path
    root=ET.parse(modelpath).getroot()
    forearm=root.find(f"worldbody/.//body[@name='{cfg.hand.forearm_body}']")
    forearm.set('pos',_vec(cfg.hand.mount_pos)); forearm.set('quat',_vec(cfg.hand.mount_quat))
    _forearm_transform(with_actuator=True)(root,cfg)
    collision,_=_name_runtime_collision_geoms(root,cfg)
    option=root.find('option')
    if option is None: option=ET.SubElement(root,'option')
    option.set('timestep',str(cfg.raw['timestep'])); option.set('gravity','0 0 -9.81')
    assets={str(p.relative_to(modelpath.parent)).replace('\\','/'):p.read_bytes() for p in (modelpath.parent/'assets').rglob('*') if p.is_file()}
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    d=mujoco.MjData(m); nominal=read(config().nominal_source)
    d.qpos[:]=nominal['qpos'][:m.nq]; d.qvel[:]=0; d.ctrl[:]=nominal['initial_ctrl']
    mujoco.mj_forward(m,d)
    if m.nq!=25 or m.nv!=25 or m.nu!=21: raise ValueError('Unexpected hand structure: audit mapping first')
    if d.ncon: raise ValueError('Nominal hand-only state unexpectedly has contact')
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied): raise ValueError('External compensation forbidden')
    return m,d,collision


def immutable_parameters(m):
    fields=('actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_forcerange',
        'actuator_ctrlrange','actuator_gear','actuator_trnid','actuator_trntype','actuator_gaintype','actuator_biastype',
        'tendon_stiffness','tendon_damping','tendon_lengthspring','wrap_type','wrap_objid','wrap_prm',
        'jnt_range','jnt_stiffness','dof_damping','dof_armature','dof_frictionloss','body_gravcomp')
    return {x:getattr(m,x).tolist() for x in fields}


def affine_model(m,d):
    if not np.all(m.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED): raise ValueError('Audit non-fixed gains before inversion')
    if not np.all(m.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE): raise ValueError('Audit non-affine bias before inversion')
    if np.any(m.actuator_dyntype!=mujoco.mjtDyn.mjDYN_NONE): raise ValueError('Activation dynamics need separate inversion')
    gain=m.actuator_gainprm[:,0].copy()
    bias=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity
    if np.any(gain==0): raise ValueError('Zero actuator gain')
    return gain,bias


def mapping_check(m,d,A):
    state=reset.state_vector(m,d); gain,bias=affine_model(m,d); probe=config().numerical_diagnostics['force_probe']; rows=[]
    for i in range(m.nu):
        copied=mujoco.MjData(m); mujoco.mj_setState(m,copied,state,reset.STATE)
        sign=1 if copied.ctrl[i]+probe/gain[i]<=m.actuator_ctrlrange[i,1] else -1
        copied.ctrl[i]+=sign*probe/gain[i]; mujoco.mj_forward(m,copied)
        actual=copied.actuator_force.copy()
        rows.append(dict(actuator_id=i,requested_probe_force=sign*probe,actual_force=actual.tolist(),
            max_mapping_error=float(np.max(np.abs(A.T@actual-copied.qfrc_actuator))),
            targeted_force_error=float(actual[i]-sign*probe)))
    return dict(relation='qfrc_actuator = actuator_moment.T @ actuator_force',matrix_shape=list(A.shape),probes=rows,
        maximum_error=max(x['max_mapping_error'] for x in rows),maximum_probe_error=max(abs(x['targeted_force_error']) for x in rows),physics_steps=0)


def allocation(A,tau,lo,hi):
    B=A.T; f,_,rank,sv=np.linalg.lstsq(B,tau,rcond=None)
    bounded=lsq_linear(B,tau,bounds=(lo,hi),tol=config().numerical_diagnostics['bounded_solver_tolerance'],lsq_solver='exact')
    def pack(x):
        residual=B@x-tau; norm=float(np.linalg.norm(residual)); scale=float(np.linalg.norm(tau))
        util=np.abs(x)/np.where(x>=0,hi,-lo)
        return dict(forces=x.tolist(),represented_tau=(B@x).tolist(),residual=residual.tolist(),norm=norm,
            relative_residual=norm/scale if scale else norm,utilization=util.tolist(),maximum_force=float(np.max(abs(x))),
            saturated_indices=np.flatnonzero(np.isclose(x,lo,rtol=0,atol=1e-10)|np.isclose(x,hi,rtol=0,atol=1e-10)).tolist())
    u,b=pack(f),pack(bounded.x); c=config().numerical_diagnostics
    tol=c['absolute_roundoff_tolerance']+c['relative_roundoff_tolerance']*float(np.linalg.norm(tau))
    case='CASE_C' if u['norm']>tol else ('CASE_B' if b['norm']>tol else 'CASE_A')
    return dict(shape=list(A.shape),rank=int(rank),singular_values=sv.tolist(),condition_nonzero=float(sv[0]/sv[-1]),
        missing_generalized_directions=int(A.shape[1]-rank),roundoff_only_tolerance=tol,unbounded=u,bounded=b,
        bounded_solver_success=bool(bounded.success),bounded_solver_message=bounded.message,case=case,
        classification_rule='Exact span/limit feasibility to stated roundoff tolerance only; not a publication or task-success score')


def require_case_a(result):
    if result['case']!='CASE_A': raise PermissionError('PI STOP: no equilibrium preload or dynamics for CASE B/C')


def equilibrium_ctrl(m,d,result):
    require_case_a(result); gain,bias=affine_model(m,d)
    ctrl=(np.asarray(result['bounded']['forces'])-bias)/gain
    if np.any(ctrl<m.actuator_ctrlrange[:,0]) or np.any(ctrl>m.actuator_ctrlrange[:,1]):
        raise ValueError('Force-feasible allocation is outside existing control limits')
    return ctrl


def audit():
    cfg=config(); m,d,collision=build_nominal(); A=moment_matrix(m,d); params=immutable_parameters(m)
    full=prior.old.setup_hand(cfg.physics_identifier,.002)
    # Deleting only the disconnected object/environment must not alter hand parameters.
    expected=immutable_parameters(full.model)
    for key in params:
        val=np.asarray(expected[key]); actual=np.asarray(params[key])
        if not np.array_equal(actual,val[:len(actual)]): raise AssertionError('Hand parameter mismatch: '+key)
    nominal=reset.raw_diagnostic(m,d,np.arange(m.nv))
    settled=read(cfg.settled_source)
    assert reset.digest(settled['integration_state'])==settled['state_sha256']
    assert np.array_equal(d.qpos,read(cfg.nominal_source)['qpos'][:m.nq])
    mapping=mapping_check(m,d,A)
    jointrows=[]; actrows=[]; rowprojector=np.linalg.pinv(A)@A
    for j in range(m.njnt):
        dof=int(m.jnt_dofadr[j]); qa=int(m.jnt_qposadr[j]); acts=np.flatnonzero(A[:,dof]); direct=[i for i in acts if m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT]
        tendon=[i for i in acts if m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_TENDON]
        delta=float(settled['qpos_eq'][qa]-d.qpos[qa]); basis=np.eye(m.nv)[:,dof]
        jointrows.append(dict(joint_id=j,dof=dof,joint_name=name(m,mujoco.mjtObj.mjOBJ_JOINT,j),joint_type=mujoco.mjtJoint(m.jnt_type[j]).name,
            q_nominal=float(d.qpos[qa]),q_settled=settled['qpos_eq'][qa],delta_q=delta,absolute_delta_q=abs(delta),
            directly_actuated=bool(direct),actuators=[name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in acts],
            transmission_types=[mujoco.mjtTrn(m.actuator_trntype[i]).name for i in acts],tendon_coupled=bool(tendon),
            unactuated=not bool(len(acts)),independently_commandable=bool(np.linalg.norm(rowprojector@basis-basis)<1e-10),
            moment_terms={name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i):float(A[i,dof]) for i in acts},
            nominal_joint_limit_margin=float(min(d.qpos[qa]-m.jnt_range[j,0],m.jnt_range[j,1]-d.qpos[qa])) if m.jnt_limited[j] else None))
    for i in range(m.nu):
        trn=int(m.actuator_trntype[i]); obj=mujoco.mjtObj.mjOBJ_JOINT if trn==mujoco.mjtTrn.mjTRN_JOINT else mujoco.mjtObj.mjOBJ_TENDON
        actrows.append(dict(actuator_id=i,name=name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i),transmission=mujoco.mjtTrn(trn).name,
            target=name(m,obj,m.actuator_trnid[i,0]),ctrlrange=m.actuator_ctrlrange[i].tolist(),forcerange=m.actuator_forcerange[i].tolist(),
            gainprm=m.actuator_gainprm[i].tolist(),biasprm=m.actuator_biasprm[i].tolist(),gear=m.actuator_gear[i].tolist(),
            gaintype=mujoco.mjtGain(m.actuator_gaintype[i]).name,biastype=mujoco.mjtBias(m.actuator_biastype[i]).name,
            dyntype=mujoco.mjtDyn(m.actuator_dyntype[i]).name,length=float(d.actuator_length[i]),moment_row=A[i].tolist()))
    inv=mujoco.MjData(m); mujoco.mj_setState(m,inv,reset.state_vector(m,d),reset.STATE)
    inv.qacc[:]=0; mujoco.mj_inverse(m,inv); tau=inv.qfrc_inverse.copy()
    cross=mujoco.MjData(m); mujoco.mj_setState(m,cross,reset.state_vector(m,d),reset.STATE)
    cross.qacc[:]=d.qacc; mujoco.mj_inverse(m,cross)
    consistency=float(np.max(abs(cross.qfrc_inverse-d.qfrc_actuator-d.qfrc_applied)))
    result=allocation(A,tau,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
    for row in jointrows:
        i=row['dof']; row.update(required_tau=float(tau[i]),unbounded_residual=result['unbounded']['residual'][i],bounded_residual=result['bounded']['residual'][i])
    if immutable_parameters(m)!=params: raise AssertionError('Audit changed a protected hand parameter')
    preservation={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in prior.OUTPUT.rglob('*') if p.is_file()}
    protocol=dict(physics_identifier=cfg.physics_identifier,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preserved_p05r_outputs=preservation,pi_decision=cfg.pi_decision,config=yaml.safe_load((ROOT/'configs/phase3CP05C_static_audit.yaml').read_text()),
        scope=dict(receiver=False,b03=False,flyby=False,bounded_force=False,handoff=False,object_B=False,rl=False,shape=False,skin=False),
        mujoco_version=mujoco.__version__,model_sha256=reset.model_signature(m),protected_parameters=params)
    save('protocol.json',protocol)
    summary=dict(physics_identifier=cfg.physics_identifier,base_commit=protocol['base_commit'],nominal=nominal,
        tau_required=tau.tolist(),inverse_passive=inv.qfrc_passive.tolist(),inverse_constraint=inv.qfrc_constraint.tolist(),
        inverse_bias=inv.qfrc_bias.tolist(),qfrc_applied=d.qfrc_applied.tolist(),xfrc_applied=d.xfrc_applied.tolist(),
        static_inverse_identity_error=float(np.max(abs(tau-(inv.qfrc_bias-inv.qfrc_passive-inv.qfrc_constraint)))),
        forward_inverse_consistency_max_error=consistency,forward_to_static_constraint_change=(inv.qfrc_constraint-d.qfrc_constraint).tolist(),
        no_object_no_external_contact=True,hand_nv=m.nv,moment_matrix=A.tolist(),mapping_verification=mapping,
        joints=jointrows,joints_sorted=sorted(jointrows,key=lambda x:-x['absolute_delta_q']),actuators=actrows,allocation=result,
        settled_cache_sha256=settled['state_sha256'],resource_fractions=cfg.resource_fractions,physics_steps=0,
        preload_constructed=False,direct_test=None,local_perturbation=None,
        decision='STOP_AND_REPORT' if result['case']!='CASE_A' else 'CASE_A_PRELOAD_ELIGIBLE',
        pi_override_recorded=True,physics_v1_created=False,foundation_ready=False)
    save('summary.json',summary)
    print('CASE',result['case'],'rank',result['rank'],'shape',A.shape,'residual',result['unbounded']['norm'],'relative',result['unbounded']['relative_residual'])
    print('MAX DRIFT',summary['joints_sorted'][0]); print('qacc',nominal['max_acceleration'],'inverse consistency',consistency)
    return summary
