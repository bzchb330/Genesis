"""Sustained actuator-coordinate preload; one frozen MRL receiver, no search."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import yaml

from . import phase3c12a as a
from .phase3.model import set_object_pose, set_fixture
from .phase3c0 import palm_transform

ROOT=a.ROOT
OUTPUT=ROOT/'outputs/phase3C12B/fixed_support'
SURFACES=('thumb','index','middle','ring','little','palm')
FAILURES=('IMMEDIATE_FORCE_ESCAPE','IMMEDIATE_TORQUE_ROLL','CONTACT_NETWORK_COLLAPSE',
          'SERVO_PRELOAD_DECAY','ACTUATOR_SATURATION','TENDON_COUPLING_FAILURE',
          'GRAVITY_ORIENTATION_MISMATCH','DELAYED_SLIP','GROSS_PENETRATION','STABLE_RECEIVER')


@dataclass(frozen=True)
class Config:
    temporary_support: dict
    primitive: dict
    receiver: dict
    numerical: dict
    video: dict


@lru_cache(maxsize=1)
def config():
    d=yaml.safe_load((ROOT/'configs/phase3C12B_weld_release_receiver.yaml').read_text())
    return Config(**{k:d[k] for k in Config.__dataclass_fields__})


def save(name,value):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    return value


def read(name): return json.loads((OUTPUT/name).read_text())
def name(m,kind,index): return mujoco.mj_id2name(m,kind,int(index))
def build():
    scene=a.build_forearm_scene(with_actuator=True).scene
    # Temporary external support is the only new constraint configuration.
    # Never modify geom/contact fields, native equality relations or actuators.
    scene.model.eq_solref[scene.fixture_eq_id]=config().temporary_support['solref']
    scene.model.eq_solimp[scene.fixture_eq_id]=config().temporary_support['solimp']
    return scene


def contract():
    return dict(object_B=False,rl=False,handoff=False,skin=False,shape_retest=False,
        geometry_search=False,orientation_retuning=False,contact_physics_changes=False,
        resource_workspace_recomputed=False,actuator_gain_changes=False)


def transmission_matrix(scene):
    """Analytic fixed joint/tendon transmission; rows map qvel -> actuator velocity."""
    m=scene.model; matrix=np.zeros((m.nu,m.nv))
    for i in range(m.nu):
        target=int(m.actuator_trnid[i,0]); gear=float(m.actuator_gear[i,0])
        if m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT:
            matrix[i,m.jnt_dofadr[target]]=gear
        elif m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_TENDON:
            for k in range(m.tendon_adr[target],m.tendon_adr[target]+m.tendon_num[target]):
                if m.wrap_type[k]!=mujoco.mjtWrap.mjWRAP_JOINT: raise ValueError('Expected fixed joint tendon')
                matrix[i,m.jnt_dofadr[m.wrap_objid[k]]]=gear*m.wrap_prm[k]
        else: raise ValueError('Unsupported transmission; audit before sending controls')
    return matrix


def physics_fingerprint(scene):
    m=scene.model; digest=hashlib.sha256()
    for field in ('geom_type','geom_size','geom_friction','geom_condim','geom_solref','geom_solimp',
                  'geom_margin','geom_gap','jnt_range','body_mass','body_inertia','actuator_gainprm',
                  'actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_forcerange','dof_damping'):
        digest.update(getattr(m,field).tobytes())
    digest.update(np.array([m.opt.timestep,m.opt.solver,m.opt.cone]).tobytes()); digest.update(m.opt.gravity.tobytes())
    return digest.hexdigest()


def actuation_audit():
    scene=build(); m=scene.model; matrix=transmission_matrix(scene); rows=[]; tendons=[]
    for i in range(m.ntendon):
        terms=[]
        for j in range(m.tendon_adr[i],m.tendon_adr[i]+m.tendon_num[i]):
            terms.append(dict(joint=name(m,mujoco.mjtObj.mjOBJ_JOINT,m.wrap_objid[j]),coefficient=float(m.wrap_prm[j])))
        tendons.append(dict(name=name(m,mujoco.mjtObj.mjOBJ_TENDON,i),terms=terms,stiffness=float(m.tendon_stiffness[i]),
                            damping=float(m.tendon_damping[i]),springlength=m.tendon_lengthspring[i].tolist()))
    for i in range(m.nu):
        target=int(m.actuator_trnid[i,0]); joint=bool(m.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_JOINT)
        kind=mujoco.mjtObj.mjOBJ_JOINT if joint else mujoco.mjtObj.mjOBJ_TENDON
        dofs=np.flatnonzero(matrix[i]); joints=[j for j in range(m.njnt) if m.jnt_dofadr[j] in dofs]
        rows.append(dict(name=name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i),id=i,type='position servo (compiled general actuator)',
            transmission='joint' if joint else 'fixed tendon',transmission_type=int(m.actuator_trntype[i]),
            target=name(m,kind,target),gear=m.actuator_gear[i].tolist(),ctrllimited=bool(m.actuator_ctrllimited[i]),
            ctrlrange=m.actuator_ctrlrange[i].tolist(),forcelimited=bool(m.actuator_forcelimited[i]),
            forcerange=m.actuator_forcerange[i].tolist(),gaintype=int(m.actuator_gaintype[i]),biastype=int(m.actuator_biastype[i]),
            dyntype=int(m.actuator_dyntype[i]),gainprm=m.actuator_gainprm[i].tolist(),biasprm=m.actuator_biasprm[i].tolist(),
            dynprm=m.actuator_dynprm[i].tolist(),effective_kp=float(m.actuator_gainprm[i,0]),
            actuator_velocity_damping=float(-m.actuator_biasprm[i,2]),
            ctrl_semantics='target joint angle (gear=1)' if joint else 'target sum of J2+J1 angles (fixed tendon, unit coefficients; not physical metres)',
            joints=[dict(name=name(m,mujoco.mjtObj.mjOBJ_JOINT,j),limits=m.jnt_range[j].tolist(),
                damping=float(m.dof_damping[m.jnt_dofadr[j]]),stiffness=float(m.jnt_stiffness[j])) for j in joints]))
    return save('actuation_audit.json',dict(actuators=rows,tendons=tendons,transmission_rank=int(np.linalg.matrix_rank(matrix)),
        hand_dofs=25,actuators_count=m.nu,distal_coupling='FF/MF/RF/LF J0 each controls J2+J1; no equality fixing their relative split. Pose-to-target mapping preserves sums, not independent distal positions.',
        physics_fingerprint=physics_fingerprint(scene),world_gravity=m.opt.gravity.tolist(),timestep_s=m.opt.timestep,
        formula='p = clip(kp * (ctrl - actuator_length), forcerange); actuator velocity damping=0; passive joint damping remains.',
        temporary_support=config().temporary_support,
        contract=contract()))


def saturation(scene):
    m,d=scene.model,scene.data; limits=m.actuator_forcerange
    denominator=np.where(d.actuator_force>=0,limits[:,1],-limits[:,0])
    fraction=np.divide(np.abs(d.actuator_force),denominator,out=np.zeros(m.nu),where=denominator>0)
    active=m.actuator_forcelimited.astype(bool)&(fraction>=1-config().numerical['force_limit_tolerance'])
    return fraction,active


def smooth_ramp(step,duration):
    x=np.clip(step/duration,0,1); return float(x*x*(3-2*x))


def contact_wrenches(scene):
    m,d=scene.model,scene.data; oid=a._object_geom_id(scene); com=d.xipos[scene.object_body_id]
    rows=[]; hand_force=np.zeros(3); hand_torque=np.zeros(3); external=np.zeros(3)
    for k in range(d.ncon):
        con=d.contact[k]; pair=[int(con.geom1),int(con.geom2)]
        if oid not in pair: continue
        other=pair[1] if pair[0]==oid else pair[0]; sign=1 if pair[1]==oid else -1
        local=np.zeros(6); mujoco.mj_contactForce(m,d,k,local); frame=con.frame.reshape(3,3)
        force=sign*frame.T@local[:3]; moment=sign*frame.T@local[3:]; torque=np.cross(con.pos-com,force)+moment
        surface=next((s for s in SURFACES if other in a._geom_ids(scene,s)),'environment')
        if surface=='environment': external+=force
        else: hand_force+=force; hand_torque+=torque
        rows.append(dict(geom_ids=pair,geom_names=[name(m,mujoco.mjtObj.mjOBJ_GEOM,x) for x in pair],
            body_names=[name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[x]) for x in pair],surface=surface,
            position_world_m=con.pos.tolist(),inward_normal_world=(sign*con.frame[:3]).tolist(),distance_m=float(con.dist),
            normal_force_n=float(local[0]),tangential_force_n=float(np.linalg.norm(local[1:3])),
            local_wrench_on_geom2=local.tolist(),force_on_sphere_world_n=force.tolist(),
            contact_moment_on_sphere_world_nm=moment.tolist(),torque_about_sphere_com_nm=torque.tolist(),
            friction=con.friction.tolist(),dim=int(con.dim),
            rho_translation=a.friction_utilization(float(local[0]),local[1:3],float(con.friction[0]))))
    return rows,hand_force,hand_torque,external


def weld_wrench(scene):
    m,d=scene.model,scene.data; filtered=np.zeros(d.nefc)
    mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id==scene.fixture_eq_id)
    filtered[mask]=d.efc_force[mask]; generalized=np.zeros(m.nv)
    mujoco.mj_mulJacTVec(m,d,generalized,filtered)
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacBodyCom(m,d,jp,jr,scene.object_body_id)
    start=m.jnt_dofadr[scene.object_joint_id]; jac=np.vstack((jp,jr))[:,start:start+6]
    wrench=np.linalg.solve(jac.T,generalized[start:start+6])
    return wrench[:3],wrench[3:]


def record(scene,step,stage,anchor):
    m,d=scene.model,scene.data; contacts,force,torque,environment=contact_wrenches(scene)
    weld_force,weld_torque=weld_wrench(scene); mass=m.body_mass[scene.object_body_id]; gravity=mass*m.opt.gravity
    velocity=np.zeros(6); mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,scene.object_body_id,velocity,0)
    rotation=d.ximat[scene.object_body_id].reshape(3,3); inertia=rotation@np.diag(m.body_inertia[scene.object_body_id])@rotation.T
    free_torque=torque; alpha=np.linalg.solve(inertia,free_torque-np.cross(velocity[:3],inertia@velocity[:3]))
    fractions,active=saturation(scene); start=m.jnt_qposadr[scene.object_joint_id]
    positive=config().numerical['positive_force_n']; topology=sorted({r['surface'] for r in contacts if r['surface']!='environment' and r['normal_force_n']>positive})
    origin,palm_rotation=palm_transform(scene)
    return dict(step=step,stage=stage,time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),
        actuator_force=d.actuator_force.tolist(),actuator_saturation_fraction=fractions.tolist(),actuator_saturated=active.tolist(),
        actuator_target_error=(d.ctrl-d.actuator_length).tolist(),actuator_length=d.actuator_length.tolist(),actuator_velocity=d.actuator_velocity.tolist(),
        tendon_length=d.ten_length.tolist(),tendon_velocity=d.ten_velocity.tolist(),contacts=contacts,topology=topology,
        sphere_position_world_m=d.xpos[scene.object_body_id].tolist(),sphere_quaternion_wxyz=d.qpos[start+3:start+7].tolist(),
        sphere_center_palm_m=(palm_rotation.T@(d.xpos[scene.object_body_id]-origin)).tolist(),
        sphere_linear_velocity_world_mps=velocity[3:].tolist(),sphere_angular_velocity_world_radps=velocity[:3].tolist(),
        sphere_displacement_from_anchor_m=float(np.linalg.norm(d.xpos[scene.object_body_id]-anchor)),
        hand_force_world_n=force.tolist(),hand_torque_world_nm=torque.tolist(),gravity_force_world_n=gravity.tolist(),
        environment_force_world_n=environment.tolist(),weld_force_world_n=weld_force.tolist(),weld_torque_world_nm=weld_torque.tolist(),
        free_net_force_world_n=(force+gravity).tolist(),free_net_torque_world_nm=free_torque.tolist(),
        counterfactual_linear_acceleration_mps2=((force+gravity)/mass).tolist(),counterfactual_angular_acceleration_radps2=alpha.tolist(),
        welded=bool(d.eq_active[scene.fixture_eq_id]),maximum_penetration_m=max([max(0,-x['distance_m']) for x in contacts]+[0]))


def save_series(relative,rows):
    path=OUTPUT/relative; path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,records_json=np.asarray([json.dumps(r,allow_nan=False) for r in rows]),qpos=np.asarray([r['qpos'] for r in rows]))
    return path.relative_to(ROOT).as_posix()


def load_series(relative):
    with np.load(ROOT/relative,allow_pickle=False) as data: return [json.loads(x) for x in data['records_json']]


def normal_virtual_direction(scene,surface,geom,normal,point):
    """Project normal material motion into available actuator coordinates.

    The minimum-norm transmission inverse is an engineering direction only;
    tendon split/passive dynamics are NOT assumed to obey that inverse.
    """
    ids=scene.actuator_ids[surface] if surface!='palm' else np.array([*scene.actuator_ids['wrist'],scene.model.nu-1])
    jac=np.zeros((3,scene.model.nv)); mujoco.mj_jac(scene.model,scene.data,jac,None,np.asarray(point),int(scene.model.geom_bodyid[geom]))
    transmission=transmission_matrix(scene)[ids]
    sensitivity=np.asarray(normal)@jac@np.linalg.pinv(transmission,rcond=config().numerical['pseudoinverse_rcond'])
    direction=sensitivity/max(np.max(np.abs(sensitivity)),1e-15)
    return ids,direction,sensitivity


def tangent_setup(surface):
    pair=a.read_old('preload_calibration.json')['pairs'][surface]; scene=build()
    tangent,ray=a.old._tangent_setup(scene,pair); a.old._assign_object_world(scene,tangent+ray*config().primitive['normal_probe_m'])
    con=next(r for r in a.contact_rows(scene) if pair['surface_geom_id'] in r['geom_ids'])
    normal=np.asarray(con['inward_normal_world']); point=tangent-a.old.SPHERE_RADIUS_M*normal
    a.old._assign_object_world(scene,tangent); set_fixture(scene,True)
    scene.data.qvel[:]=0; mujoco.mj_forward(scene.model,scene.data); scene.data.ctrl[:]=scene.data.actuator_length
    ids,direction,sensitivity=normal_virtual_direction(scene,surface,pair['surface_geom_id'],normal,point)
    return scene,pair,ids,direction,sensitivity


def primitive_summary(rows,surface,geom,ids):
    tail=rows[-config().primitive['tail_steps']:]; forces=[]; positions=[]; all_pairs=[]
    for r in rows:
        matches=[x for x in r['contacts'] if geom in x['geom_ids']]
        forces.append(sum(x['normal_force_n'] for x in matches))
        positions.extend(x['position_world_m'] for x in matches)
        all_pairs.extend(x['geom_names'] for x in r['contacts'])
    tail_forces=np.asarray(forces[-len(tail):]); switched=any(any(geom not in x['geom_ids'] for x in r['contacts']) for r in rows)
    saturated=any(any(np.asarray(r['actuator_saturated'])[ids]) for r in rows)
    maximum_penetration=max(r['maximum_penetration_m'] for r in rows)
    persistent=bool(np.all(tail_forces>config().numerical['positive_force_n']))
    durations=[]; run=0
    for f in forces:
        run=run+1 if f>config().numerical['positive_force_n'] else 0; durations.append(run)
    return dict(surface=surface,intended_geom_id=int(geom),mean_force_n=float(np.mean(tail_forces)),variance_force_n2=float(np.var(tail_forces)),
        minimum_tail_force_n=float(np.min(tail_forces)),force_to_weight=float(np.mean(tail_forces)/a.object_weight(build())),
        persistent_final_100=persistent,max_contact_run_steps=max(durations),contact_switch=switched,saturated=saturated,
        maximum_saturation_fraction=float(max(np.max(np.asarray(r['actuator_saturation_fraction'])[ids]) for r in rows)),
        mean_target_error=np.mean([np.asarray(r['actuator_target_error'])[ids] for r in tail],axis=0).tolist(),
        mean_actuator_force=np.mean([np.asarray(r['actuator_force'])[ids] for r in tail],axis=0).tolist(),
        maximum_penetration_m=maximum_penetration,maximum_weld_drift_m=max(r['sphere_displacement_from_anchor_m'] for r in rows),
        observed_contact_pairs=sorted({tuple(x) for x in all_pairs}),
        admissible_primitive=bool(persistent and not switched and not saturated and maximum_penetration<=config().numerical['inherited_penetration_reference_m']))


def run_primitives():
    if (OUTPUT/'fixed_sphere_primitives.json').exists(): raise FileExistsError('Primitive outcomes already exist; do not retune sweep.')
    cfg=config().primitive; summaries=[]
    save('frozen_protocol.json',dict(config=yaml.safe_load((ROOT/'configs/phase3C12B_weld_release_receiver.yaml').read_text()),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hash=hashlib.sha256((a.OUTPUT/'mechanics_audits.json').read_bytes()).hexdigest(),
        frozen_before_primitive_outcomes=True,selection='For middle/little: admissible force nearest C12A normal load; ring/palm: smallest admissible positive offset. If no admissible curve, maximum frozen offset is an explicitly diagnostic receiver command, not a validated primitive.'))
    for surface in cfg['surfaces']:
        template,pair,ids,direction,sensitivity=tangent_setup(surface)
        for number,offset in enumerate(cfg['virtual_offsets']):
            scene=build(); scene.data.qpos[:]=template.data.qpos; scene.data.mocap_pos[:]=template.data.mocap_pos; scene.data.mocap_quat[:]=template.data.mocap_quat
            set_fixture(scene,True); mujoco.mj_forward(scene.model,scene.data)
            initial=scene.data.actuator_length.copy(); target=initial.copy(); target[ids]+=float(offset)*direction
            commanded=np.clip(target,scene.model.actuator_ctrlrange[:,0],scene.model.actuator_ctrlrange[:,1]); scene.data.ctrl[:]=np.clip(initial,scene.model.actuator_ctrlrange[:,0],scene.model.actuator_ctrlrange[:,1])
            anchor=scene.data.xpos[scene.object_body_id].copy(); rows=[]
            for step in range(1,cfg['total_steps']+1):
                scene.data.ctrl[:]=np.clip(initial+smooth_ramp(step,cfg['ramp_steps'])*(commanded-initial),scene.model.actuator_ctrlrange[:,0],scene.model.actuator_ctrlrange[:,1])
                mujoco.mj_step(scene.model,scene.data); mujoco.mj_forward(scene.model,scene.data); rows.append(record(scene,step,'FIXED_SPHERE_PRELOAD',anchor))
            result=primitive_summary(rows,surface,pair['surface_geom_id'],ids)
            result.update(virtual_offset=float(offset),actuator_ids=ids.tolist(),actuator_direction=direction.tolist(),
                predicted_normal_sensitivity_m_per_actuator_unit=sensitivity.tolist(),requested_targets=target.tolist(),
                actual_targets=commanded.tolist(),command_clipping=(target-commanded).tolist(),
                timeseries=save_series(f'primitives/{surface}_{number:02d}.npz',rows),
                physics_unchanged=physics_fingerprint(scene)==read('actuation_audit.json')['physics_fingerprint'])
            summaries.append(result); print(surface,offset,'force',result['mean_force_n'],'persistent',result['persistent_final_100'],'switch',result['contact_switch'],flush=True)
    return save('fixed_sphere_primitives.json',dict(rows=summaries,total_trials=len(summaries),steps_each=cfg['total_steps']))


def source_receiver():
    return next(x for x in json.loads((a.OUTPUT/'mechanics_audits.json').read_text())['candidates'] if x['network']['candidate_id']==config().receiver['source_candidate'])


def receiver_setup():
    source=source_receiver(); scene=build(); scene.data.qpos[:]=source['network']['qpos']
    a._gravity_at(scene,dict(zip(a.ORIENTATION_NAMES,source['best']['configuration_rad'])))
    a._set_object_palm(scene,np.asarray(source['network']['center_palm_m']))
    set_fixture(scene,True); mujoco.mj_forward(scene.model,scene.data); scene.data.ctrl[:]=a.actuator_target_from_qpos(scene,scene.data.qpos)
    return scene,source


def selected_offsets():
    source=source_receiver(); target_forces={r['surface']:r['normal_force_n'] for r in source['best']['actual_friction']['forces']}; result={}
    for surface in ('middle','ring','little'):
        rows=[r for r in read('fixed_sphere_primitives.json')['rows'] if r['surface']==surface]
        eligible=[r for r in rows if r['admissible_primitive'] and r['virtual_offset']>0]
        chosen=min(eligible,key=lambda r:(abs(r['mean_force_n']-target_forces[surface]),r['virtual_offset'])) if eligible and surface in target_forces else min(eligible,key=lambda r:r['virtual_offset']) if eligible else rows[-1]
        result[surface]=dict(offset=chosen['virtual_offset'],calibrated=bool(eligible),primitive=chosen['timeseries'],mean_force_n=chosen['mean_force_n'])
    return result


def receiver_readiness(rows):
    tail=rows[-config().primitive['tail_steps']:]
    persistent=set(('middle','ring','little'))
    for r in tail: persistent&=set(r['topology'])
    return dict(persistent_MRL_contacts=sorted(persistent),multi_contact_formed=len(persistent)>=2,
        fixed_support_position_error_m=max(r['sphere_displacement_from_anchor_m'] for r in rows),
        mean_free_net_force_n=np.mean([r['free_net_force_world_n'] for r in tail],axis=0).tolist(),
        mean_free_net_torque_nm=np.mean([r['free_net_torque_world_nm'] for r in tail],axis=0).tolist(),
        any_saturation=any(any(r['actuator_saturated']) for r in tail),
        maximum_penetration_m=max(r['maximum_penetration_m'] for r in tail),
        environment_support=any(any(c['surface']=='environment' and c['normal_force_n']>config().numerical['positive_force_n'] for c in r['contacts']) for r in tail),
        force_stability='Raw final-100-step force variance reported; no publication stability threshold invented.')


def run_construction():
    if (OUTPUT/'receiver_construction.json').exists(): raise FileExistsError('One deterministic construction already executed.')
    scene,source=receiver_setup(); cfg=config().receiver; choices=selected_offsets(); initial=scene.data.ctrl.copy(); target=initial.copy(); directions={}
    for surface in ('middle','ring','little'):
        stored=next((r for r in source['network']['contacts'] if r['surface']==surface),None)
        if stored:
            geom=next(g for g in stored['geom_ids'] if g!=a._object_geom_id(scene)); origin,rotation=palm_transform(scene)
            normal=rotation@stored['inward_normal_palm']; point=origin+rotation@stored['position_palm_m']
        else:
            oid=a._object_geom_id(scene); geom=min(a._geom_ids(scene,surface),key=lambda g:a.old._pair_distance(scene,oid,g)[0])
            gap,points=a.old._pair_distance(scene,oid,geom); point=points[3:]; normal=scene.data.xpos[scene.object_body_id]-point; normal/=np.linalg.norm(normal)
        ids,direction,sensitivity=normal_virtual_direction(scene,surface,geom,normal,point)
        target[ids]+=choices[surface]['offset']*direction
        directions[surface]=dict(geom_id=int(geom),actuator_ids=ids.tolist(),direction=direction.tolist(),normal_world=np.asarray(normal).tolist())
    target=np.clip(target,scene.model.actuator_ctrlrange[:,0],scene.model.actuator_ctrlrange[:,1])
    protocol=dict(qpos=scene.data.qpos.tolist(),initial_ctrl=initial.tolist(),target_ctrl=target.tolist(),
        center_palm_m=source['network']['center_palm_m'],orientation_rad=source['best']['configuration_rad'],
        ramp_steps=cfg['ramp_steps'],settle_steps=cfg['settle_steps'],release_step=cfg['release_step'],
        choices=choices,directions=directions,source_candidate='ROLE_MRL_05',frozen_before_construction=True)
    save('receiver_protocol.json',protocol); anchor=scene.data.xpos[scene.object_body_id].copy(); rows=[]
    for step in range(1,cfg['release_step']+1):
        scene.data.ctrl[:]=initial+smooth_ramp(step,cfg['ramp_steps'])*(target-initial)
        mujoco.mj_step(scene.model,scene.data); mujoco.mj_forward(scene.model,scene.data); rows.append(record(scene,step,'WELDED_CONSTRUCTION',anchor))
    readiness=receiver_readiness(rows); eligible=readiness['multi_contact_formed'] and not readiness['any_saturation'] and not readiness['environment_support'] and readiness['maximum_penetration_m']<=config().numerical['inherited_penetration_reference_m'] and readiness['fixed_support_position_error_m']<=config().temporary_support['fixed_pose_numerical_tolerance_m']
    result=dict(status='READY_FOR_FROZEN_RELEASE' if eligible else 'RECEIVER_CONSTRUCTION_FAILURE',readiness=readiness,
        release_permitted=eligible,timeseries=save_series('receiver_construction.npz',rows),last=rows[-1],
        physics_unchanged=physics_fingerprint(scene)==read('actuation_audit.json')['physics_fingerprint'])
    save('receiver_construction.json',result)
    # Save full integration state for a continuation, never reconstruction of a
    # hand-picked good-looking instant. mjSTATE_INTEGRATION includes warmstart.
    state=np.empty(mujoco.mj_stateSize(scene.model,mujoco.mjtState.mjSTATE_INTEGRATION)); mujoco.mj_getState(scene.model,scene.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(OUTPUT/'frozen_release_integration_state.npz',state=state,anchor=anchor)
    print(result['status'],readiness,flush=True); return result


def retained(row):
    return not row['welded'] and bool(set(row['topology'])&{'middle','ring','little','palm'}) and not any(c['surface']=='environment' and c['normal_force_n']>config().numerical['positive_force_n'] for c in row['contacts']) and row['maximum_penetration_m']<=config().numerical['inherited_penetration_reference_m']


def run_release():
    if (OUTPUT/'release_results.json').exists(): raise FileExistsError('Release disposition already recorded.')
    construction=read('receiver_construction.json')
    if not construction['release_permitted']:
        return save('release_results.json',dict(executed=False,trial_count=0,status='RECEIVER_CONSTRUCTION_FAILURE',
            checkpoints={str(k):None for k in config().receiver['checkpoints']},reason='No physically admissible persistent multi-contact MRL network; weld was not released.'))
    scene=build()
    with np.load(OUTPUT/'frozen_release_integration_state.npz') as data:
        mujoco.mj_setState(scene.model,scene.data,data['state'],mujoco.mjtState.mjSTATE_INTEGRATION); anchor=data['anchor'].copy()
    mujoco.mj_forward(scene.model,scene.data); before=scene.data.ctrl.copy(); set_fixture(scene,False); mujoco.mj_forward(scene.model,scene.data)
    immediate=record(scene,0,'POST_RELEASE_INITIAL',anchor); rows=[]
    for step in range(1,config().receiver['post_release_steps']+1):
        mujoco.mj_step(scene.model,scene.data); mujoco.mj_forward(scene.model,scene.data); rows.append(record(scene,step,'POST_RELEASE',anchor))
    assert np.array_equal(scene.data.ctrl,before)
    checkpoints={str(k):all(retained(r) for r in rows[:k]) for k in config().receiver['checkpoints']}
    return save('release_results.json',dict(executed=True,trial_count=1,immediate=immediate,checkpoints=checkpoints,
        status='RECEIVER-VALIDATED' if checkpoints['1000'] else 'RELEASE_FAILURE_REQUIRES_CAUSAL_AUDIT',
        timeseries=save_series('weld_release_receiver_trial.npz',rows),last=rows[-1],
        first_support_loss_step=next((r['step'] for r in rows if not retained(r)),None),
        peak_linear_speed_mps=max(np.linalg.norm(r['sphere_linear_velocity_world_mps']) for r in rows),
        peak_angular_speed_radps=max(np.linalg.norm(r['sphere_angular_velocity_world_radps']) for r in rows),
        maximum_displacement_m=max(r['sphere_displacement_from_anchor_m'] for r in rows)))
