"""Phase 3C-1.2A: local contact diagnostics and offline gravity/wrench mechanics.

No receiver holds, altered contact parameters, or trajectory optimizer live here.
Friction scaling operates on offline cone generators, never on MjModel fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog, minimize, nnls
from scipy.stats import qmc
import yaml

from .config import ROOT
from . import phase3c11 as old
from .phase3.config import SUPPORT_SURFACES
from .phase3.contacts import extract_shadow_contacts
from .phase3.control import actuator_target_from_qpos
from .phase3.model import set_fixture
from .phase3c0 import palm_transform, object_pose_in_palm
from .phase3c07 import _geom_ids, _object_geom_id, _set_object_palm
from .phase3c08 import build_forearm_scene, _gravity_at, load_phase3c08_config
from .phase3c09 import _finger_interpolation

OUTPUT = ROOT / 'outputs/phase3C12A'
ORIENTATION_NAMES = ('forearm_PS', 'rh_WRJ1', 'rh_WRJ2')


@dataclass(frozen=True)
class AuditConfig:
    calibration: dict
    mechanics: dict
    orientation: dict
    true_role_t: dict
    workspace: dict


def config() -> AuditConfig:
    d = yaml.safe_load((ROOT/'configs/phase3C12A_contact_gravity_wrench.yaml').read_text())
    return AuditConfig(**{k: d[k] for k in AuditConfig.__dataclass_fields__})


def read_old(name: str) -> dict:
    return json.loads((old.OUTPUT/name).read_text(encoding='utf-8'))


def save(name: str, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    return value


def phase_contract():
    return dict(object_B=False, handoff=False, rl=False, trajectory_optimizer=False,
                contact_implicit_optimizer=False, skin=False, changed_contact_physics=False,
                shape_dynamics=False, receiver_hold_dynamics=False,
                offline_equilibrium_and_orientation_optimization=True)


def object_weight(scene) -> float:
    return float(scene.model.body_mass[scene.object_body_id]*np.linalg.norm(scene.model.opt.gravity))


def contact_rows(scene) -> list[dict]:
    """Measured normals point INTO the object (force on object), not geom order."""
    result = []; oid = _object_geom_id(scene)
    for i in range(scene.data.ncon):
        con = scene.data.contact[i]
        if oid not in (con.geom1, con.geom2): continue
        g1, g2 = int(con.geom1), int(con.geom2); other = g2 if g1 == oid else g1
        normal = np.asarray(con.frame[:3]).copy() * (1 if g2 == oid else -1)
        f = np.zeros(6); mujoco.mj_contactForce(scene.model, scene.data, i, f)
        geom_names = [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (g1,g2)]
        body_ids = [int(scene.model.geom_bodyid[g]) for g in (g1,g2)]
        surface = next((s for s in SUPPORT_SURFACES if other in _geom_ids(scene,s)), 'environment')
        distance, closest = old._pair_distance(scene, oid, other)
        result.append(dict(geom_ids=[g1,g2], geom_names=geom_names, body_ids=body_ids,
            body_names=[mujoco.mj_id2name(scene.model,mujoco.mjtObj.mjOBJ_BODY,b) for b in body_ids],
            surface=surface, position_world_m=con.pos.tolist(), inward_normal_world=normal.tolist(),
            contact_dist_m=float(con.dist), closest_point_distance_m=distance,
            closest_points_world_m=closest.tolist(), normal_force_n=float(f[0]),
            tangential_force_n=float(np.linalg.norm(f[1:3])), runtime_dim=int(con.dim),
            runtime_friction=con.friction.tolist(), efc_address=int(con.efc_address)))
    return result


def motion_components(delta, normal) -> dict:
    delta=np.asarray(delta); normal=np.asarray(normal); length=float(np.linalg.norm(delta))
    projection=float(delta@normal); tangent=delta-projection*normal
    return dict(cartesian_displacement_m=delta.tolist(), normal_displacement_m=projection,
                normal_motion_fraction=None if length<1e-14 else abs(projection)/length,
                tangential_drift_m=float(np.linalg.norm(tangent)))


def pair_changed(records, target_geom_id, object_geom_id) -> bool:
    return any(set(r['geom_ids']) != {target_geom_id,object_geom_id} for r in records)


def old_calibration_autopsy() -> dict:
    stored=read_old('preload_calibration.json'); rows=[]
    for surface,pair in stored['pairs'].items():
        scene=build_forearm_scene(with_actuator=True).scene
        tangent,direction=old._tangent_setup(scene,pair); previous=set(); previous_pos=None; seen_contact=False
        for sample in [r for r in stored['rows'] if r['surface']==surface]:
            mujoco.mj_resetData(scene.model,scene.data); scene.data.qpos[:]=pair['qpos']
            center=tangent+direction*sample['approach_mm']/1000
            old._assign_object_world(scene,center); scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos)
            set_fixture(scene,False); mujoco.mj_forward(scene.model,scene.data)
            contacts=contact_rows(scene); keys={tuple(r['geom_ids']) for r in contacts}
            selected=next((r for r in contacts if pair['surface_geom_id'] in r['geom_ids']),None)
            normal=np.asarray(selected['inward_normal_world']) if selected else -direction
            row=dict(surface=surface,old_approach_mm=sample['approach_mm'],contacts=contacts,
                     commanded_joint_displacement_rad=0.0,hand_material_point_displacement_m=[0,0,0],
                     relative_object_motion=motion_components(center-tangent,normal),
                     pair_changed=bool(previous and keys and keys!=previous),
                     disappeared=bool(previous and not keys),reappeared=bool(seen_contact and not previous and keys),
                     different_geom_active=pair_changed(contacts,pair['surface_geom_id'],_object_geom_id(scene)),
                     stored_force_n=sample['normal_force_n'],reconstructed_force_n=float(extract_shadow_contacts(scene).normal_forces[SUPPORT_SURFACES.index(surface)]))
            position=None if selected is None else np.asarray(selected['position_world_m'])
            row['contact_position_migration_m']=None if position is None or previous_pos is None else float(np.linalg.norm(position-previous_pos))
            row['contact_tangential_migration_m']=None if position is None or previous_pos is None else motion_components(position-previous_pos,normal)['tangential_drift_m']
            rows.append(row); previous=keys; previous_pos=position; seen_contact|=bool(keys)
    result=dict(rows=rows,method='Reconstruct all 66 pre-settling states; no previous calibration dynamics rerun.',
                old_parameterization='Cartesian sphere displacement along geom-centroid radial ray, not finger joint increment; material hand surfaces were stationary.',
                old_quasi_static_issue='Force sampled immediately after fixture removal, while 25 subsequent free-object steps were labeled stability; not a controlled normal-offset quasi-static force curve.',
                switching_count=sum(r['pair_changed'] or r['different_geom_active'] for r in rows),
                maximum_force_reconstruction_error_n=max(abs(r['stored_force_n']-r['reconstructed_force_n']) for r in rows))
    return save('calibration_autopsy.json',result)


def material_point(scene, body, local):
    return scene.data.xpos[body]+scene.data.xmat[body].reshape(3,3)@local


def cartesian_ik(scene, body, local, target, joints):
    addresses=scene.model.jnt_qposadr[joints]; dofs=scene.model.jnt_dofadr[joints]
    initial=scene.data.qpos.copy(); lower=scene.model.jnt_range[joints,0]; upper=scene.model.jnt_range[joints,1]
    def assign(x):
        scene.data.qpos[:]=initial; scene.data.qpos[addresses]=x; mujoco.mj_forward(scene.model,scene.data)
    def residual(x):
        assign(x); return (material_point(scene,body,local)-target)*1000
    def jacobian(x):
        assign(x); jac=np.zeros((3,scene.model.nv)); mujoco.mj_jac(scene.model,scene.data,jac,None,material_point(scene,body,local),body)
        return jac[:,dofs]*1000
    start=np.clip(initial[addresses],lower+1e-12,upper-1e-12)
    solved=least_squares(residual,start,jac=jacobian,bounds=(lower,upper),max_nfev=int(config().calibration['ik_iterations']),gtol=1e-11,xtol=1e-11,ftol=1e-11)
    assign(solved.x)
    return dict(qpos=scene.data.qpos.copy(),joint_delta_rad=(solved.x-initial[addresses]).tolist(),
                cartesian_error_m=float(np.linalg.norm(residual(solved.x))/1000),solver_success=bool(solved.success))


def corrected_calibration() -> dict:
    cfg=config().calibration; stored=read_old('preload_calibration.json'); rows=[]
    for surface,pair in stored['pairs'].items():
        scene=build_forearm_scene(with_actuator=True).scene; tangent,ray=old._tangent_setup(scene,pair)
        scene.data.qpos[:]=pair['qpos']; old._assign_object_world(scene,tangent+ray*cfg['normal_probe_m'])
        probe=next(r for r in contact_rows(scene) if pair['surface_geom_id'] in r['geom_ids'])
        normal=np.asarray(probe['inward_normal_world']); body=int(scene.model.geom_bodyid[pair['surface_geom_id']])
        # The sphere surface point at the tangent is the reference material point.
        reference=tangent-normal*old.SPHERE_RADIUS_M
        local=scene.data.xmat[body].reshape(3,3).T@(reference-scene.data.xpos[body])
        base=np.asarray(pair['qpos']); joints=np.asarray(scene.joint_ids[surface] if surface!='palm' else [mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ORIENTATION_NAMES])
        terminated=False
        for offset in cfg['command_offsets_mm']:
            if terminated: break
            mujoco.mj_resetData(scene.model,scene.data); scene.data.qpos[:]=base; old._assign_object_world(scene,tangent); set_fixture(scene,True)
            before=material_point(scene,body,local).copy(); target=before+normal*float(offset)/1000
            ik=cartesian_ik(scene,body,local,target,joints); commanded=material_point(scene,body,local).copy()
            scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); scene.data.qvel[:]=0; mujoco.mj_forward(scene.model,scene.data)
            start_contacts=contact_rows(scene); timeline=[]; changed=False
            for step in range(int(cfg['settle_steps'])):
                mujoco.mj_step(scene.model,scene.data); mujoco.mj_forward(scene.model,scene.data); contacts=contact_rows(scene)
                switched=pair_changed(contacts,pair['surface_geom_id'],_object_geom_id(scene)); changed |= switched
                selected=[r for r in contacts if pair['surface_geom_id'] in r['geom_ids']]
                timeline.append(dict(step=step+1,contacts=contacts,normal_force_n=sum(r['normal_force_n'] for r in selected),
                                     tangential_force_n=sum(r['tangential_force_n'] for r in selected)))
                if switched: break
            tail=timeline[-int(cfg['tail_steps']):]; actual=material_point(scene,body,local)-before
            row=dict(surface=surface,target_geom=pair['surface_geom_name'],target_geom_id=pair['surface_geom_id'],
                command_offset_mm=float(offset),reference_normal_world=normal.tolist(),
                ik_joint_delta_rad=ik['joint_delta_rad'],ik_error_m=ik['cartesian_error_m'],
                commanded_motion=motion_components(commanded-before,normal),actual_settled_motion=motion_components(actual,normal),
                initial_contacts=start_contacts,settled_contacts=timeline[-1]['contacts'],timeline=timeline,
                mean_normal_force_n=float(np.mean([r['normal_force_n'] for r in tail])),
                std_normal_force_n=float(np.std([r['normal_force_n'] for r in tail])),
                mean_tangential_force_n=float(np.mean([r['tangential_force_n'] for r in tail])),
                contact_pair_switched=changed,branch_terminated=changed,
                object_acceleration_mps2=scene.data.qacc[scene.model.jnt_dofadr[scene.object_joint_id]:scene.model.jnt_dofadr[scene.object_joint_id]+3].tolist(),
                weight_n=object_weight(scene),fixture_enabled_during_settling=True,
                settling_method='Existing actuator position targets and existing object weld; no kinematic clamping or physics override')
            row['force_to_weight_ratio']=row['mean_normal_force_n']/row['weight_n']; rows.append(row); terminated=changed
        print('corrected calibration',surface,flush=True)
    return save('corrected_calibration.json',dict(command_offsets_mm=cfg['command_offsets_mm'],rows=rows,weight_n=object_weight(scene),
        note='Command target and realized material-point motion are distinct. A lost contact is a measured loss, not an invented force; switched branches terminate.'))


def normal_cone(normals, support_force):
    normals=np.asarray(normals,dtype=float).reshape(-1,3); target=np.asarray(support_force,dtype=float)
    if len(normals)==0: return dict(feasible=False,angular_distance_deg=180.0,closest_vector_n=[0,0,0],weights_n=[],residual_n=float(np.linalg.norm(target)))
    weights,residual=nnls(normals.T,target); projected=normals.T@weights
    if np.linalg.norm(target)<1e-15:
        angle=0.0
    elif np.linalg.norm(projected)<1e-15:
        # In the polar cone the projection is the origin, which has no angle.
        # The nearest nonzero cone direction is a generator in this case.
        cosine=max((n@target)/(np.linalg.norm(n)*np.linalg.norm(target)) for n in normals)
        angle=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
    else:
        angle=float(np.rad2deg(np.arccos(np.clip(target@projected/(np.linalg.norm(target)*np.linalg.norm(projected)),-1,1))))
    return dict(feasible=bool(residual<=config().mechanics['projection_tolerance_n']),angular_distance_deg=angle,
                closest_vector_n=projected.tolist(),weights_n=weights.tolist(),residual_n=float(residual))


def tangent_basis(normal):
    n=np.asarray(normal); helper=np.eye(3)[np.argmin(np.abs(n))]; t=np.cross(n,helper); t/=np.linalg.norm(t)
    return np.column_stack((n,t,np.cross(n,t)))


def friction_utilization(normal_force,tangent_force,mu):
    if normal_force<=config().mechanics['numerical_force_zero_n']: return None
    return 0.0 if np.linalg.norm(tangent_force)<=1e-12 else (float(np.linalg.norm(tangent_force)/(mu*normal_force)) if mu>0 else None)


def equilibrium(network, gravity_direction, friction_scale=1.0, *, outer=False):
    """Conservative 64-ray Coulomb SOCP approximation, full 6D equalities.

    Inscribed friction circle has at most 1-cos(pi/64) radial conservatism.
    No torsional/rolling capacity is folded into translational utilization.
    """
    contacts=network['contacts']; weight=network['weight_n']; radius=old.SPHERE_RADIUS_M
    target=np.r_[-np.asarray(gravity_direction)*weight,np.zeros(3)]
    if not contacts: return dict(feasible=False,failure='NO_CONTACT_NETWORK',rho_max=None)
    columns=[]; owners=[]; generators=[]
    for i,c in enumerate(contacts):
        basis=tangent_basis(c['inward_normal_palm']); mu=c['friction']*friction_scale
        if outer: mu/=np.cos(np.pi/config().mechanics['cone_rays'])
        angles=np.linspace(0,2*np.pi,int(config().mechanics['cone_rays']),endpoint=False) if mu>0 else [0]
        for a in angles:
            force=basis@np.array([1,mu*np.cos(a),mu*np.sin(a)]); columns.append(np.r_[force,np.cross(c['lever_arm_palm_m'],force)/radius]); owners.append(i); generators.append(force)
    matrix=np.asarray(columns).T; rhs=target.copy(); rhs[3:]/=radius
    solved=linprog(np.ones(matrix.shape[1]),A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs')
    if not solved.success:
        result=dict(feasible=False,failure='OFFLINE_CONE_INFEASIBLE',solver_status=int(solved.status),rho_max=None,
                    model='translational point forces; no spin/rolling moments')
        if friction_scale>0 and not outer:
            certificate=equilibrium(network,gravity_direction,friction_scale,outer=True)
            result['outer_cone_feasible']=certificate['feasible']
            result['circular_cone_infeasibility_certified']=not certificate['feasible'] and certificate.get('solver_status')==2
        return result
    forces=np.zeros((len(contacts),3))
    for i,coefficient,force in zip(owners,solved.x,generators): forces[i]+=coefficient*force
    descriptors=[]; rho=[]; torque=np.zeros(3)
    for c,f in zip(contacts,forces):
        normal=np.asarray(c['inward_normal_palm']); fn=float(f@normal); tangent=f-fn*normal; torque+=np.cross(c['lever_arm_palm_m'],f)
        utilization=friction_utilization(fn,tangent,c['friction']); rho.extend([] if utilization is None else [utilization])
        descriptors.append(dict(surface=c['surface'],normal_force_n=fn,tangential_force_palm_n=tangent.tolist(),friction_utilization_actual_mu=utilization))
    return dict(feasible=True,force_residual_n=(forces.sum(axis=0)-target[:3]).tolist(),torque_residual_nm=torque.tolist(),
                forces=descriptors,total_normal_force_n=float(sum(x['normal_force_n'] for x in descriptors)),
                rho_max=max(rho,default=0),rho_mean=float(np.mean(rho)) if rho else 0,rho_median=float(np.median(rho)) if rho else 0)


def compiled_wrench_equilibrium(network, gravity_direction, scale=1.0):
    """Reference elliptic condim-6 wrench cone, separate from point-force audit.

    Offline scale multiplies translational coefficients ONLY. At zero scale,
    spin/rolling capacity remains; this is not the frictionless diagnostic.
    Unknown solver outcomes are never labeled physical infeasibility.
    """
    contacts=network['contacts']; count=len(contacts); radius=old.SPHERE_RADIUS_M
    if not count: return dict(feasible=False,status='NO_CONTACTS')
    columns=[]; bases=[]; coefficients=[]
    for c in contacts:
        basis=tangent_basis(c['inward_normal_palm']); bases.append(basis)
        coefficients.append(np.asarray(c['runtime_friction'])*np.array([scale,scale,1,1,1]))
        mu=coefficients[-1]; r=c['lever_arm_palm_m']
        for j in range(6):
            f=np.zeros(3); moment=np.zeros(3)
            if j==0: f=basis[:,0]
            elif j<3: f=basis[:,j]*mu[j-1]
            else: moment=basis[:,j-3]*mu[j-1]
            columns.append(np.r_[f,(np.cross(r,f)+moment)/radius])
    matrix=np.asarray(columns).T; rhs=np.r_[-np.asarray(gravity_direction)*network['weight_n'],np.zeros(3)]
    u,singular,_=np.linalg.svd(matrix,full_matrices=False); rank=int(np.sum(singular>1e-10))
    projection=u[:,:rank]; reduced=projection.T@matrix; reduced_rhs=projection.T@rhs
    if np.linalg.norm(projection@reduced_rhs-rhs)>1e-7:
        return dict(feasible=False,status='LINEAR_WRENCH_INFEASIBLE',rho_max=None)
    def cone(x):
        rows=x.reshape(count,6); return rows[:,0]-np.linalg.norm(rows[:,1:],axis=1)
    # Polyhedral OUTER approximations plus exact Euclidean-norm separation.
    # An infeasible outer LP certifies infeasibility of the elliptic cone;
    # a feasible result is returned only after checking the actual cone.
    cuts=[]
    for i in range(count):
        for axis in range(5):
            for sign in (-1,1):
                cut=np.zeros(count*6); cut[6*i]=-1; cut[6*i+1+axis]=sign; cuts.append(cut)
    cost=np.zeros(count*6); cost[::6]=1
    for iteration in range(500):
        solution=linprog(cost,A_eq=reduced,b_eq=reduced_rhs,A_ub=np.asarray(cuts),b_ub=np.zeros(len(cuts)),
            bounds=[(0,None) if i%6==0 else (None,None) for i in range(6*count)],method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not solution.success:
            return dict(feasible=False,status='ELLIPTIC_CONE_INFEASIBLE' if solution.status==2 else 'SOLVER_UNRESOLVED',
                        solver_message=solution.message,rho_max=None,outer_lp_iterations=iteration+1)
        slack=cone(solution.x)
        if np.min(slack)>=-1e-9: break
        for i,x in enumerate(solution.x.reshape(count,6)):
            if slack[i]<-1e-9:
                cut=np.zeros(count*6); cut[6*i]=-1; cut[6*i+1:6*i+6]=x[1:]/np.linalg.norm(x[1:]); cuts.append(cut)
    residual=matrix@solution.x-rhs; slack=cone(solution.x)
    valid=bool(solution.success and np.max(np.abs(residual))<1e-7 and np.min(slack)>-1e-7)
    records=[]
    for c,basis,mu,x in zip(contacts,bases,coefficients,solution.x.reshape(count,6)):
        fn=float(x[0]); ft=x[1:3]*mu[:2]; moments=x[3:]*mu[2:]
        records.append(dict(surface=c['surface'],normal_force_n=fn,tangential_components_n=ft.tolist(),
            spin_rolling_moments_nm=moments.tolist(),rho_translation_actual_mu=friction_utilization(fn,ft,c['friction']),
            elliptic_cone_utilization=float(np.linalg.norm(x[1:])/fn) if fn>1e-9 else None))
    rho=[r['rho_translation_actual_mu'] for r in records if r['rho_translation_actual_mu'] is not None]
    return dict(feasible=valid,status='FEASIBLE' if valid else 'SOLVER_UNRESOLVED',solver_message=str(solution.message),
        model='compiled elliptic condim-6; separate translational, spin, rolling components',
        translational_friction_scale=scale,force_residual_n=residual[:3].tolist(),torque_residual_nm=(residual[3:]*radius).tolist(),
        forces=records,total_normal_force_n=float(sum(r['normal_force_n'] for r in records)),
        rho_max=max(rho,default=0),rho_mean=float(np.mean(rho)) if rho else 0,rho_median=float(np.median(rho)) if rho else 0,
        outer_lp_iterations=iteration+1,cone_slack_n=slack.tolist(),
        objective='minimum total normal load; rho is descriptive, not minimum-rho optimum')


def friction_curve(network,direction):
    rows=[dict(scale=float(s),solution=equilibrium(network,direction,float(s))) for s in config().mechanics['friction_scales']]
    feasible=[r['scale'] for r in rows if r['solution']['feasible']]
    return dict(rows=rows,minimum_tested_feasible_scale=min(feasible) if feasible else None)


def minimum_rho_solution(network,direction):
    zero=equilibrium(network,direction,0)
    if zero['feasible']: return zero
    actual=equilibrium(network,direction,1)
    if not actual['feasible']: return actual
    lo,hi=0.,1.; best=actual
    for _ in range(int(config().mechanics['rho_bisections'])):
        mid=(lo+hi)/2; solution=equilibrium(network,direction,mid)
        if solution['feasible']: hi=mid; best=solution
        else: lo=mid
    best['minimum_feasible_scale_bracket']=[lo,hi]; return best


def network_from_scene(scene,identifier,allowed):
    origin,rotation=palm_transform(scene); center=scene.data.xpos[scene.object_body_id].copy(); rows=[]
    for r in contact_rows(scene):
        if r['surface'] not in allowed: continue
        p=np.asarray(r['position_world_m']); n=np.asarray(r['inward_normal_world'])
        rows.append(dict(**r,position_palm_m=(rotation.T@(p-origin)).tolist(),inward_normal_palm=(rotation.T@n).tolist(),
                         outward_normal_palm=(-rotation.T@n).tolist(),lever_arm_palm_m=(rotation.T@(p-center)).tolist(),
                         friction=r['runtime_friction'][0],achievable_normal_command_direction='local material-point Jacobian projection',
                         preload_capacity_note='see corrected calibration; not a state-independent capacity guarantee'))
    return dict(candidate_id=identifier,qpos=scene.data.qpos.tolist(),center_palm_m=(rotation.T@(center-origin)).tolist(),
                contacts=rows,weight_n=object_weight(scene),original_gravity_direction= (rotation.T@scene.model.opt.gravity/9.81).tolist())


def archived_candidates():
    roles=read_old('storage_role_mechanics.json')['roles']; b03=read_old('preloaded_B03_manifest.json')['rows']; result=[]
    for family,identifier in [('ROLE-MRL','ROLE_MRL_05'),('ROLE-T','ROLE_T_03')]:
        r=roles[family]; trial=next(x for x in r['selected'] if x['candidate_id']==identifier); init=next(x for x in r['initialized'] if x['candidate_id']==identifier)
        scene=build_forearm_scene(with_actuator=True).scene; old._apply_trial_pose(scene,trial,np.asarray(init['qpos'])); set_fixture(scene,False); mujoco.mj_forward(scene.model,scene.data)
        result.append(network_from_scene(scene,identifier,r['definition']['storage']))
    best=min(b03,key=lambda r:(-len(r['active_storage_contacts']),r['maximum_penetration_m'],r['trial_id']))
    trial=next(t for t in old._manifest()['trials'] if t['trial_id']==best['trial_id']); scene=build_forearm_scene(with_actuator=True).scene; old._apply_trial_pose(scene,trial,np.asarray(best['qpos'])); set_fixture(scene,False); mujoco.mj_forward(scene.model,scene.data)
    n=network_from_scene(scene,'B03_PREVIOUS_BEST',('middle','ring','little','palm')); n['source_trial_id']=best['trial_id']; n['selection_rule']='most active storage surfaces, then least attempted penetration; no dynamic outcome'; result.append(n)
    return result


def orientation_bounds(scene):
    bounds=np.asarray([scene.model.jnt_range[mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ORIENTATION_NAMES]).copy()
    diagnostic=np.deg2rad(load_phase3c08_config()['forearm']['diagnostic_range_deg']); bounds[0]=[max(bounds[0,0],diagnostic[0]),min(bounds[0,1],diagnostic[1])]
    return bounds


def reachable_orientations(scene):
    bounds=orientation_bounds(scene); grids=[np.linspace(a,b,n) for (a,b),n in zip(bounds,config().orientation['coarse_counts'])]
    return np.asarray(list(itertools.product(*grids)))


def storage_orientation_audit(network):
    scene=build_forearm_scene(with_actuator=True).scene; scene.data.qpos[:]=network['qpos']; normals=[c['inward_normal_palm'] for c in network['contacts']]
    def direction(q): return _gravity_at(scene,dict(zip(ORIENTATION_NAMES,q)))
    def objective(q): return normal_cone(normals,-direction(q)*network['weight_n'])['residual_n']**2/network['weight_n']**2
    coarse=reachable_orientations(scene); values=list(coarse); ranked=sorted(coarse,key=objective)
    for q in ranked[:int(config().orientation['refinement_starts'])]:
        solution=minimize(objective,q,bounds=orientation_bounds(scene),method='L-BFGS-B',options={'maxiter':int(config().orientation['refinement_iterations']),'ftol':1e-15,'gtol':1e-10})
        values.append(solution.x)
    rows=[]
    for i,q in enumerate(values):
        g=direction(q); cone=normal_cone(normals,-g*network['weight_n']); frictionless=equilibrium(network,g,0); actual=minimum_rho_solution(network,g)
        rows.append(dict(configuration_rad=np.asarray(q).tolist(),gravity_direction_palm=g.tolist(),cone=cone,
                         frictionless=frictionless,actual_friction=actual,stage='coarse' if i<len(coarse) else 'refined'))
    def key(row):
        sol=row['actual_friction']; return (not row['frictionless']['feasible'],not sol['feasible'],round(row['cone']['angular_distance_deg'],6),sol.get('rho_max') if sol.get('rho_max') is not None else 1e9,sol.get('total_normal_force_n',1e9))
    best=min(rows,key=key); worst=max(rows,key=key)
    original_g=np.asarray(network['original_gravity_direction']); original=dict(cone=normal_cone(normals,-original_g*network['weight_n']),frictionless=equilibrium(network,original_g,0),actual_friction=minimum_rho_solution(network,original_g),friction_curve=friction_curve(network,original_g))
    transport=json.loads((ROOT/'outputs/phase3C08/kinematic_audit.json').read_text())['reachable_gravity_audit']['rows'][0]
    tq=transport['augmented']; transport_q=[tq['best_forearm_PS_rad'],tq['best_WRJ1_rad'],tq['best_WRJ2_rad']]
    tg=direction(transport_q); comparison=dict(source_state_id=transport['state_id'],selection='first archived transport state, fixed before storage comparison',
        transport_configuration_rad=transport_q,transport_gravity_direction=tg.tolist(),transport_actual_friction=minimum_rho_solution(network,tg),
        gravity_direction_difference_deg=float(np.rad2deg(np.arccos(np.clip(tg@best['gravity_direction_palm'],-1,1)))))
    return dict(network=network,original=original,best=best,worst=worst,rows=rows,
                sampled_feasible_count=sum(r['actual_friction']['feasible'] for r in rows),sampled_frictionless_count=sum(r['frictionless']['feasible'] for r in rows),sample_count=len(rows),
                best_friction_curve=friction_curve(network,best['gravity_direction_palm']),region_measure='sample counts only; not a solid-angle volume',
                cone_approximation_max_radial_error=1-np.cos(np.pi/config().mechanics['cone_rays']),transport_comparison=comparison)


def old_role_t_audit():
    role=read_old('storage_role_mechanics.json')['roles']['ROLE-T']
    return save('old_role_t_audit.json',dict(thumb_required_by_prefilter=False,thumb_required_by_closure=False,
        explanation='ROLE-T merely allowed thumb among storage digits. The prefilter counted any two nearby surfaces and closure targeted the two nearest digit surfaces; neither required thumb. Topology reporting correctly reported ring+little.',
        candidates=[dict(candidate_id=r['candidate_id'],target_surfaces=r['target_surfaces'],
                         actual_support=r['load_bearing_storage_topology'],thumb_force_n=r['actual_normal_force_n'].get('thumb',0)) for r in role['initialized']],
        conclusion='The old ROLE-T experiment did not test mandatory thumb-assisted storage, so its result cannot reject thumb participation.'))


def is_true_role_t(network,require_preload=True):
    support={c['surface'] for c in network['contacts'] if not require_preload or c['normal_force_n']>config().mechanics['numerical_force_zero_n']}
    return 'thumb' in support and bool(support.intersection(('ring','little','palm'))) and not support.intersection(('index','middle','environment'))


def true_role_t_search():
    cfg=config().true_role_t; sources=read_old('storage_role_mechanics.json')['roles']['ROLE-T']['selected'][:cfg['source_count']]; rows=[]
    for source,partner in itertools.product(sources,cfg['partners']):
        scene=build_forearm_scene(with_actuator=True).scene; q=np.asarray(source['qpos']); scene.data.qpos[:]=q
        for free in ('index','middle'): _finger_interpolation(scene,q,free,0.0)
        q=scene.data.qpos.copy(); center=np.asarray(source['center_palm_m']); _set_object_palm(scene,center)
        digits=('thumb',) if partner=='palm' else ('thumb',partner)
        joints=np.concatenate([scene.joint_ids[s] for s in digits]); addresses=scene.model.jnt_qposadr[joints]; x0=np.r_[q[addresses],center]
        radius=float(cfg['center_radius_m']); lower=np.r_[scene.model.jnt_range[joints,0],center-radius]; upper=np.r_[scene.model.jnt_range[joints,1],center+radius]
        oid=_object_geom_id(scene); all_geoms=old._all_hand_collision_geom_ids(scene)
        def assign(x):
            scene.data.qpos[:]=q; scene.data.qpos[addresses]=x[:-3]; mujoco.mj_forward(scene.model,scene.data); _set_object_palm(scene,x[-3:])
        def residual(x):
            assign(x); gaps=old._surface_distances(scene,('thumb',partner,'index','middle'))
            collision=np.asarray([old._pair_distance(scene,oid,g)[0] for g in all_geoms])
            return np.r_[[gaps['thumb']*1000,gaps[partner]*1000],np.minimum(collision,0)*1000,
                         (x[:-3]-x0[:-3])*.0001,(x[-3:]-center)*.01]
        fit=least_squares(residual,np.clip(x0,lower+1e-10,upper-1e-10),bounds=(lower,upper),max_nfev=int(cfg['tangent_solver_evaluations']),ftol=1e-9,xtol=1e-9,gtol=1e-9)
        assign(fit.x); tangent_gaps=old._surface_distances(scene,('thumb',partner)); commands=[]
        for surface in (partner,'thumb'):
            geom=min(_geom_ids(scene,surface),key=lambda g:old._pair_distance(scene,oid,g)[0]); gap,points=old._pair_distance(scene,oid,geom)
            # Separated closest points define the outward support normal. At exact
            # tangent use the object-center direction from the surface witness.
            normal=scene.data.geom_xpos[oid]-points[3:]; normal/=max(np.linalg.norm(normal),1e-15)
            body=int(scene.model.geom_bodyid[geom]); p=points[3:]; local=scene.data.xmat[body].reshape(3,3).T@(p-scene.data.xpos[body])
            offset=float(cfg['command_offset_mm'])/1000
            if surface=='palm':
                old._assign_object_world(scene,scene.data.geom_xpos[oid]-normal*(max(gap,0)+offset)); command=dict(surface=surface,method='Cartesian object fixture displacement relative to fixed palm',normal_offset_m=offset)
            else:
                ik=cartesian_ik(scene,body,local,p+normal*(max(gap,0)+offset),np.asarray(scene.joint_ids[surface])); command=dict(surface=surface,method='material-point Jacobian IK',normal_offset_m=offset,ik_error_m=ik['cartesian_error_m'])
            commands.append(command)
        scene.data.qvel[:]=0; scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); set_fixture(scene,False); mujoco.mj_forward(scene.model,scene.data)
        identifier=f'ROLE_T_TRUE_{len(rows):02d}'; network=network_from_scene(scene,identifier,SUPPORT_SURFACES)
        row=dict(candidate_id=identifier,source_candidate_id=source['candidate_id'],partner=partner,tangent_solver_success=bool(fit.success),
                 tangent_gaps_m=tangent_gaps,commands=commands,network=network,
                 real_thumb_contact=any(c['surface']=='thumb' for c in network['contacts']),
                 true_role_geometric=is_true_role_t(network,False),true_role_preloaded=is_true_role_t(network,True),
                 maximum_penetration_m=max([-c['contact_dist_m'] for c in network['contacts']]+[0]))
        rows.append(row); print('true thumb candidate',len(rows),partner,row['true_role_preloaded'],flush=True)
    eligible=[r for r in rows if r['true_role_preloaded']]
    geometric=[r for r in rows if r['true_role_geometric']]
    selected=min(eligible or geometric or rows,key=lambda r:(not r['true_role_geometric'],r['maximum_penetration_m'],r['candidate_id']))
    return save('true_role_t_search.json',dict(rows=rows,evaluated=len(rows),real_thumb_contact_count=sum(r['real_thumb_contact'] for r in rows),
        true_geometric_count=len(geometric),true_preloaded_count=len(eligible),selected_candidate_id=selected['candidate_id'],
        selected=selected,selection_rule='mandatory thumb+opposing support and no free-digit assistance; prefer preloaded then minimum overlap; before offline orientation outcomes',
        not_a_receiver_claim=True))


def true_role_workspace(candidate):
    network=candidate['network']; scene=build_forearm_scene(with_actuator=True).scene; base=np.asarray(network['qpos']); center=np.asarray(network['center_palm_m']); results={'baseline':{},'role_t_true':{}}
    baseline=base.copy(); scene.data.qpos[:]=baseline
    for s in ('thumb','ring','little'): _finger_interpolation(scene,baseline,s,0.0)
    baseline=scene.data.qpos.copy()
    for index,finger in enumerate(('index','middle')):
        samples=qmc.Sobol(d=len(scene.joint_ids[finger]),scramble=True,seed=int(config().workspace['seed'])+index).random_base2(int(np.log2(config().workspace['samples'])))
        results['baseline'][finger]=old._workspace_sample(scene,baseline,finger,samples,None,('thumb','ring','little'))
        results['role_t_true'][finger]=old._workspace_sample(scene,base,finger,samples,center,('thumb','ring','little'))
    for section in ('baseline','role_t_true'):
        results[section]['joint_acquisition']=old._opposition_descriptor(results[section]['index'],results[section]['middle'],config().workspace['diagnostic_aperture_m'])
        for pair in results[section]['joint_acquisition']['representative_pairs']:
            for suffix in ('point_palm_m','orientation_axis_palm'):
                pair['middle_'+suffix]=pair.pop('index_'+suffix)
                pair['index_'+suffix]=pair.pop('thumb_'+suffix)
    results['retained_fractions']={s:results['role_t_true'][s]['reachable_volume_m3']/results['baseline'][s]['reachable_volume_m3'] if results['baseline'][s]['reachable_volume_m3'] else None for s in ('index','middle')}
    before=results['baseline']['joint_acquisition']['opposition_midpoint_volume_m3']; after=results['role_t_true']['joint_acquisition']['opposition_midpoint_volume_m3']; results['retained_fractions']['joint_acquisition']=after/before if before else None
    results['qualification']='Kinematic aperture/midpoint descriptor only; side-by-side index/middle are not asserted to have thumb-like opposition. Independently sampled finger configurations are not a joint collision-free grasp proof.'
    results['source_candidate_id']=candidate['candidate_id']; results['source_true_preloaded']=candidate['true_role_preloaded']
    return save('true_role_t_workspace.json',results)


def run_mechanics():
    old_role_t_audit(); search=true_role_t_search(); networks=archived_candidates(); networks.append(search['selected']['network']); audits=[]
    for network in networks:
        result=storage_orientation_audit(network); audits.append(result); save('mechanics_audits.json',dict(candidates=audits)); print('orientation audit',network['candidate_id'],result['best']['actual_friction']['feasible'],flush=True)
    workspace=true_role_workspace(search['selected']); return audits,workspace


def verify_oriented_network(network, configuration):
    """Rigidly carry the sphere with the palm for a static geometry check only."""
    scene=build_forearm_scene(with_actuator=True).scene; scene.data.qpos[:]=network['qpos']
    _gravity_at(scene,dict(zip(ORIENTATION_NAMES,configuration)))
    _set_object_palm(scene,np.asarray(network['center_palm_m']))
    actual=network_from_scene(scene,network['candidate_id'],SUPPORT_SURFACES)
    errors=[]
    for before in network['contacts']:
        after=next((c for c in actual['contacts'] if c['geom_ids']==before['geom_ids']),None)
        errors.append(dict(surface=before['surface'],pair_preserved=after is not None,
            normal_error=None if after is None else float(np.linalg.norm(np.asarray(after['inward_normal_palm'])-before['inward_normal_palm'])),
            position_error_m=None if after is None else float(np.linalg.norm(np.asarray(after['position_palm_m'])-before['position_palm_m']))))
    return dict(errors=errors,all_pairs_preserved=all(e['pair_preserved'] for e in errors),
                all_object_contacts=contact_rows(scene),joint_bounds=orientation_bounds(scene).tolist(),
                world_gravity=scene.model.opt.gravity.tolist(),no_steps=True)


def complete_offline_audits():
    """Recompute offline certificates only, from already frozen contact networks."""
    previous=json.loads((OUTPUT/'mechanics_audits.json').read_text())['candidates']; results=[]
    for old_result in previous:
        network=old_result['network']; result=storage_orientation_audit(network)
        normals=np.asarray([c['inward_normal_palm'] for c in network['contacts']])
        result['normal_cone_rank']=int(np.linalg.matrix_rank(normals))
        result['normal_cone_generator_span_deg']=float(np.rad2deg(np.arccos(np.clip(normals[0]@normals[1],-1,1)))) if len(normals)==2 else None
        result['compiled_original']=compiled_wrench_equilibrium(network,network['original_gravity_direction'])
        result['compiled_original_friction_curve']=[dict(scale=s,solution=compiled_wrench_equilibrium(network,network['original_gravity_direction'],s)) for s in config().mechanics['friction_scales']]
        result['compiled_best']=compiled_wrench_equilibrium(network,result['best']['gravity_direction_palm'])
        result['compiled_transport']=compiled_wrench_equilibrium(network,result['transport_comparison']['transport_gravity_direction'])
        for row in result['rows']:
            row['compiled_wrench']=compiled_wrench_equilibrium(network,row['gravity_direction_palm'])
        result['sampled_compiled_feasible_count']=sum(r['compiled_wrench']['feasible'] for r in result['rows'])
        result['oriented_geometry_check']=verify_oriented_network(network,result['best']['configuration_rad'])
        results.append(result); print('offline certificates',network['candidate_id'],flush=True)
    return save('mechanics_audits.json',dict(candidates=results))


def audit_frozen_local_candidates():
    """Audit every already stored true-T attempt, without another geometry search."""
    search=json.loads((OUTPUT/'true_role_t_search.json').read_text()); rows=[]
    for candidate in search['rows']:
        network=candidate['network']; entry=dict(candidate_id=candidate['candidate_id'],
            true_geometric=candidate['true_role_geometric'],true_preloaded=candidate['true_role_preloaded'])
        if not candidate['true_role_geometric']:
            entry['status']='NOT_A_MANDATORY_THUMB_NETWORK'
        else:
            audit=storage_orientation_audit(network)
            entry.update(status='STATIC_NETWORK_ONLY',audit=audit,
                compiled_original=compiled_wrench_equilibrium(network,network['original_gravity_direction']),
                compiled_best=compiled_wrench_equilibrium(network,audit['best']['gravity_direction_palm']))
        rows.append(entry)
    return save('local_candidate_mechanics.json',dict(rows=rows,
        selected_candidate_unchanged=search['selected_candidate_id'],no_new_search=True,no_dynamics=True))
