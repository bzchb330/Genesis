from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import warnings
import imageio.v2 as imageio
import mujoco
import numpy as np

from ..config import ConfigBundle, ROOT
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..env.termination import Phase, failure_reason
from ..scene_builder import build_scene, randomize_objects
from ..sensing import compute_tactile_features, extract_contacts, group_contacts_by_finger

@dataclass
class DiagnosticRun:
    arrays: dict[str, np.ndarray]
    metadata: dict
    output_dir: Path | None

def _joint_target(model, cfg, indices, fractions):
    if set(fractions) != set(cfg.hand.actuator_names): raise ValueError("diagnostic joint fractions must match configured actuator names")
    values=np.asarray([fractions[name] for name in cfg.hand.actuator_names],dtype=float)
    if np.any((values<0)|(values>1)): raise ValueError("diagnostic joint fractions must lie in [0, 1]")
    ranges=model.jnt_range[indices.joint_ids]; return ranges[:,0]+values*(ranges[:,1]-ranges[:,0])

def _object_addresses(model,name):
    body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name); joint_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"{name}_free")
    if body_id<0 or joint_id<0: raise ValueError(f"missing configured diagnostic object {name}")
    return body_id,model.jnt_qposadr[joint_id],model.jnt_dofadr[joint_id]

def _row_snapshot(row):
    keys=("time","object_position","object_orientation","object_linear_velocity","object_angular_velocity","finger_contact_count","tactile_contact_flags","finger_total_normal_force_raw","total_configured_fingertip_normal_force")
    return {key:(row[key].tolist() if isinstance(row[key],np.ndarray) else float(row[key])) for key in keys}

def _write_outputs(run,cfg):
    out=run.output_dir; out.mkdir(parents=True,exist_ok=True); (out/"metadata.json").write_text(json.dumps(run.metadata,indent=2),encoding="utf-8")
    if cfg.diagnostic.save_npz:
        np.savez_compressed(out/"timesteps.npz",**run.arrays)
        resource_keys=("time","joint_positions","joint_velocities","joint_limits","distance_to_joint_limits","actuator_controls","actuator_control_limits","active_fingers","active_finger_count","contact_count","active_contact_count","finger_contact_count","tactile_contact_flags","tactile_normal_force","palm_pose","object_position","object_orientation")
        np.savez_compressed(out/"resource_state.npz",**{key:run.arrays[key] for key in resource_keys})
    if cfg.diagnostic.save_csv:
        keys=list(run.arrays); rows=len(run.arrays["time"]); columns=[]
        for key in keys:
            shape=run.arrays[key].shape[1:]
            columns.extend([key] if not shape else [key+"_"+"_".join(map(str,index)) for index in np.ndindex(shape)])
        with (out/"timesteps.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(columns)
            for i in range(rows):
                row=[]
                for key in keys:
                    value=np.asarray(run.arrays[key][i]); row.extend([value.item()] if value.ndim==0 else value.reshape(-1).tolist())
                writer.writerow(row)

def run_scripted_grasp(cfg:ConfigBundle,seed:int|None=None,output_dir:str|Path|None=None,render_video:bool|None=None,save_outputs:bool=True,profile_name:str|None=None)->DiagnosticRun:
    """Run a deterministic engineering-only object contact/support-release probe."""
    diag=cfg.diagnostic
    if not diag.diagnostic_only: raise ValueError("scripted trajectory must remain diagnostic_only")
    profile_name=profile_name or diag.active_profile
    if profile_name not in diag.profiles: raise ValueError(f"unknown diagnostic profile {profile_name}")
    profile=diag.profiles[profile_name]
    if profile.support_release_stage not in profile.stage_durations_seconds: raise ValueError("support release stage must be present in the diagnostic schedule")
    if profile.support_release_stage in profile.kinematic_fixture_stages: raise ValueError("support release stage cannot remain fixture-held")
    seed=diag.seed if seed is None else seed; rng=np.random.default_rng(seed); model,data=build_scene(cfg); randomize_objects(model,data,cfg,rng)
    indices=resolve_hand_indices(model,cfg.hand); q0,_=hand_state(data,indices); joint_limits=model.jnt_range[indices.joint_ids].copy()
    open_q=_joint_target(model,cfg,indices,profile.open_joint_fractions); closed_q=_joint_target(model,cfg,indices,profile.closed_joint_fractions)
    object_id,object_qadr,object_vadr=_object_addresses(model,diag.object_name); object_cfg=next(obj for obj in cfg.scene.objects if obj.name==diag.object_name)
    fixture_pos=np.asarray(profile.object_fixture_pos,dtype=float).copy(); fixture_pos[:2]+=rng.uniform(-diag.fixture_jitter_xy,diag.fixture_jitter_xy,2); fixture_quat=np.asarray(profile.object_fixture_quat,dtype=float)
    half_height=object_cfg.size[2] if object_cfg.shape=="cube" else object_cfg.size[1]; table_top=cfg.scene.table_pos[2]+cfg.scene.table_size[2]; table_resting_center_z=table_top+half_height
    controller=JointImpedanceController(cfg.task.impedance_stiffness,cfg.task.impedance_damping,cfg.task.torque_limit); control_limits=np.tile([-cfg.task.torque_limit,cfg.task.torque_limit],(cfg.hand.dof_count,1))
    palm_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body); fingers=list(cfg.hand.finger_geom_mapping); records=[]; frames=[]; transitions=[]
    previous=q0.copy(); terminated_reason=None; release_time=None; release_before=None; release_after=None
    renderer=None; want_video=diag.render_video if render_video is None else render_video
    if want_video:
        try: renderer=mujoco.Renderer(model,cfg.scene.render_height,cfg.scene.render_width)
        except Exception as exc: warnings.warn(f"video disabled: {exc}")
    for stage,duration in profile.stage_durations_seconds.items():
        if stage==profile.support_release_stage:
            release_time=float(data.time); release_before=records[-1] if records else None
        steps=max(1,round(duration/model.opt.timestep)); start=previous.copy(); target=open_q if stage=="open_pregrasp" else closed_q; phase=Phase(profile.episode_phase_by_stage[stage])
        transitions.append({"time":float(data.time),"stage":stage,"episode_phase":int(phase),"reason":"diagnostic_time_schedule","support_active":stage in profile.kinematic_fixture_stages})
        for k in range(steps):
            alpha=(k+1)/steps; desired=(1-alpha)*start+alpha*target; q,qvel=hand_state(data,indices); torque=controller.torque(desired,q,qvel); data.ctrl[indices.actuator_ids]=torque
            support_active=stage in profile.kinematic_fixture_stages
            if support_active:
                data.qpos[object_qadr:object_qadr+3]=fixture_pos; data.qpos[object_qadr+3:object_qadr+7]=fixture_quat; data.qvel[object_vadr:object_vadr+6]=0.0
            mujoco.mj_step(model,data); contacts=extract_contacts(model,data); grouped=group_contacts_by_finger(contacts,cfg.hand.finger_geom_mapping); tactile=compute_tactile_features(grouped,cfg); q,qvel=hand_state(data,indices)
            velocity=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,object_id,velocity,0); finger_counts=np.asarray([len(grouped[f]) for f in fingers]); raw_forces=np.asarray([sum(c.normal_force for c in grouped[f]) for f in fingers]); active_fingers=(finger_counts>0).astype(np.int8)
            position=data.xpos[object_id].copy(); orientation=data.xquat[object_id].copy(); distance_to_limits=np.stack([q-joint_limits[:,0],joint_limits[:,1]-q],axis=1)
            displacement=np.zeros(3) if release_before is None else position-np.asarray(release_before["object_position"]); orientation_change=0.0 if release_before is None else 2*np.arccos(np.clip(abs(float(np.dot(orientation,release_before["object_orientation"]))),0,1))
            row={"time":data.time,"episode_phase":int(phase),"diagnostic_stage":stage,"support_active":int(support_active),"joint_positions":q,"joint_velocities":qvel,"joint_limits":joint_limits.copy(),"distance_to_joint_limits":distance_to_limits,"actuator_controls":data.ctrl[indices.actuator_ids].copy(),"actuator_control_limits":control_limits.copy(),"object_position":position,"object_orientation":orientation,"object_linear_velocity":velocity[3:].copy(),"object_angular_velocity":velocity[:3].copy(),"table_clearance":position[2]-half_height-table_top,"object_displacement_after_release":displacement,"object_translational_displacement_after_release":float(np.linalg.norm(displacement)),"object_orientation_change_after_release":float(orientation_change),"palm_pose":np.r_[data.xpos[palm_id],data.xquat[palm_id]],"contact_count":len(contacts),"active_contact_count":len(contacts),"finger_contact_count":finger_counts,"active_fingers":active_fingers,"active_finger_count":int(active_fingers.sum()),"finger_total_normal_force_raw":raw_forces,"total_configured_fingertip_normal_force":float(raw_forces.sum()),"tactile_contact_flags":tactile["contact_flags"],"tactile_normal_force":tactile["normal_force"]}
            records.append(row)
            if release_time is not None and release_after is None and stage==profile.support_release_stage: release_after=row
            if renderer is not None and len(records)%diag.render_stride==0:
                try: renderer.update_scene(data); frames.append(renderer.render().copy())
                except Exception as exc: warnings.warn(f"video capture stopped: {exc}"); renderer.close(); renderer=None
            terminated_reason=failure_reason(model,data,cfg,phase)
            if terminated_reason: break
        previous=target.copy()
        if terminated_reason: break
    if renderer is not None: renderer.close()
    arrays={key:np.asarray([row[key] for row in records]) for key in records[0]}; max_force=float(np.max(arrays["finger_total_normal_force_raw"])); release_event=None
    if release_before is not None and release_after is not None:
        release_event={"support_release_time":release_time,"before":_row_snapshot(release_before),"after":_row_snapshot(release_after),"immediate_vertical_displacement_m":float(release_after["object_position"][2]-release_before["object_position"][2]),"final_vertical_displacement_m":float(records[-1]["object_position"][2]-release_before["object_position"][2])}
    metadata={"diagnostic_only":True,"profile_name":profile_name,"seed":seed,"object_name":diag.object_name,"finger_order":fingers,"actuator_order":list(cfg.hand.actuator_names),"joint_order":list(cfg.hand.joint_names),"tactile_normalization":cfg.task.tactile_normalization,"force_units":"N" if cfg.task.tactile_normalization is None else "normalized N","fixture_position":fixture_pos.tolist(),"fixture_stages":list(profile.kinematic_fixture_stages),"support_release_event":release_event,"table_top_m":table_top,"table_resting_center_z_m":table_resting_center_z,"phase_transitions":transitions,"terminated_early":terminated_reason is not None,"termination_reason":terminated_reason,"steps":len(records),"maximum_raw_finger_normal_force_N":max_force,"scientific_success_assigned":False,"video_requested":bool(want_video),"video_written":False}
    out=Path(output_dir) if output_dir is not None else ROOT/diag.output_dir; run=DiagnosticRun(arrays,metadata,out if save_outputs else None)
    if save_outputs:
        _write_outputs(run,cfg)
        if diag.save_plots:
            from .plotting import plot_diagnostics
            plot_diagnostics(run.arrays,run.metadata,out/"plots")
        if want_video and frames:
            try: imageio.mimsave(out/diag.video_filename,frames,fps=diag.video_fps); run.metadata["video_written"]=True; _write_outputs(run,cfg)
            except Exception as exc: warnings.warn(f"video encoding unavailable: {exc}")
    return run
