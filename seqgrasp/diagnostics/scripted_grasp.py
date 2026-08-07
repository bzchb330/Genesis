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
    if set(fractions) != set(cfg.hand.actuator_names):
        raise ValueError("diagnostic joint fractions must match configured actuator names")
    f = np.asarray([fractions[name] for name in cfg.hand.actuator_names], dtype=float)
    if np.any((f < 0) | (f > 1)): raise ValueError("diagnostic joint fractions must lie in [0, 1]")
    ranges = model.jnt_range[indices.joint_ids]
    return ranges[:, 0] + f * (ranges[:, 1] - ranges[:, 0])

def _object_addresses(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
    if body_id < 0 or joint_id < 0: raise ValueError(f"missing configured diagnostic object {name}")
    return body_id, model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]

def _write_outputs(run: DiagnosticRun, cfg: ConfigBundle) -> None:
    out = run.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(run.metadata, indent=2), encoding="utf-8")
    if cfg.diagnostic.save_npz: np.savez_compressed(out / "timesteps.npz", **run.arrays)
    if cfg.diagnostic.save_csv:
        keys = list(run.arrays); rows = len(run.arrays["time"])
        columns=[]
        for key in keys:
            a=run.arrays[key]
            columns.extend([key] if a.ndim==1 else [f"{key}_{i}" for i in range(a.shape[1])])
        with (out / "timesteps.csv").open("w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(columns)
            for i in range(rows):
                row=[]
                for key in keys:
                    a=run.arrays[key]; row.extend([a[i]] if a.ndim==1 else a[i].tolist())
                writer.writerow(row)

def run_scripted_grasp(cfg: ConfigBundle, seed: int | None = None, output_dir: str | Path | None = None, render_video: bool | None = None, save_outputs: bool = True) -> DiagnosticRun:
    """Run a deterministic engineering-only object-A contact/retention probe."""
    diag=cfg.diagnostic
    if not diag.diagnostic_only: raise ValueError("scripted trajectory must remain diagnostic_only")
    seed=diag.seed if seed is None else seed; rng=np.random.default_rng(seed)
    model,data=build_scene(cfg); randomize_objects(model,data,cfg,rng)
    indices=resolve_hand_indices(model,cfg.hand); q0,_=hand_state(data,indices)
    open_q=_joint_target(model,cfg,indices,diag.open_joint_fractions)
    closed_q=_joint_target(model,cfg,indices,diag.closed_joint_fractions)
    object_id,object_qadr,object_vadr=_object_addresses(model,diag.object_name)
    fixture_pos=np.asarray(diag.object_fixture_pos,dtype=float).copy()
    fixture_pos[:2]+=rng.uniform(-diag.fixture_jitter_xy,diag.fixture_jitter_xy,2)
    fixture_quat=np.asarray(diag.object_fixture_quat,dtype=float)
    controller=JointImpedanceController(cfg.task.impedance_stiffness,cfg.task.impedance_damping,cfg.task.torque_limit)
    palm_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body)
    fingers=list(cfg.hand.finger_geom_mapping); records=[]; frames=[]; transitions=[]
    previous=q0.copy(); terminated_reason=None
    renderer=None; want_video=diag.render_video if render_video is None else render_video
    if want_video:
        try: renderer=mujoco.Renderer(model,cfg.scene.render_height,cfg.scene.render_width)
        except Exception as exc: warnings.warn(f"video disabled: {exc}")
    for stage,duration in diag.stage_durations_seconds.items():
        steps=max(1,round(duration/model.opt.timestep)); start=previous.copy()
        target=open_q if stage=="open_pregrasp" else closed_q
        phase=Phase(diag.episode_phase_by_stage[stage]); transitions.append({"time":float(data.time),"stage":stage,"episode_phase":int(phase),"reason":"diagnostic_time_schedule"})
        for k in range(steps):
            alpha=(k+1)/steps; desired=(1-alpha)*start+alpha*target
            q,qvel=hand_state(data,indices); torque=controller.torque(desired,q,qvel)
            data.ctrl[indices.actuator_ids]=torque
            if stage in diag.kinematic_fixture_stages:
                data.qpos[object_qadr:object_qadr+3]=fixture_pos; data.qpos[object_qadr+3:object_qadr+7]=fixture_quat
                data.qvel[object_vadr:object_vadr+6]=0.0
            mujoco.mj_step(model,data)
            contacts=extract_contacts(model,data); grouped=group_contacts_by_finger(contacts,cfg.hand.finger_geom_mapping)
            tactile=compute_tactile_features(grouped,cfg); q,qvel=hand_state(data,indices)
            velocity=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,object_id,velocity,0)
            row={"time":data.time,"episode_phase":int(phase),"diagnostic_stage":stage,"joint_positions":q,"joint_velocities":qvel,"actuator_controls":data.ctrl[indices.actuator_ids].copy(),"object_position":data.xpos[object_id].copy(),"object_orientation":data.xquat[object_id].copy(),"object_linear_velocity":velocity[3:].copy(),"object_angular_velocity":velocity[:3].copy(),"palm_pose":np.r_[data.xpos[palm_id],data.xquat[palm_id]],"contact_count":len(contacts),"finger_contact_count":np.asarray([len(grouped[f]) for f in fingers]),"finger_total_normal_force_raw":np.asarray([sum(c.normal_force for c in grouped[f]) for f in fingers]),"tactile_contact_flags":tactile["contact_flags"],"tactile_normal_force":tactile["normal_force"]}
            records.append(row)
            if renderer is not None and len(records)%diag.render_stride==0:
                try: renderer.update_scene(data); frames.append(renderer.render().copy())
                except Exception as exc: warnings.warn(f"video capture stopped: {exc}"); renderer.close(); renderer=None
            terminated_reason=failure_reason(model,data,cfg,phase)
            if terminated_reason: break
        previous=target.copy()
        if terminated_reason: break
    if renderer is not None: renderer.close()
    arrays={key:np.asarray([row[key] for row in records]) for key in records[0]}
    max_force=float(np.max(arrays["finger_total_normal_force_raw"]))
    metadata={"diagnostic_only":True,"seed":seed,"object_name":diag.object_name,"finger_order":fingers,"actuator_order":list(cfg.hand.actuator_names),"joint_order":list(cfg.hand.joint_names),"tactile_normalization":cfg.task.tactile_normalization,"force_units":"N" if cfg.task.tactile_normalization is None else "normalized N","fixture_position":fixture_pos.tolist(),"fixture_stages":list(diag.kinematic_fixture_stages),"phase_transitions":transitions,"terminated_early":terminated_reason is not None,"termination_reason":terminated_reason,"steps":len(records),"maximum_raw_finger_normal_force_N":max_force,"scientific_success_assigned":False,"video_requested":bool(want_video),"video_written":False}
    out=Path(output_dir) if output_dir is not None else ROOT/diag.output_dir
    run=DiagnosticRun(arrays,metadata,out if save_outputs else None)
    if save_outputs:
        _write_outputs(run,cfg)
        if cfg.diagnostic.save_plots:
            from .plotting import plot_diagnostics
            plot_diagnostics(run.arrays,run.metadata,out/"plots")
        if want_video and frames:
            try: imageio.mimsave(out/diag.video_filename,frames,fps=diag.video_fps); run.metadata["video_written"]=True; _write_outputs(run,cfg)
            except Exception as exc: warnings.warn(f"video encoding unavailable: {exc}")
    return run
