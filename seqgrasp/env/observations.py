from dataclasses import dataclass
import numpy as np
import mujoco
from ..sensing import extract_contacts, group_contacts_by_finger, compute_tactile_features

@dataclass(frozen=True)
class ObservationComponent:
    name: str; dimension: int; unit: str; source: str; privileged: bool; enabled: bool

def _catalog(cfg):
    n, nf = cfg.hand.dof_count, len(cfg.hand.finger_geom_mapping); flags=cfg.task.observations
    force_unit = "N" if cfg.task.tactile_normalization is None else "normalized N"
    candidates=[
        ("joint_positions",n,"rad","joint encoders",False),
        ("joint_velocities",n,"rad/s","joint encoders",False),
        ("tactile_contact_flags",nf,"1","reference tactile contact records",False),
        ("tactile_normal_forces",nf,force_unit,"reference tactile contact records",False),
        ("palm_pose",7,"m, quaternion","robot state estimator",False),
        ("phase_one_hot",5,"1","task state machine",False),
        ("privileged_target_position",3,"m","MuJoCo body pose",True),
    ]
    return [ObservationComponent(*x, bool(flags.get(x[0], False))) for x in candidates]

def metadata(cfg): return [component for component in _catalog(cfg) if component.enabled]

def observation_spec(cfg):
    return [{"name":m.name,"dimension":m.dimension,"unit":m.unit,"source":m.source,"privileged":m.privileged,"enabled":m.enabled} for m in _catalog(cfg)]

def build_observation(model, data, cfg, phase, indices=None):
    from ..control import hand_state, resolve_hand_indices
    indices = indices or resolve_hand_indices(model, cfg.hand)
    q, qvel = hand_state(data, indices)
    tactile=compute_tactile_features(group_contacts_by_finger(extract_contacts(model,data),cfg.hand.finger_geom_mapping),cfg)
    palm_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body)
    target=cfg.scene.objects[0 if int(phase)<2 else 1].name
    target_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,target)
    vals={"joint_positions":q,"joint_velocities":qvel,"tactile_contact_flags":tactile["contact_flags"],"tactile_normal_forces":tactile["normal_force"],"palm_pose":np.r_[data.xpos[palm_id],data.xquat[palm_id]],"phase_one_hot":np.eye(5,dtype=np.float32)[int(phase)],"privileged_target_position":data.xpos[target_id]}
    return np.concatenate([np.asarray(vals[m.name],dtype=np.float32) for m in metadata(cfg)]), tactile
