from dataclasses import dataclass
import numpy as np
import mujoco
from ..sensing import extract_contacts, group_contacts_by_finger, compute_tactile_features

@dataclass(frozen=True)
class ObservationComponent:
    name: str; dimension: int; unit: str

def metadata(cfg):
    n, nf = cfg.hand.dof_count, len(cfg.hand.finger_geom_mapping); flags=cfg.task.observations
    candidates=[("joint_positions",n,"rad"),("joint_velocities",n,"rad/s"),("tactile_contact_flags",nf,"1"),("tactile_normal_forces",nf,"normalized N"),("palm_pose",7,"m, quaternion"),("phase_one_hot",5,"1"),("privileged_target_position",3,"m")]
    return [ObservationComponent(*x) for x in candidates if flags.get(x[0], False)]

def build_observation(model, data, cfg, phase):
    tactile=compute_tactile_features(group_contacts_by_finger(extract_contacts(model,data),cfg.hand.finger_geom_mapping),cfg)
    palm_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body)
    target=cfg.scene.objects[0 if int(phase)<2 else 1].name
    target_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,target)
    vals={"joint_positions":data.qpos[:cfg.hand.dof_count],"joint_velocities":data.qvel[:cfg.hand.dof_count],"tactile_contact_flags":tactile["contact_flags"],"tactile_normal_forces":tactile["normal_force"],"palm_pose":np.r_[data.xpos[palm_id],data.xquat[palm_id]],"phase_one_hot":np.eye(5,dtype=np.float32)[int(phase)],"privileged_target_position":data.xpos[target_id]}
    return np.concatenate([np.asarray(vals[m.name],dtype=np.float32) for m in metadata(cfg)]), tactile
