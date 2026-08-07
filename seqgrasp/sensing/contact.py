from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np

@dataclass(frozen=True)
class ContactRecord:
    geom1_id: int; geom2_id: int; geom1_name: str; geom2_name: str
    body1_id: int; body2_id: int; body1_name: str; body2_name: str
    position: np.ndarray; normal: np.ndarray; force_world: np.ndarray; torque_world: np.ndarray
    normal_force: float; tangential_force: float

def _name(model, kind, idx): return mujoco.mj_id2name(model, kind, int(idx)) or ""

def extract_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[ContactRecord]:
    out = []
    for i in range(data.ncon):
        c = data.contact[i]; wrench = np.zeros(6); mujoco.mj_contactForce(model, data, i, wrench)
        # MuJoCo stores contact-frame axes as rows; wrench is expressed in that frame.
        frame = np.asarray(c.frame).reshape(3, 3)
        fw, tw = frame.T @ wrench[:3], frame.T @ wrench[3:]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        out.append(ContactRecord(int(c.geom1), int(c.geom2), _name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1), _name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2), int(b1), int(b2), _name(model,mujoco.mjtObj.mjOBJ_BODY,b1), _name(model,mujoco.mjtObj.mjOBJ_BODY,b2), np.asarray(c.pos).copy(), frame[0].copy(), fw, tw, abs(float(wrench[0])), float(np.linalg.norm(wrench[1:3]))))
    return out

def group_contacts_by_finger(records, mapping):
    result = {finger: [] for finger in mapping}
    for r in records:
        names = {r.geom1_name, r.geom2_name}
        for finger, geoms in mapping.items():
            if names.intersection(geoms): result[finger].append(r)
    return result

