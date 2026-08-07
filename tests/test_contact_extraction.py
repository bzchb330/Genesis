import mujoco
import numpy as np
from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene
from seqgrasp.sensing.contact import extract_contacts, group_contacts_by_finger

def _finger_contacts(model, data, cfg):
    return group_contacts_by_finger(extract_contacts(model, data), cfg.hand.finger_geom_mapping)

def test_scripted_allegro_contact_force_and_separation():
    cfg = load_configs(); model, data = build_scene(cfg)
    mujoco.mj_forward(model, data)
    assert not any(_finger_contacts(model, data, cfg).values())

    # Script a closed index configuration, then put object A at its fingertip.
    for joint_name in cfg.hand.joint_names[1:4]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = 0.8 * model.jnt_range[jid, 1]
    mujoco.mj_forward(model, data)
    tip_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping["index"][0])
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    object_qpos = model.jnt_qposadr[object_joint]
    data.qpos[object_qpos:object_qpos + 3] = data.geom_xpos[tip_geom] + np.array([0.02, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    index_contacts = _finger_contacts(model, data, cfg)["index"]
    assert index_contacts and max(c.normal_force for c in index_contacts) > 0.0

    data.qpos[object_qpos:object_qpos + 3] = np.array([1.0, 1.0, 1.0])
    mujoco.mj_forward(model, data)
    assert not any(_finger_contacts(model, data, cfg).values())
