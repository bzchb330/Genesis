import mujoco
import numpy as np
from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene
from seqgrasp.sensing.contact import extract_contacts, group_contacts_by_finger
from seqgrasp.control import resolve_hand_indices

def _finger_contacts(model, data, cfg):
    return group_contacts_by_finger(extract_contacts(model, data), cfg.hand.finger_geom_mapping)

def test_scripted_allegro_contact_force_and_separation():
    cfg = load_configs(); model, data = build_scene(cfg)
    mujoco.mj_forward(model, data)
    assert not any(_finger_contacts(model, data, cfg).values())

    # Script the configured diagnostic closed pose, then put the configured object at a fingertip.
    profile=cfg.diagnostic.profiles[cfg.diagnostic.active_profile]
    indices=resolve_hand_indices(model,cfg.hand); fractions=np.asarray([profile.closed_joint_fractions[name] for name in cfg.hand.actuator_names]); ranges=model.jnt_range[indices.joint_ids]
    data.qpos[indices.qpos_addresses]=ranges[:,0]+fractions*(ranges[:,1]-ranges[:,0])
    mujoco.mj_forward(model, data)
    finger = next(iter(cfg.hand.finger_geom_mapping)); tip_name=cfg.hand.finger_geom_mapping[finger][0]
    tip_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tip_name)
    object_name=cfg.diagnostic.object_name
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_free")
    object_qpos = model.jnt_qposadr[object_joint]
    data.qpos[object_qpos:object_qpos + 3] = data.geom_xpos[tip_geom] + np.array([0.02, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    finger_contacts = _finger_contacts(model, data, cfg)[finger]
    assert finger_contacts and max(c.normal_force for c in finger_contacts) > 0.0
    record=next(c for c in finger_contacts if tip_name in {c.geom1_name,c.geom2_name})
    assert model.geom_bodyid[record.geom1_id]==record.body1_id
    assert model.geom_bodyid[record.geom2_id]==record.body2_id
    assert np.isclose(np.linalg.norm(record.normal),1.0)
    raw=np.zeros(6); contact_index=next(i for i in range(data.ncon) if tip_geom in {data.contact[i].geom1,data.contact[i].geom2})
    mujoco.mj_contactForce(model,data,contact_index,raw); frame=np.asarray(data.contact[contact_index].frame).reshape(3,3)
    np.testing.assert_allclose(record.force_world,frame.T@raw[:3])
    np.testing.assert_allclose(record.torque_world,frame.T@raw[3:])
    np.testing.assert_allclose(record.normal,frame[0])

    data.qpos[object_qpos:object_qpos + 3] = np.array([1.0, 1.0, 1.0])
    open_fractions=np.asarray([profile.open_joint_fractions[name] for name in cfg.hand.actuator_names])
    data.qpos[indices.qpos_addresses]=ranges[:,0]+open_fractions*(ranges[:,1]-ranges[:,0])
    mujoco.mj_forward(model, data)
    assert not any(_finger_contacts(model, data, cfg).values())
