from __future__ import annotations

from dataclasses import asdict, replace
import hashlib

import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2cm import (
    STATE_SPEC,
    _finger_b_wrenches,
    _load_state,
    _restore_state,
    construct_contact_variant,
    replay_release_state,
    select_frozen_trials,
    verify_model_isolation,
)
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2cm_config import load_phase2cm_config


def _trial(wrist: str = "wrist_0", index: int = 0) -> dict:
    return {
        "trial_id": f"trial_{wrist}_{index:03d}",
        "wrist_pose_id": wrist,
        "wrist_pose": {"relative_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
    }


def _actual_contact(condim: int):
    base = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    cfg, model, data = construct_contact_variant(base, _trial(), condim)
    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ff_tip")
    b_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    qadr = model.jnt_qposadr[b_joint]
    mujoco.mj_forward(model, data)
    data.qpos[qadr:qadr + 3] = data.xpos[tip_body]
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    return cfg, model, data


def test_runtime_condim_extraction_uses_actual_mujoco_contact():
    for condim in (3, 4, 6):
        cfg, model, data = _actual_contact(condim)
        counts, dims, *_ = _finger_b_wrenches(model, data, cfg)
        assert counts[0] > 0
        assert dims[0] == condim
        actual = [int(contact.dim) for contact in data.contact if {int(contact.geom1), int(contact.geom2)} == {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ff_tip_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom"),
        }]
        assert actual and set(actual) == {condim}


def test_cm_variants_change_only_fingertip_condim_and_preserve_physics():
    cfg_cm, _ = load_phase2cm_config()
    base = load_configs(scene_filename=cfg_cm.source_scene_filename)
    report = verify_model_isolation(base, _trial(), cfg_cm.variants, write_artifact=False)
    assert report["status"] == "PASS"
    assert not report["failures"]
    built = {name: construct_contact_variant(base, _trial(), value)[1] for name, value in cfg_cm.variants.items()}
    reference = built["CM3"]
    for model in built.values():
        np.testing.assert_array_equal(model.geom_friction, reference.geom_friction)
        np.testing.assert_array_equal(model.body_mass, reference.body_mass)
        np.testing.assert_array_equal(model.body_inertia, reference.body_inertia)
        np.testing.assert_array_equal(model.geom_size, reference.geom_size)
        np.testing.assert_array_equal(model.geom_pos, reference.geom_pos)
        np.testing.assert_array_equal(model.geom_quat, reference.geom_quat)
        np.testing.assert_array_equal(model.geom_solref, reference.geom_solref)
        np.testing.assert_array_equal(model.geom_solimp, reference.geom_solimp)
        np.testing.assert_array_equal(model.actuator_gainprm, reference.actuator_gainprm)


def test_200_state_freeze_is_deterministic_hash_ordered_and_covers_ten_wrists():
    rows = [_trial(f"wrist_{wrist}", index) for wrist in range(10) for index in range(30)]
    first = select_frozen_trials(rows, 200)
    second = select_frozen_trials(list(reversed(rows)), 200)
    assert [row["trial_id"] for row in first] == [row["trial_id"] for row in second]
    assert len(first) == len({row["trial_id"] for row in first}) == 200
    assert {row["wrist_pose_id"] for row in first} == {f"wrist_{index}" for index in range(10)}
    for wrist in {row["wrist_pose_id"] for row in first}:
        selected = [row for row in first if row["wrist_pose_id"] == wrist]
        assert selected == sorted(selected, key=lambda row: hashlib.sha256(row["trial_id"].encode()).hexdigest())


def _frozen_initial_state(base, trial):
    _, model, data = construct_contact_variant(base, trial, 3)
    mujoco.mj_forward(model, data)
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    b_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in base.hand.joint_names]
    return {
        "state_spec": np.asarray(int(STATE_SPEC)),
        "integration_state": state,
        "B_position": data.xpos[b_body].copy(),
        "B_quaternion": data.xquat[b_body].copy(),
        "post_release_controller_target": data.qpos[model.jnt_qposadr[joint_ids]].copy(),
    }


def test_release_state_equivalence_and_paired_replay_determinism(tmp_path):
    cfg_cm, _ = load_phase2cm_config()
    cfg25, _ = load_phase2_5_config()
    base = load_configs(scene_filename=cfg_cm.source_scene_filename)
    trial = _trial()
    frozen = _frozen_initial_state(base, trial)
    restored = []
    for condim in cfg_cm.variants.values():
        _, model, data = construct_contact_variant(base, trial, condim)
        _restore_state(model, data, frozen)
        restored.append((data.qpos.copy(), data.qvel.copy()))
    for qpos, qvel in restored[1:]:
        np.testing.assert_array_equal(qpos, restored[0][0])
        np.testing.assert_array_equal(qvel, restored[0][1])

    short = replace(cfg_cm, post_release_steps=5)
    output = ROOT / "tmp" / "pytest_phase2cm" / tmp_path.name
    first_path = output / "first.npz"
    second_path = output / "second.npz"
    first = replay_release_state(short, cfg25, base, trial, "CM3", frozen, first_path)
    second = replay_release_state(short, cfg25, base, trial, "CM3", frozen, second_path)
    for key in first:
        if key != "timeseries_path":
            assert first[key] == second[key]
    with np.load(first_path, allow_pickle=False) as left, np.load(second_path, allow_pickle=False) as right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


def test_phase2cm_reuses_frozen_criteria_friction_mass_geometry_and_controller():
    cfg_cm, _ = load_phase2cm_config()
    cfg25, _ = load_phase2_5_config()
    base = load_configs(scene_filename=cfg_cm.source_scene_filename)
    criteria = asdict(cfg25.criteria)
    controller = (base.task.impedance_stiffness, base.task.impedance_damping, base.task.torque_limit)
    models = [construct_contact_variant(base, _trial(), dim)[1] for dim in cfg_cm.variants.values()]
    assert asdict(cfg25.criteria) == criteria
    assert (base.task.impedance_stiffness, base.task.impedance_damping, base.task.torque_limit) == controller
    for model in models[1:]:
        np.testing.assert_array_equal(model.geom_friction, models[0].geom_friction)
        np.testing.assert_array_equal(model.body_mass, models[0].body_mass)
        np.testing.assert_array_equal(model.body_inertia, models[0].body_inertia)
        np.testing.assert_array_equal(model.geom_size, models[0].geom_size)
