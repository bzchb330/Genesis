from __future__ import annotations

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from seqgrasp import load_configs
from seqgrasp.scene_builder import build_scene
from seqgrasp.phase3.config import FINGERS, SUPPORT_SURFACES, load_phase3_config
from seqgrasp.phase3.contacts import extract_shadow_contacts, fingertip_object_penetration
from seqgrasp.phase3.control import ContactAwareCloser, VariableImpedanceController
from seqgrasp.phase3.env import Phase3ShadowHandEnv
from seqgrasp.phase3.events import HandoffEvent, HandoffEventDetector
from seqgrasp.phase3.experiments import run_acquisition_candidate
from seqgrasp.phase3.model import build_shadow_scene, set_fixture, set_object_pose
from seqgrasp.phase3.resource import compute_resource_snapshot
from seqgrasp.phase3.roles import FingerRole, ManipulationPhase, PalmRole, RoleState


def test_shadow_hand_loads_with_five_semantic_fingers_palm_and_wrist():
    scene = build_shadow_scene()
    assert scene.model.nbody == 28  # world + 25 hand + object + mocap fixture
    assert scene.model.njnt == 25  # 24 hand joints + object freejoint
    assert scene.model.nu == 20
    assert tuple(scene.config.hand.finger_order) == FINGERS
    assert scene.config.hand.palm_body == "rh_palm"
    assert scene.config.hand.wrist_joints == ("rh_WRJ2", "rh_WRJ1")
    for name in (*scene.config.hand.wrist_joints, scene.config.hand.palm_body):
        kind = mujoco.mjtObj.mjOBJ_JOINT if "WRJ" in name else mujoco.mjtObj.mjOBJ_BODY
        assert mujoco.mj_name2id(scene.model, kind, name) >= 0


def test_shadow_semantic_joint_and_tip_mappings_are_explicit_and_correct():
    cfg = load_phase3_config().hand
    assert cfg.finger_joints["thumb"] == ("rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1")
    assert cfg.finger_joints["index"] == ("rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1")
    assert cfg.finger_joints["middle"] == ("rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1")
    assert cfg.finger_joints["ring"] == ("rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1")
    assert cfg.finger_joints["little"] == ("rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1")
    assert cfg.fingertip_bodies == {
        "thumb": "rh_thdistal",
        "index": "rh_ffdistal",
        "middle": "rh_mfdistal",
        "ring": "rh_rfdistal",
        "little": "rh_lfdistal",
    }


def test_shadow_joint_limits_and_actuator_commands_remain_bounded():
    scene = build_shadow_scene()
    assert np.all(scene.model.jnt_limited[:24])
    assert np.all(scene.model.jnt_range[:24, 1] > scene.model.jnt_range[:24, 0])
    controller = VariableImpedanceController(scene)
    controller.reset()
    controller.step(np.full(controller.action_dimension, 100.0))
    assert np.all(scene.data.ctrl <= scene.model.actuator_ctrlrange[:, 1])
    assert np.all(scene.data.ctrl >= scene.model.actuator_ctrlrange[:, 0])
    assert np.all(scene.model.actuator_gainprm[:, 0] <= controller.base_gain + 1e-12)


def test_contact_mapping_and_penetration_work_for_all_five_fingertips():
    for finger in FINGERS:
        scene = build_shadow_scene()
        mujoco.mj_resetData(scene.model, scene.data)
        set_fixture(scene, True)
        mujoco.mj_forward(scene.model, scene.data)
        geom_name = scene.fingertip_geoms[finger][0]
        geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        set_object_pose(scene, scene.data.geom_xpos[geom_id])
        mujoco.mj_forward(scene.model, scene.data)
        contacts = extract_shadow_contacts(scene)
        index = SUPPORT_SURFACES.index(finger)
        assert contacts.contact_flags[index] == 1.0
        assert contacts.normal_forces[index] > 0.0
        assert fingertip_object_penetration(scene, finger) >= 0.0


def test_fingertip_object_penetration_extraction_is_nonzero_in_scripted_pinch():
    trial = run_acquisition_candidate(0, np.asarray([0.379, -0.040, 0.023]))
    penetration = trial["release_state"]["penetration_by_surface_m"]
    assert penetration[0] > 0.0
    assert penetration[1] > 0.0


def test_role_transitions_are_dynamic_not_permanent_occupancy():
    roles = RoleState()
    roles.begin_probe()
    assert roles.fingers["thumb"] == FingerRole.PROBING
    roles.acquisition_contact()
    assert roles.phase == ManipulationPhase.MINIMAL_ACQUIRE
    roles.recruit("middle")
    assert roles.fingers["middle"] == FingerRole.SUPPORTING
    roles.begin_transfer()
    roles.palm_contact()
    assert roles.palm == PalmRole.CONTACT
    roles.palmar_secure_diagnostic()
    roles.begin_release(("thumb",))
    roles.resource_recovered("thumb")
    assert roles.fingers["thumb"] == FingerRole.FREE
    assert roles.phase == ManipulationPhase.RESOURCE_RECOVERED


def test_support_vector_load_fraction_and_resource_mask_preserve_identity():
    scene = build_shadow_scene()
    mujoco.mj_resetData(scene.model, scene.data)
    set_fixture(scene, False)
    set_object_pose(scene, (1.0, 1.0, 1.0))
    roles = RoleState()
    contacts = extract_shadow_contacts(scene)
    assert contacts.support_vector.shape == (6,)
    assert contacts.support_load_fraction.shape == (6,)
    np.testing.assert_array_equal(contacts.support_load_fraction, 0.0)
    resource = compute_resource_snapshot(scene, contacts, roles)
    assert resource.free_finger_mask.shape == (5,)
    assert tuple(resource.fingers) == FINGERS
    assert all(item.local_reachable_workspace.shape == (3, 2) for item in resource.fingers.values())
    assert resource.n_free == int(resource.free_finger_mask.sum())


def test_contact_aware_closer_latches_and_prevents_blind_overclosure():
    scene = build_shadow_scene()
    mujoco.mj_resetData(scene.model, scene.data)
    mujoco.mj_forward(scene.model, scene.data)
    thumb_geom = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, scene.fingertip_geoms["thumb"][0]
    )
    set_object_pose(scene, scene.data.geom_xpos[thumb_geom])
    set_fixture(scene, True)
    mujoco.mj_forward(scene.model, scene.data)
    closer = ContactAwareCloser(scene, force_threshold_n=0.0)
    before = scene.data.ctrl.copy()
    proposed = before.copy()
    proposed[scene.actuator_ids["thumb"]] += 0.5
    limited = closer.limit_target(proposed)
    assert closer.latched["thumb"]
    np.testing.assert_array_equal(limited[scene.actuator_ids["thumb"]], before[scene.actuator_ids["thumb"]])


def test_resource_recovered_event_requires_retention_support_and_motion():
    detector = HandoffEventDetector()
    missing_support = detector.update(
        thumb_index_contact=False,
        middle_contact=False,
        palm_contact=False,
        acquisition_released=True,
        object_retained=True,
        alternate_support=False,
        released_finger_has_motion=True,
    )
    assert HandoffEvent.RESOURCE_RECOVERED not in missing_support
    recovered = detector.update(
        thumb_index_contact=False,
        middle_contact=True,
        palm_contact=True,
        acquisition_released=True,
        object_retained=True,
        alternate_support=True,
        released_finger_has_motion=True,
    )
    assert HandoffEvent.RESOURCE_RECOVERED in recovered


def test_phase3_gymnasium_contract_dimension_and_fixed_seed_determinism():
    env = Phase3ShadowHandEnv()
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=19)
    second, _ = env.reset(seed=19)
    np.testing.assert_array_equal(first, second)
    assert first.shape == env.observation_space.shape
    assert first.size == sum(item.dimension for item in env.observation_metadata if item.actor_available)
    _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert not terminated and not truncated
    assert set(info["privileged_observation"]) == {
        item.name for item in env.observation_metadata if not item.actor_available
    }
    assert set(info["reward_terms"]) == set(env.config.raw["reward_weights"])
    assert all(weight == 0.0 for weight in env.config.raw["reward_weights"].values())
    env.close()


def test_historical_allegro_path_still_builds_without_phase3_configuration():
    cfg = load_configs()
    model, _ = build_scene(cfg)
    assert cfg.hand.model_path.endswith("allegro/right_hand.xml")
    assert model.nu == 16
