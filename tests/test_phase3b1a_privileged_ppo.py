from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from seqgrasp.config import ROOT
from seqgrasp.phase3.env import load_keyframe_qpos
from seqgrasp.phase3b1a import project_feasible_hand_qpos
from seqgrasp.phase3b1a_env import OBSERVATION_CONTRACT, Phase3B1APrivilegedEnv, load_phase3b1a_config
from seqgrasp.phase3b1a_ppo import NumPyPPO, collect_rollout, phase3a_structured_return, run_reward_exploit_test


def test_pregrasp_projection_is_minimum_l2_and_feasible() -> None:
    requested = load_keyframe_qpos("pre grasp")
    result = project_feasible_hand_qpos(requested)
    assert result.optimizer_success
    assert result.original_violating_joints == ("rh_FFJ4", "rh_FFJ1", "rh_LFJ4", "rh_LFJ2", "rh_LFJ1")
    assert result.projected_minimum_joint_margin_rad >= -1e-10
    assert result.l2_magnitude_rad < 0.02
    assert all(min(margins) >= -1e-10 for margins in result.tendon_constraint_margins_rad.values())


def test_frozen_reset_split_is_valid_and_isolated() -> None:
    payload = json.loads((ROOT / "outputs/phase3B1A/resets/split.json").read_text(encoding="utf-8"))
    assert (len(payload["train"]), len(payload["validation"]), len(payload["test"])) == (300, 100, 100)
    assert payload["zero_id_overlap"] and payload["zero_state_hash_overlap"]
    assert not (set(payload["train"]) & set(payload["validation"]))
    assert not (set(payload["train"]) & set(payload["test"]))
    assert not (set(payload["validation"]) & set(payload["test"]))
    for paths in payload["state_paths"].values():
        assert all((ROOT / path).exists() for path in paths)


def test_privileged_observations_and_action_contract() -> None:
    env = Phase3B1APrivilegedEnv("train")
    observation, info = env.reset(seed=33101, options={"reset_index": 0})
    assert observation.shape == (OBSERVATION_CONTRACT.actor_dimension,) == (131,)
    assert info["critic_observation"].shape == (OBSERVATION_CONTRACT.critic_dimension,) == (140,)
    assert np.isfinite(observation).all() and np.isfinite(info["critic_observation"]).all()
    assert env.action_space.shape == (26,)
    _, reward, _, _, step_info = env.step(np.full(26, 2.0))
    assert np.isfinite(reward)
    assert step_info["action_bound_hits"] == 26
    assert np.all(step_info["stiffness_scales"] >= 0.75)
    assert np.all(step_info["stiffness_scales"] <= 1.0)
    assert np.max(np.abs(env.desired - env.data.ctrl)) == 0.0
    env.close()


def test_target_rate_bound_and_reset_determinism() -> None:
    config = load_phase3b1a_config()
    env = Phase3B1APrivilegedEnv("train")
    first, first_info = env.reset(seed=90210)
    desired = env.desired.copy()
    env.step(np.r_[np.ones(20), np.zeros(6)])
    change = np.abs(env.desired - desired)
    assert change.max() <= config["action"]["target_rate_cap_rad_per_control_step"] + 1e-12
    second, second_info = env.reset(seed=90210)
    assert np.array_equal(first, second)
    assert np.array_equal(first_info["critic_observation"], second_info["critic_observation"])
    assert first_info["candidate_id"] == second_info["candidate_id"]
    env.close()


def test_gymnasium_checker_and_episode_termination() -> None:
    env = Phase3B1APrivilegedEnv("train")
    check_env(env, skip_render_check=True)
    env.reset(seed=4, options={"reset_index": 0})
    terminated = truncated = False
    for _ in range(1000):
        _, _, terminated, truncated, _ = env.step(np.zeros(26))
        if terminated or truncated:
            break
    assert terminated or truncated
    env.close()


def test_curriculum_and_reward_gating() -> None:
    env = Phase3B1APrivilegedEnv("train", curriculum_stage=4)
    env.reset(seed=10, options={"reset_index": 0})
    state = env._state_features()
    assert state["potentials"][4] == 0.0  # no release credit before alternate support
    env.set_curriculum_stage(5)
    assert env.curriculum_stage == 5
    env.close()


def test_checkpoint_roundtrip_and_short_rollout_repeatability(tmp_path: Path) -> None:
    config = load_phase3b1a_config()
    first = NumPyPPO(131, 140, 26, config["ppo"], 100)
    path = first.save(tmp_path / "checkpoint.npz", environment_steps=12, curriculum_stage=2)
    second = NumPyPPO(131, 140, 26, config["ppo"], 999)
    metadata = second.load(path)
    observation = np.linspace(-1.0, 1.0, 131)
    assert metadata == {"environment_steps": 12, "curriculum_stage": 2}
    assert np.array_equal(first.act(observation, deterministic=True)[0], second.act(observation, deterministic=True)[0])

    def short(seed: int) -> tuple[np.ndarray, np.ndarray]:
        env = Phase3B1APrivilegedEnv("train")
        actor, info = env.reset(seed=seed)
        agent = NumPyPPO(131, 140, 26, config["ppo"], seed)
        batch, _, _, _ = collect_rollout(agent, env, 8, actor, info["critic_observation"])
        env.close()
        return batch["actions"], batch["rewards"]
    action_a, reward_a = short(12345)
    action_b, reward_b = short(12345)
    assert np.array_equal(action_a, action_b)
    assert np.array_equal(reward_a, reward_b)


def test_phase3a_reward_is_structured_resource_recovery() -> None:
    result = phase3a_structured_return()
    assert result["resource_recovered"]
    assert result["return"] > 10.0


def test_pathological_reward_gate_passes_from_recorded_audit() -> None:
    path = ROOT / "outputs/phase3B1A/reward_exploit_test.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else run_reward_exploit_test(episodes_per_policy=1)
    assert result["status"] == "PASS"
    assert result["return_margin"] >= 2.0
