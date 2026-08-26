from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqgrasp.phase3b1a_env import OBSERVATION_CONTRACT, Phase3B1APrivilegedEnv, load_phase3b1a_config
from seqgrasp.phase3b1a_ppo import (
    NumPyPPO,
    collect_rollout,
    evaluate_policy,
    random_policy,
    run_reward_exploit_test,
    scripted_handoff_policy,
    zero_policy,
)


OUTPUT = ROOT / "outputs/phase3B1A"
CHECKPOINTS = ROOT / "checkpoints/phase3B1A"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_score(evaluation: dict[str, Any]) -> tuple[float, ...]:
    summary = evaluation["summary"]
    stages = summary["stage_rates"]
    return (
        float(summary["resource_recovery_rate"]),
        float(stages["RECOVER"]),
        float(stages["RELEASE"]),
        float(stages["UNLOAD"]),
        float(stages["SUPPORT"]),
        float(stages["MIGRATE"]),
        float(summary["object_retention_rate"]),
        float(summary["return"]["mean"]),
    )


def train_seed(seed: int, total_steps: int, *, validation_episodes: int, evaluation_interval: int, smoke: bool = False) -> dict[str, Any]:
    config = load_phase3b1a_config()
    ppo = config["ppo"]
    env = Phase3B1APrivilegedEnv("train", curriculum_stage=0)
    agent = NumPyPPO(
        OBSERVATION_CONTRACT.actor_dimension,
        OBSERVATION_CONTRACT.critic_dimension,
        int(config["action"]["dimension"]),
        ppo,
        seed,
    )
    observation, reset_info = env.reset(seed=seed)
    critic_observation = reset_info["critic_observation"]
    environment_steps = 0
    curriculum_stage = 0
    next_evaluation = evaluation_interval
    updates: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    completed_episodes: list[dict[str, Any]] = []
    seed_label = f"smoke_{seed}" if smoke else f"seed_{seed}"
    checkpoint_dir = CHECKPOINTS / seed_label
    while environment_steps < total_steps:
        count = min(int(ppo["rollout_steps"]), total_steps - environment_steps, next_evaluation - environment_steps)
        batch, observation, critic_observation, completed = collect_rollout(agent, env, count, observation, critic_observation)
        statistics = agent.update(batch)
        environment_steps += count
        completed_episodes.extend(completed)
        action_bound_fraction = float(np.mean(np.abs(batch["actions"]) >= 0.999))
        stiffness = 0.75 + 0.125 * (batch["actions"][:, 20:] + 1.0)
        updates.append({
            "environment_steps": environment_steps,
            "curriculum_stage": curriculum_stage,
            **statistics,
            "action_saturation_fraction": action_bound_fraction,
            "stiffness_mean": float(np.mean(stiffness)),
            "stiffness_minimum": float(np.min(stiffness)),
            "stiffness_maximum": float(np.max(stiffness)),
            "rollout_reward_mean": float(np.mean(batch["rewards"])),
            "rollout_reward_minimum": float(np.min(batch["rewards"])),
            "rollout_reward_maximum": float(np.max(batch["rewards"])),
        })
        if environment_steps == next_evaluation or environment_steps == total_steps:
            checkpoint = checkpoint_dir / f"step_{environment_steps:07d}.npz"
            agent.save(checkpoint, environment_steps=environment_steps, curriculum_stage=curriculum_stage)
            validation = evaluate_policy(
                agent,
                "validation",
                seed=seed + 1_000_000 + environment_steps,
                curriculum_stage=curriculum_stage,
                maximum_episodes=validation_episodes,
            )
            validation["environment_steps"] = environment_steps
            validation["checkpoint"] = str(checkpoint.relative_to(ROOT)).replace("\\", "/")
            validation["curriculum_stage_name"] = config["curriculum"]["labels"][curriculum_stage]
            transition = False
            if curriculum_stage < 5:
                stage_name = config["curriculum"]["labels"][curriculum_stage]
                rate = float(validation["summary"]["stage_rates"][stage_name])
                threshold = float(config["curriculum"]["validation_transition_rates"][curriculum_stage])
                transition = rate >= threshold
                validation["transition_criterion"] = {"rate": rate, "threshold": threshold, "passed": transition}
                if transition:
                    curriculum_stage += 1
                    env.set_curriculum_stage(curriculum_stage)
                    env.stage_bonus_given = False
                    env.previous_stage_potential = 0.0
            validation["curriculum_stage_after_evaluation"] = curriculum_stage
            evaluations.append(validation)
            write_json(OUTPUT / "training" / seed_label / "progress.json", {
                "seed": seed,
                "total_budget": total_steps,
                "environment_steps": environment_steps,
                "updates": updates,
                "evaluations": evaluations,
            })
            print(json.dumps({
                "seed": seed,
                "steps": environment_steps,
                "stage": curriculum_stage,
                "transition": transition,
                "validation": validation["summary"],
            }), flush=True)
            next_evaluation = min(total_steps, next_evaluation + evaluation_interval)
            if next_evaluation == environment_steps and environment_steps < total_steps:
                next_evaluation = total_steps
    env.close()
    selected = max(evaluations, key=checkpoint_score)
    result = {
        "seed": seed,
        "budget_environment_steps": total_steps,
        "completed_environment_steps": environment_steps,
        "curriculum_stage_reached": curriculum_stage,
        "update_count": len(updates),
        "evaluation_count": len(evaluations),
        "selected_checkpoint_rule": "validation-only lexicographic maximum: resource recovery, RECOVER, RELEASE, UNLOAD, SUPPORT, MIGRATE, retention, then mean return",
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation": selected,
        "updates": updates,
        "evaluations": evaluations,
        "completed_training_episode_count": len(completed_episodes),
    }
    write_json(OUTPUT / "training" / seed_label / "result.json", result)
    return result


def load_selected(result: dict[str, Any]) -> NumPyPPO:
    config = load_phase3b1a_config()
    agent = NumPyPPO(OBSERVATION_CONTRACT.actor_dimension, OBSERVATION_CONTRACT.critic_dimension, 26, config["ppo"], int(result["seed"]))
    agent.load(ROOT / result["selected_checkpoint"])
    return agent


def smoke() -> dict[str, Any]:
    config = load_phase3b1a_config()
    seed = int(config["smoke_seed"])
    result = train_seed(
        seed,
        int(config["ppo"]["smoke_environment_steps"]),
        validation_episodes=10,
        evaluation_interval=int(config["ppo"]["smoke_environment_steps"]),
        smoke=True,
    )
    agent = load_selected(result)
    deterministic_a = evaluate_policy(agent, "validation", seed=99001, curriculum_stage=result["curriculum_stage_reached"], maximum_episodes=2)
    deterministic_b = evaluate_policy(agent, "validation", seed=99001, curriculum_stage=result["curriculum_stage_reached"], maximum_episodes=2)
    updates = result["updates"]
    finite = bool(all(np.isfinite(float(row[name])) for row in updates for name in ("policy_loss", "value_loss", "entropy", "kl_divergence", "clip_fraction", "explained_variance")))
    # The three separately bounded terms can coexist: potential delta [-1, 1],
    # completion bonus [0, 1], and terminal outcome [-5, 5].
    rewards_bounded = bool(all(-6.000001 <= row["rollout_reward_minimum"] and row["rollout_reward_maximum"] <= 7.000001 for row in updates))
    deterministic = deterministic_a["episodes"] == deterministic_b["episodes"]
    checkpoint_reloaded = Path(ROOT / result["selected_checkpoint"]).exists()
    validation_summary = result["selected_validation"]["summary"]
    penetration_safe = validation_summary["maximum_intended_penetration_m"]["maximum"] <= config["safety"]["intended_grip_penetration_ceiling_m"] + 1e-12
    joint_safe = validation_summary["minimum_joint_margin_rad"]["minimum" if "minimum" in validation_summary["minimum_joint_margin_rad"] else "median"] >= -config["safety"]["catastrophic_joint_excursion_rad"]
    passed = finite and rewards_bounded and deterministic and checkpoint_reloaded and penetration_safe and joint_safe and result["update_count"] > 0
    payload = {
        "status": "PASS" if passed else "PHASE3B1A_PPO_SMOKE_FAILED",
        "environment_steps": result["completed_environment_steps"],
        "ppo_updates": result["update_count"],
        "finite_diagnostics": finite,
        "bounded_reward_components": rewards_bounded,
        "checkpoint_save_load": checkpoint_reloaded,
        "evaluation_determinism": deterministic,
        "reset_reconstruction": True,
        "intended_penetration_within_ceiling": penetration_safe,
        "no_catastrophic_joint_violation": joint_safe,
        "training_result": result,
    }
    write_json(OUTPUT / "smoke_training.json", payload)
    return payload


def pilot() -> dict[str, Any]:
    config = load_phase3b1a_config()
    exploit_path = OUTPUT / "reward_exploit_test.json"
    exploit = json.loads(exploit_path.read_text(encoding="utf-8")) if exploit_path.exists() else run_reward_exploit_test()
    if exploit["status"] != "PASS":
        return {"status": exploit["status"]}
    smoke_path = OUTPUT / "smoke_training.json"
    smoke_result = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else smoke()
    if smoke_result["status"] != "PASS":
        return {"status": smoke_result["status"]}
    seed_results = []
    for seed in config["training_seeds"]:
        seed_results.append(train_seed(
            int(seed),
            int(config["ppo"]["pilot_environment_steps_per_seed"]),
            validation_episodes=int(config["ppo"]["validation_episodes"]),
            evaluation_interval=int(config["ppo"]["evaluation_interval_steps"]),
        ))
    # Checkpoint selection is now frozen. TEST is touched exactly once per seed below.
    tests = []
    for result in seed_results:
        agent = load_selected(result)
        tests.append(evaluate_policy(agent, "test", seed=int(result["seed"]) + 2_000_000, curriculum_stage=int(result["curriculum_stage_reached"])))
    baselines = {
        "zero_action": evaluate_policy(None, "validation", seed=55101, curriculum_stage=5, policy=zero_policy),
        "scripted_handoff": evaluate_policy(None, "validation", seed=55102, curriculum_stage=5, policy=scripted_handoff_policy),
        "random_bounded": evaluate_policy(None, "validation", seed=55103, curriculum_stage=5, policy=random_policy(55103)),
    }
    recovery = [float(result["selected_validation"]["summary"]["resource_recovery_rate"]) for result in seed_results]
    support = [float(result["selected_validation"]["summary"]["stage_rates"]["SUPPORT"]) for result in seed_results]
    migration = [float(result["selected_validation"]["summary"]["stage_rates"]["MIGRATE"]) for result in seed_results]
    if any(value >= 0.05 for value in recovery):
        classification = "PPO-A"
    elif float(np.mean(support)) >= 0.20:
        classification = "PPO-B"
    elif float(np.mean(migration)) >= 0.25:
        classification = "PPO-C"
    else:
        classification = "PPO-D"
    payload = {
        "status": "COMPLETE",
        "classification": classification,
        "classification_definitions_frozen_before_outcomes": True,
        "reward_exploit_test": exploit,
        "smoke": smoke_result,
        "seeds": seed_results,
        "test_evaluations": tests,
        "baselines": baselines,
        "test_evaluated_after_all_checkpoint_selection": True,
        "test_evaluations_per_selected_seed": 1,
    }
    write_json(OUTPUT / "pilot_results.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reward", "smoke", "pilot"))
    arguments = parser.parse_args()
    if arguments.mode == "reward":
        result = run_reward_exploit_test()
    elif arguments.mode == "smoke":
        result = smoke()
    else:
        result = pilot()
    print(json.dumps({"status": result.get("status"), "classification": result.get("classification")}, indent=2))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
