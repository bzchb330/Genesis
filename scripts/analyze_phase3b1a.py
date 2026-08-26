from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqgrasp.phase3b1a_env import CURRICULUM_STAGES, OBSERVATION_CONTRACT, Phase3B1APrivilegedEnv, load_phase3b1a_config
from seqgrasp.phase3b1a_ppo import NumPyPPO, evaluate_policy, scripted_handoff_policy, zero_policy


OUTPUT = ROOT / "outputs/phase3B1A"
FIGURE_DIR = ROOT / "docs/figures/phase3B1A"
VIDEO_DIR = ROOT / "videos/phase3B1A"
COLORS = ("#0072B2", "#D55E00", "#009E73")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_agent(result: dict[str, Any]) -> tuple[NumPyPPO, int]:
    config = load_phase3b1a_config()
    agent = NumPyPPO(OBSERVATION_CONTRACT.actor_dimension, OBSERVATION_CONTRACT.critic_dimension, 26, config["ppo"], int(result["seed"]))
    metadata = agent.load(ROOT / result["selected_checkpoint"])
    return agent, int(metadata["curriculum_stage"])


def selected_validation_diagnostics(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = []
    for result in pilot["seeds"]:
        agent, stage = load_agent(result)
        evaluation = evaluate_policy(
            agent,
            "validation",
            seed=9_331_000 + int(result["seed"]),
            curriculum_stage=stage,
            capture_trajectories=True,
        )
        evaluation["policy_seed"] = result["seed"]
        evaluation["selected_checkpoint"] = result["selected_checkpoint"]
        evaluation["purpose"] = "post-selection validation diagnostic only; did not affect checkpoint selection"
        diagnostics.append(evaluation)
    write_json(OUTPUT / "analysis/selected_validation_diagnostics.json", diagnostics)
    return diagnostics


def style_axes(axis, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / name, format="pdf", bbox_inches="tight")
    plt.close(figure)


def plot_figures(pilot: dict[str, Any], diagnostics: list[dict[str, Any]]) -> list[str]:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    seeds = pilot["seeds"]
    seed_labels = [str(seed["seed"]) for seed in seeds]

    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    for stage_index, (axis, stage) in enumerate(zip(axes.flat, CURRICULUM_STAGES)):
        for color, result in zip(COLORS, seeds):
            evaluations = result["evaluations"]
            axis.plot([row["environment_steps"] for row in evaluations], [row["summary"]["stage_rates"][stage] for row in evaluations], color=color, label=str(result["seed"]))
        style_axes(axis, stage, "environment steps", "validation rate")
        axis.set_ylim(-0.02, 1.02)
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=7)
    save(figure, "curriculum_stage_learning.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for color, result in zip(COLORS, seeds):
        evaluations = result["evaluations"]
        axis.plot([row["environment_steps"] for row in evaluations], [row["summary"]["resource_recovery_rate"] for row in evaluations], marker="o", ms=3, color=color, label=str(result["seed"]))
    style_axes(axis, "Held-out validation resource recovery", "environment steps", "RESOURCE_RECOVERED rate")
    axis.legend(title="seed", frameon=False, ncol=3)
    axis.set_ylim(bottom=0)
    save(figure, "resource_recovery_learning_curve.pdf")

    metrics = ("RETAIN", "MIGRATE", "SUPPORT", "UNLOAD", "RELEASE", "RECOVER")
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(metrics)); width = 0.24
    for offset, (color, result) in enumerate(zip(COLORS, seeds)):
        summary = result["selected_validation"]["summary"]
        axis.bar(x + (offset - 1) * width, [summary["stage_rates"][metric] for metric in metrics], width, color=color, label=str(result["seed"]))
    axis.set_xticks(x, metrics)
    axis.legend(title="seed", frameon=False)
    style_axes(axis, "Frozen validation-selected checkpoint funnel", ylabel="rate")
    save(figure, "validation_success_by_seed.pdf")

    candidate_trajectories = [trajectory for diagnostic in diagnostics for trajectory in diagnostic["trajectories"] if trajectory.get("result", {}).get("resource_recovered")]
    representative = candidate_trajectories[0] if candidate_trajectories else diagnostics[0]["trajectories"][0]
    samples = representative["samples"]
    steps = [sample["step"] for sample in samples]
    support = np.asarray([sample["support_load_fraction"] for sample in samples])
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for index, label in enumerate(("thumb", "index", "middle", "ring", "little", "palm")):
        axis.plot(steps, support[:, index], label=label)
    style_axes(axis, "Representative support-load evolution", "control step", "normal-load fraction")
    axis.legend(frameon=False, ncol=3)
    save(figure, "support_load_evolution.pdf")

    flags = np.asarray([sample["contact_flags"] for sample in samples])
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    image = axis.imshow(flags.T, aspect="auto", interpolation="nearest", origin="lower", extent=(steps[0], steps[-1], -0.5, 5.5), cmap="Blues", vmin=0, vmax=1)
    axis.set_yticks(range(6), ("thumb", "index", "middle", "ring", "little", "palm"))
    style_axes(axis, "Representative finger-role/contact evolution", "control step", "surface")
    figure.colorbar(image, ax=axis, label="contact flag", fraction=0.025)
    save(figure, "finger_role_evolution.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(3); width = 0.34
    axis.bar(x - width / 2, [seed["selected_validation"]["summary"]["thumb_recovery_rate"] for seed in seeds], width, color="#CC79A7", label="thumb")
    axis.bar(x + width / 2, [seed["selected_validation"]["summary"]["index_recovery_rate"] for seed in seeds], width, color="#56B4E9", label="index")
    axis.set_xticks(x, seed_labels); axis.legend(frameon=False)
    style_axes(axis, "Acquisition-finger recovery on validation", "seed", "recovery rate")
    save(figure, "thumb_index_recovery.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    palm = [diagnostic["summary"]["palm_contact_rate"] for diagnostic in diagnostics]
    alternate = [diagnostic["summary"]["alternate_support_rate"] for diagnostic in diagnostics]
    axis.bar(x - width / 2, palm, width, color="#E69F00", label="palm contact")
    axis.bar(x + width / 2, alternate, width, color="#009E73", label="any alternate support")
    axis.set_xticks(x, seed_labels); axis.legend(frameon=False)
    style_axes(axis, "Post-selection validation support diagnostics", "seed", "episode rate")
    save(figure, "palm_contact_learning.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for color, result in zip(COLORS, seeds):
        rows = result["evaluations"]
        axis.plot([row["environment_steps"] for row in rows], [1000 * row["summary"]["maximum_intended_penetration_m"]["maximum"] for row in rows], color=color, label=str(result["seed"]))
    axis.axhline(3.0, color="black", ls="--", lw=1, label="3 mm ceiling")
    style_axes(axis, "Maximum intended-grip penetration", "environment steps", "penetration (mm)")
    axis.legend(frameon=False, ncol=2)
    save(figure, "penetration_during_training.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for color, result in zip(COLORS, seeds):
        rows = result["updates"]
        axis.plot([row["environment_steps"] for row in rows], [row["stiffness_mean"] for row in rows], alpha=0.8, color=color, label=str(result["seed"]))
    axis.axhspan(0.75, 1.0, color="#999999", alpha=0.1)
    style_axes(axis, "Policy stiffness-scale distribution (rollout mean)", "environment steps", "stiffness scale")
    axis.set_ylim(0.74, 1.01); axis.legend(frameon=False, ncol=3)
    save(figure, "stiffness_policy_distribution.pdf")

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for color, result in zip(COLORS, seeds):
        rows = result["updates"]
        axis.plot([row["environment_steps"] for row in rows], [row["action_saturation_fraction"] for row in rows], color=color, label=str(result["seed"]))
    style_axes(axis, "Policy action-bound hits", "environment steps", "coordinate fraction")
    axis.legend(frameon=False, ncol=3)
    save(figure, "action_bound_hits.pdf")

    names = ("zero", "scripted", "random", *seed_labels)
    baseline_values = [pilot["baselines"][name]["summary"]["resource_recovery_rate"] for name in ("zero_action", "scripted_handoff", "random_bounded")]
    ppo_values = [result["selected_validation"]["summary"]["resource_recovery_rate"] for result in seeds]
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    axis.bar(np.arange(6), baseline_values + ppo_values, color=("#777777", "#E69F00", "#999999", *COLORS))
    axis.set_xticks(np.arange(6), names)
    style_axes(axis, "Validation RESOURCE_RECOVERED: baselines vs PPO", ylabel="rate")
    save(figure, "PPO_vs_scripted_baselines.pdf")

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    test_recovery = [row["summary"]["resource_recovery_rate"] for row in pilot["test_evaluations"]]
    test_retention = [row["summary"]["object_retention_rate"] for row in pilot["test_evaluations"]]
    axis.bar(x - width / 2, test_recovery, width, color="#0072B2", label="resource recovery")
    axis.bar(x + width / 2, test_retention, width, color="#009E73", label="final retention")
    axis.set_xticks(x, seed_labels); axis.legend(frameon=False)
    style_axes(axis, "One-time pose-disjoint TEST performance", "seed", "rate")
    save(figure, "pose_disjoint_test_performance.pdf")
    return [str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted(FIGURE_DIR.glob("*.pdf"))]


def render_episode(name: str, agent: NumPyPPO | None, policy: Callable | None, reset_index: int, seed: int, stage: int) -> dict[str, Any]:
    env = Phase3B1APrivilegedEnv("validation", curriculum_stage=stage, render_mode="rgb_array")
    observation, info = env.reset(seed=seed, options={"reset_index": reset_index})
    frames = [env.render()]
    result = {}
    for step in range(1000):
        action = policy(observation, env, step) if policy is not None else agent.act(observation, deterministic=True)[0]  # type: ignore[union-attr]
        observation, _, terminated, truncated, info = env.step(action)
        if step % 5 == 0:
            frames.append(env.render())
        if terminated or truncated:
            result = info.get("episode_metrics", {})
            break
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDEO_DIR / f"{name}.mp4"
    imageio.mimsave(path, frames, fps=100, codec="libx264", quality=7, macro_block_size=16)
    env.close()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "reset_index": reset_index, "seed": seed, "stage": stage, "result": result}


def render_videos(pilot: dict[str, Any], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    diagnostic = max(diagnostics, key=lambda row: row["summary"]["resource_recovery_rate"])
    result = next(row for row in pilot["seeds"] if row["seed"] == diagnostic["policy_seed"])
    agent, stage = load_agent(result)
    trajectories = diagnostic["trajectories"]
    categories = {
        "best_full_resource_recovery": lambda row: row["result"].get("resource_recovered", False),
        "support_achieved_release_failed": lambda row: row["result"].get("stage_ever", [0] * 6)[2] and not row["result"].get("stage_ever", [0] * 6)[4],
        "release_followed_by_object_loss": lambda row: row["result"].get("stage_ever", [0] * 6)[4] and not row["result"].get("resource_recovered", False) and not row["result"].get("object_retained", False),
    }
    for ordinal, (name, predicate) in enumerate(categories.items()):
        match = next((row for row in trajectories if predicate(row)), None)
        if match is not None:
            entries.append(render_episode(name, agent, None, int(match["reset_index"]), 8_000 + ordinal, stage))
        else:
            entries.append({"path": None, "category": name, "status": "not observed; no video fabricated"})
    returns = np.asarray([row["result"]["return"] for row in trajectories])
    median_index = int(np.argmin(np.abs(returns - np.median(returns))))
    entries.append(render_episode("median_ppo_behavior", agent, None, int(trajectories[median_index]["reset_index"]), 8_100, stage))
    entries.append(render_episode("scripted_baseline", None, scripted_handoff_policy, 0, 8_200, 5))
    entries.append(render_episode("zero_action_baseline", None, zero_policy, 0, 8_300, 5))
    write_json(VIDEO_DIR / "manifest.json", {"source_split": "validation only", "test_replayed": False, "videos": entries})
    return entries


def main() -> int:
    pilot = json.loads((OUTPUT / "pilot_results.json").read_text(encoding="utf-8"))
    diagnostics = selected_validation_diagnostics(pilot)
    figures = plot_figures(pilot, diagnostics)
    videos = render_videos(pilot, diagnostics)
    write_json(OUTPUT / "analysis/artifact_manifest.json", {"figures": figures, "videos": videos, "test_replayed": False})
    print(json.dumps({"figures": len(figures), "videos_rendered": sum(row.get("path") is not None for row in videos), "test_replayed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
