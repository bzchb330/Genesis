"""Truthful vector figures and replay videos for Phase 3C-0."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .config import ROOT
from .phase3.config import FINGERS
from .phase3.model import build_shadow_scene, set_fixture
from .phase3c0 import Phase3CFingerRole, configured_storage_region


FIGURE_NAMES = (
    "open_hand_initial_state.pdf",
    "minimal_acquisition_with_clear_fingers.pdf",
    "open_transfer_corridor.pdf",
    "old_vs_open_corridor.pdf",
    "palmar_storage_region.pdf",
    "delayed_storage_finger_closure.pdf",
    "object_A_secured_and_acquisition_fingers_free.pdf",
    "wrist_reorientation_feasibility.pdf",
    "gravity_in_palm_frame.pdf",
    "dynamic_storage_aperture.pdf",
    "candidate_insertion_corridors.pdf",
    "aperture_relaxation.pdf",
    "object_B_insertion_sequence.pdf",
    "multi_object_resecure.pdf",
    "multi_object_support_graph.pdf",
    "full_sequential_manipulation_sequence.pdf",
)


def _style(axis, title: str, subtitle: str | None = None) -> None:
    axis.set_title(title, loc="left", fontsize=13, weight="bold", pad=28)
    if subtitle:
        text_method = getattr(axis, "text2D", axis.text)
        text_method(0.0, 1.01, subtitle, transform=axis.transAxes, fontsize=8, color="#4b5563")
    axis.grid(alpha=0.18)


def _save(path: Path, draw: Callable) -> None:
    figure = plt.figure(figsize=(9.2, 5.5), constrained_layout=True)
    draw(figure)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _not_run(figure, title: str, reason: str) -> None:
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.5, 0.65, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.49, "NOT RUN - prerequisite gate not passed", ha="center", va="center",
              fontsize=13, color="#9b1c1c", weight="bold")
    axis.text(0.5, 0.34, reason, ha="center", va="center", fontsize=10, wrap=True, color="#374151")
    axis.text(0.5, 0.14, "No success state or trajectory was synthesized.", ha="center", fontsize=9)


def _role_codes(samples: list[dict]) -> tuple[np.ndarray, list[str]]:
    roles = list(Phase3CFingerRole)
    lookup = {role.value: i for i, role in enumerate(roles)}
    return np.asarray([[lookup[s["roles"][finger]] for s in samples] for finger in FINGERS]), [r.value for r in roles]


def _render_samples(samples: list[dict], maximum_frames: int = 150) -> list[np.ndarray]:
    scene = build_shadow_scene()
    set_fixture(scene, False)
    renderer = mujoco.Renderer(scene.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.34, -0.02, 0.01)
    camera.distance = 0.38
    camera.azimuth = 145
    camera.elevation = -18
    address = scene.model.jnt_qposadr[scene.object_joint_id]
    indices = np.linspace(0, len(samples) - 1, min(maximum_frames, len(samples)), dtype=int)
    frames = []
    for index in indices:
        sample = samples[int(index)]
        scene.data.qpos[:24] = sample["hand_qpos"]
        scene.data.qpos[address:address + 7] = sample["object_qpos"]
        scene.data.qvel[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        renderer.update_scene(scene.data, camera=camera)
        frame = renderer.render().copy()
        frames.append(frame)
    renderer.close()
    return frames


def generate_phase3c0_visuals(
    result_path: Path | None = None,
    figure_dir: Path | None = None,
    video_dir: Path | None = None,
) -> dict:
    result_path = result_path or ROOT / "outputs/phase3C0/phase3c0_results.json"
    figure_dir = figure_dir or ROOT / "docs/figures/phase3C0"
    video_dir = video_dir or ROOT / "outputs/phase3C0/videos"
    figure_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    old = result["C0_A_and_B"]["trials"]["old_early_support"]
    opened = result["C0_A_and_B"]["trials"]["open_corridor"]
    nominal = opened[0]
    samples = nominal["samples"]
    times = np.asarray([s["time_s"] for s in samples])
    role_matrix, role_names = _role_codes(samples)

    def fig1(fig):
        ax = fig.add_subplot(111, projection="3d")
        initial = samples[0]
        object_palm = np.asarray(initial["object_position_palm_m"])
        ax.scatter(*object_palm, s=160, color="#2f855a", label="object A COM")
        ax.scatter(0, 0, 0, s=120, marker="s", color="#374151", label="palm frame origin")
        ax.quiver(0, 0, 0, .05, 0, 0, color="#dc2626"); ax.quiver(0, 0, 0, 0, .05, 0, color="#16a34a"); ax.quiver(0, 0, 0, 0, 0, .05, color="#2563eb")
        ax.set(xlabel="palm x (m)", ylabel="palm y (m)", zlabel="palm z (m)")
        ax.legend(); _style(ax, "OPEN_HAND initial state", "All 24 projected joint coordinates are zero; unused digits have no A contact")

    def fig2(fig):
        ax = fig.add_subplot(111)
        flags = np.asarray([s["contact_flags"][:5] for s in samples])
        for i, finger in enumerate(FINGERS):
            ax.step(times, flags[:, i] + 1.4 * i, where="post", label=finger)
        ax.axvline(nominal["fixture_release_step"] * .002, color="black", ls="--", label="fixture release")
        ax.set(xlabel="time (s)", ylabel="contact flag + identity offset")
        ax.legend(ncol=3); _style(ax, "Minimal acquisition with clear fingers", "Thumb/index acquire; middle/ring/little start clear")

    def fig3(fig):
        ax = fig.add_subplot(111)
        corridor_samples = [s for s in samples if s["stage"] == "TRANSFER_A_TO_PALM"]
        selected = corridor_samples[0] if corridor_samples else samples[-1]
        clearance = np.asarray(selected["corridor"]["clearance_m"])
        ax.plot(np.linspace(0, 1, len(clearance)), clearance * 1000, color="#2563eb", lw=2)
        ax.axhline(0, color="black", lw=1)
        ax.set(xlabel="candidate path fraction", ylabel="conservative clearance (mm)")
        _style(ax, "Open transfer corridor", f"Obstructing links: {', '.join(selected['corridor']['obstructing_links']) or 'none'}")

    def fig4(fig):
        ax = fig.add_subplot(111)
        labels = ["old early support", "open corridor"]
        summaries = [result["C0_A_and_B"]["summary"]["old_early_support"], result["C0_A_and_B"]["summary"]["open_corridor"]]
        x = np.arange(2); width = .25
        ax.bar(x - width, [s["transfer_success_rate"] for s in summaries], width, label="storage entry rate")
        ax.bar(x, [s["secure_storage_successes"] / s["attempts"] for s in summaries], width, label="secure rate")
        ax.bar(x + width, [s["resource_recovery_successes"] / s["attempts"] for s in summaries], width, label="recovery rate")
        ax.set_xticks(x, labels); ax.set_ylim(0, 1.08); ax.set_ylabel("fraction of matched trials")
        ax.legend(); _style(ax, "Old style vs open corridor", "Raw structural outcomes; 6 matched trials per condition")

    def fig5(fig):
        ax = fig.add_subplot(111)
        region = configured_storage_region()
        c, h = np.asarray(region.center_palm_m), np.asarray(region.half_extents_m)
        rectangle = plt.Rectangle((c[0]-h[0], c[2]-h[2]), 2*h[0], 2*h[2], fill=False, lw=2, color="#7c3aed", label="diagnostic region")
        ax.add_patch(rectangle)
        path = np.asarray([s["object_position_palm_m"] for s in samples])
        ax.plot(path[:, 0], path[:, 2], color="#2f855a", label="A COM path")
        ax.scatter(path[0, 0], path[0, 2], marker="o", label="start"); ax.scatter(path[-1, 0], path[-1, 2], marker="x", label="final")
        ax.set(xlabel="palm x (m)", ylabel="palm z (m)"); ax.axis("equal"); ax.legend()
        _style(ax, "Palmar storage region", "Geometric palm-frame volume; not a scalar storage score")

    def fig6(fig):
        ax = fig.add_subplot(111)
        image = ax.imshow(role_matrix, aspect="auto", interpolation="nearest", cmap="tab20")
        ax.set_yticks(range(len(FINGERS)), FINGERS); ax.set_xlabel("logged sample index")
        entry = next((i for i, s in enumerate(samples) if s["step"] >= (nominal["recruitment_step"] or 10**9)), None)
        if entry is not None: ax.axvline(entry, color="white", lw=2, ls="--", label="storage-triggered recruitment")
        _style(ax, "Delayed storage-finger closure", f"Recruitment step: {nominal['recruitment_step']}")

    def fig7(fig):
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(.5,.72,"A secured and acquisition fingers free",ha="center",fontsize=16,weight="bold")
        ax.text(.5,.53,"NOT DEMONSTRATED",ha="center",fontsize=15,color="#9b1c1c",weight="bold")
        ax.text(.5,.35,f"Secure storage: {sum(t['secure_storage'] for t in opened)}/6\nThumb+index recovery with secure A: {sum(t['resource_recovered'] for t in opened)}/6",ha="center",fontsize=11)
        ax.text(.5,.17,"All six A states reached the diagnostic region but were lost during securing/release.",ha="center",fontsize=9)

    def fig8(fig): _not_run(fig, "Stored-A wrist reorientation feasibility", result["C0_C_wrist_feasibility"]["reason"])

    def fig9(fig):
        ax = fig.add_subplot(111)
        g = np.asarray([s["gravity_in_palm_frame"] for s in samples])
        for i, label in enumerate(("palm x", "palm y", "palm z")): ax.plot(times, g[:, i], label=label)
        ax.set(xlabel="time (s)", ylabel="gravity component (m/s^2)"); ax.legend()
        _style(ax, "Gravity in palm frame", "World gravity remained [0, 0, -9.81]; only hand orientation changes components")

    def fig10(fig):
        ax = fig.add_subplot(111)
        widths = np.asarray([s["aperture"]["effective_width_m"] for s in samples])
        heights = np.asarray([s["aperture"]["effective_height_m"] for s in samples])
        ax.plot(times, widths*1000, label="effective width"); ax.plot(times, heights*1000, label="effective height")
        ax.set(xlabel="time (s)", ylabel="raw aperture span (mm)"); ax.legend()
        _style(ax, "Dynamic storage aperture", "Measured from palm and current storage-finger geometry; no quality score")

    def fig11(fig): _not_run(fig, "Candidate insertion corridors", "No secure stored-A state was available for C0-C corridor search.")
    def fig12(fig): _not_run(fig, "Aperture relaxation", "C0-D was gated on secure A and therefore not executed.")
    def fig13(fig): _not_run(fig, "Object B insertion sequence", "C0-E was withheld because the first-object storage mechanism did not validate.")
    def fig14(fig): _not_run(fig, "Multi-object resecure", "No B insertion occurred; A+B resecure was not attempted.")

    def fig15(fig):
        ax = fig.add_subplot(111); ax.axis("off")
        hand_y = np.linspace(.15,.85,6); object_y = [.35,.65]
        for y,name in zip(hand_y,[*FINGERS,"palm"]): ax.text(.15,y,name,ha="center",va="center",bbox=dict(boxstyle="round",fc="#dbeafe"))
        for y,name in zip(object_y,["A","B"]): ax.text(.85,y,name,ha="center",va="center",bbox=dict(boxstyle="round",fc="#dcfce7"))
        ax.text(.5,.9,"Bipartite support representation",ha="center",fontsize=15,weight="bold")
        ax.text(.5,.06,"Architecture validated; no two-object contact edges observed because C0-E was gated.",ha="center",fontsize=9)

    def fig16(fig):
        ax = fig.add_subplot(111); ax.axis("off")
        stages = ["OPEN", "T+I ACQUIRE", "CORRIDOR", "A REGION", "SECURE A", "WRIST", "B INSERT", "RESECURE"]
        states = ["done", "done", "done", "done", "failed", "gated", "gated", "gated"]
        x = np.linspace(.07,.93,len(stages))
        colors = {"done":"#bbf7d0","failed":"#fecaca","gated":"#e5e7eb"}
        for i,(label,state) in enumerate(zip(stages,states)):
            ax.text(x[i],.52,label,ha="center",va="center",rotation=30,bbox=dict(boxstyle="round",fc=colors[state]))
            if i<len(stages)-1: ax.annotate("",xy=(x[i+1]-.035,.52),xytext=(x[i]+.035,.52),arrowprops=dict(arrowstyle="->"))
        ax.text(.5,.84,"Phase 3C-0 sequential manipulation funnel",ha="center",fontsize=15,weight="bold")
        ax.text(.5,.18,"The sequence stops honestly at secure-A failure; later physics stages were not run.",ha="center",fontsize=10)

    drawers = (fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13,fig14,fig15,fig16)
    for name, draw in zip(FIGURE_NAMES, drawers): _save(figure_dir / name, draw)

    # Only actual recorded failure trajectories are rendered. Success-specific
    # video names are omitted because no secure-A or A+B success occurred.
    video_specs = {
        "single_object_open_corridor_transfer.mp4": nominal,
        "old_style_obstruction_failure.mp4": max(old, key=lambda row: row["gross_collision_steps"]),
        "failed_A_loss_example.mp4": max(opened, key=lambda row: row["palmward_progress_m"] * -1.0),
    }
    video_paths = []
    for name, trial in video_specs.items():
        path = video_dir / name
        imageio.mimsave(path, _render_samples(trial["samples"]), fps=25, codec="libx264", quality=7,
                        macro_block_size=16)
        video_paths.append(path)
    manifest = {"figures": [str(figure_dir / name) for name in FIGURE_NAMES],
                "videos": [str(path) for path in video_paths],
                "omitted_success_videos": ["successful_delayed_storage_secure", "wrist_reorientation_while_retaining_A",
                                             "aperture_relaxation", "B_insertion_attempt", "successful_A_B_resecure",
                                             "failed_B_collision"],
                "omission_reason": "secure-A prerequisite gate did not pass; no later trajectories exist"}
    (result_path.parent / "visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
