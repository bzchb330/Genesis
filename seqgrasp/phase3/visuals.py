from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ..config import ROOT
from .config import FINGERS, SUPPORT_SURFACES
from .model import build_shadow_scene, set_fixture
from .roles import FingerRole


FIGURE_NAMES = (
    "shadow_hand_initial_pose.pdf",
    "thumb_index_probe.pdf",
    "minimal_acquisition.pdf",
    "middle_finger_recruitment.pdf",
    "contact_handoff_sequence.pdf",
    "palmar_support_sequence.pdf",
    "acquisition_finger_release.pdf",
    "finger_role_timeline.pdf",
    "support_load_timeline.pdf",
    "free_finger_resource_timeline.pdf",
)


def _render_samples(samples: list[dict]) -> list[np.ndarray]:
    scene = build_shadow_scene()
    set_fixture(scene, False)
    renderer = mujoco.Renderer(
        scene.model,
        height=int(scene.config.raw["render_height"]),
        width=int(scene.config.raw["render_width"]),
    )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.34, -0.02, 0.01)
    camera.distance = 0.36
    camera.azimuth = 145
    camera.elevation = -18
    object_qpos = scene.model.jnt_qposadr[scene.object_joint_id]
    frames = []
    for sample in samples:
        scene.data.qpos[:24] = sample["hand_qpos"]
        scene.data.qpos[object_qpos : object_qpos + 3] = sample["object_position"]
        scene.data.qpos[object_qpos + 3 : object_qpos + 7] = sample["object_quaternion"]
        scene.data.qvel[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        renderer.update_scene(scene.data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def _pick(samples: list[dict], count: int) -> list[dict]:
    if len(samples) <= count:
        return samples
    return [samples[index] for index in np.linspace(0, len(samples) - 1, count, dtype=int)]


def _save_frame_grid(path: Path, samples: list[dict], title: str) -> None:
    samples = _pick(samples, 6)
    frames = _render_samples(samples)
    columns = min(3, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, frame, sample in zip(axes.ravel(), frames, samples):
        axis.imshow(frame)
        axis.set_title(f"{sample['stage']}\nt={sample['time_s']:.3f} s")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _role_matrix(samples: list[dict]) -> np.ndarray:
    values = np.full((len(samples), len(FINGERS)), int(FingerRole.FREE), dtype=int)
    for row, sample in enumerate(samples):
        stage = sample["stage"]
        if stage in {"CONTACT_AWARE_CLOSE", "FIXTURE_SETTLE", "FIXTURE_RELEASE", "MINIMAL_UNSUPPORTED_HOLD"}:
            values[row, 0:2] = int(FingerRole.ACQUIRING)
        elif stage == "MIDDLE_FIRST_DYNAMIC_TRANSFER":
            values[row, 0:2] = int(FingerRole.TRANSFERRING)
            values[row, 2] = int(FingerRole.SUPPORTING)
        elif stage == "RING_LITTLE_SUPPORT":
            values[row, 0:2] = int(FingerRole.TRANSFERRING)
            values[row, 2:] = int(FingerRole.SUPPORTING)
        elif stage == "ACQUISITION_FINGER_RELEASE":
            values[row, 0] = int(FingerRole.RELEASING)
            values[row, 1] = int(FingerRole.ACQUIRING)
            values[row, 2:] = int(FingerRole.SUPPORTING)
        elif stage in {"POST_RELEASE_RETENTION", "FINAL"}:
            values[row, 0] = int(FingerRole.FREE)
            values[row, 1] = int(FingerRole.SUPPORTING)
            values[row, 2:] = int(FingerRole.SUPPORTING)
    return values


def generate_phase3a_visuals(
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
) -> dict:
    output_dir = output_dir or ROOT / "outputs/phase3A"
    figure_dir = figure_dir or ROOT / "docs/figures/phase3A"
    video_dir = output_dir / "videos"
    figure_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((output_dir / "acquisition_and_recruitment.json").read_text(encoding="utf-8"))
    handoff = json.loads((output_dir / "contact_handoff.json").read_text(encoding="utf-8"))
    successful = next(
        trial for trial in cohort["minimal_trials"] if trial["classification"] == "THUMB_INDEX_SUCCESS"
    )
    recruited = cohort["middle_recruitment_trials"][0]
    handoff_samples = [handoff["fixture_release_state"], *handoff["samples"]]

    _save_frame_grid(figure_dir / FIGURE_NAMES[0], [successful["initial_state"]], "Official right Shadow Hand E3M5")
    probe_samples = [successful["initial_state"], *[s for s in successful["samples"] if s["stage"] == "CONTACT_AWARE_CLOSE"]]
    _save_frame_grid(figure_dir / FIGURE_NAMES[1], probe_samples, "Independent thumb/index contact-aware probe")
    minimal_samples = [s for s in successful["samples"] if s["stage"] in {"FIXTURE_SETTLE", "UNSUPPORTED_HOLD"}]
    _save_frame_grid(figure_dir / FIGURE_NAMES[2], minimal_samples, "Minimal thumb-index acquisition")
    middle_samples = [s for s in recruited["samples"] if s["stage"] == "MIDDLE_RECRUITMENT"]
    _save_frame_grid(figure_dir / FIGURE_NAMES[3], middle_samples, "Middle finger recruitment after insufficient minimal hold")
    _save_frame_grid(figure_dir / FIGURE_NAMES[4], handoff_samples, "Dynamics-only contact handoff sequence")
    palm_samples = [s for s in handoff_samples if s["normal_forces_n"][5] > 0.0]
    _save_frame_grid(figure_dir / FIGURE_NAMES[5], palm_samples, "Dynamically established palmar support")
    release_samples = [s for s in handoff_samples if s["stage"] in {"ACQUISITION_FINGER_RELEASE", "POST_RELEASE_RETENTION"}]
    _save_frame_grid(figure_dir / FIGURE_NAMES[6], release_samples, "Thumb release and retained object")

    times = np.asarray([sample["time_s"] for sample in handoff_samples])
    roles = _role_matrix(handoff_samples)
    figure, axes = plt.subplots(len(FINGERS), 1, figsize=(10, 8), sharex=True)
    for index, (axis, finger) in enumerate(zip(axes, FINGERS)):
        axis.step(times, roles[:, index], where="post")
        axis.set_yticks(range(len(FingerRole)), [role.name for role in FingerRole], fontsize=7)
        axis.set_ylabel(finger)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Time-varying semantic finger roles")
    figure.tight_layout()
    figure.savefig(figure_dir / FIGURE_NAMES[7], bbox_inches="tight")
    plt.close(figure)

    loads = np.asarray([sample["support_load_fraction"] for sample in handoff_samples])
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, surface in enumerate(SUPPORT_SURFACES):
        axis.plot(times, loads[:, index], label=surface)
    axis.set(xlabel="time (s)", ylabel="normal-force load fraction", ylim=(-0.02, 1.02))
    axis.legend(ncol=3)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / FIGURE_NAMES[8], bbox_inches="tight")
    plt.close(figure)

    flags = np.asarray([sample["contact_flags"][:5] for sample in handoff_samples], dtype=bool)
    free = (roles == int(FingerRole.FREE)) & ~flags
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].step(times, free.sum(axis=1), where="post")
    axes[0].set_ylabel("N_free")
    for index, finger in enumerate(FINGERS):
        axes[1].step(times, free[:, index] + index * 1.2, where="post", label=finger)
    axes[1].set(xlabel="time (s)", ylabel="identity-preserving free mask")
    axes[1].legend(ncol=5)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / FIGURE_NAMES[9], bbox_inches="tight")
    plt.close(figure)

    minimal_video_samples = _pick([successful["initial_state"], *successful["samples"]], 100)
    handoff_video_samples = _pick(handoff_samples, 160)
    video_paths = [video_dir / "minimal_acquisition.mp4", video_dir / "contact_handoff.mp4"]
    for path, samples in zip(video_paths, (minimal_video_samples, handoff_video_samples)):
        imageio.mimsave(path, _render_samples(samples), fps=25, codec="libx264", quality=8)
    return {
        "figures": [str(figure_dir / name) for name in FIGURE_NAMES],
        "videos": [str(path) for path in video_paths],
    }
