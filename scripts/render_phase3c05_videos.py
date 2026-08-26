"""Render truthful Phase 3C-0.5 state replays from deterministic conditions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.model import build_shadow_scene, set_fixture
from seqgrasp.phase3c05 import (
    CaptureStrategy,
    capture_trial,
    load_frozen_states,
    release_trial,
)


def _renderer() -> tuple[Any, mujoco.Renderer, mujoco.MjvCamera]:
    scene = build_shadow_scene()
    set_fixture(scene, False)
    renderer = mujoco.Renderer(scene.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.34, -0.02, 0.01)
    camera.distance = 0.38
    camera.azimuth = 145
    camera.elevation = -18
    return scene, renderer, camera


def _render_qpos(samples: list[dict], path: Path, maximum_frames: int = 180) -> None:
    scene, renderer, camera = _renderer()
    indices = np.unique(np.linspace(0, len(samples) - 1, min(maximum_frames, len(samples)), dtype=int))
    frames = []
    for index in indices:
        scene.data.qpos[:] = samples[int(index)]["qpos"]
        scene.data.qvel[:] = samples[int(index)].get("qvel", np.zeros(scene.model.nv))
        mujoco.mj_forward(scene.model, scene.data)
        renderer.update_scene(scene.data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(path, frames, fps=30, codec="libx264", quality=7, macro_block_size=16)


def _render_phase3c0(samples: list[dict], path: Path, maximum_frames: int = 180) -> None:
    scene, renderer, camera = _renderer()
    address = scene.model.jnt_qposadr[scene.object_joint_id]
    indices = np.unique(np.linspace(0, len(samples) - 1, min(maximum_frames, len(samples)), dtype=int))
    frames = []
    for index in indices:
        sample = samples[int(index)]
        scene.data.qpos[:24] = sample["hand_qpos"]
        scene.data.qpos[address:address + 7] = sample["object_qpos"]
        scene.data.qvel[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        renderer.update_scene(scene.data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(path, frames, fps=30, codec="libx264", quality=7, macro_block_size=16)


def _condition_row(rows: list[dict], strategy: str, *, success: bool,
                   command: tuple[float, float] | None = None) -> dict:
    choices = [row for row in rows if row["strategy"] == strategy
               and bool(row["coordinated_capture"]) == success]
    if command is not None:
        choices = [row for row in choices if tuple(row["wrist_delta_command_deg"]) == command]
    if not choices:
        raise RuntimeError(f"no stored representative for {strategy}, success={success}, command={command}")
    return choices[0]


def main() -> None:
    output = ROOT / "outputs/phase3C05"
    video_dir = output / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    formal = json.loads((output / "phase3c05_results.json").read_text(encoding="utf-8"))
    states = {state.state_id: state for state in load_frozen_states()}
    rows = formal["capture_trials"]
    records: list[dict[str, Any]] = []

    phase3c0 = json.loads((ROOT / "outputs/phase3C0/phase3c0_results.json").read_text(encoding="utf-8"))
    loss = phase3c0["C0_A_and_B"]["trials"]["open_corridor"][0]
    path = video_dir / "original_phase3C0_loss.mp4"
    _render_phase3c0(loss["samples"], path)
    records.append({"path": str(path), "source": "recorded Phase 3C-0 open-corridor trial 1",
                    "outcome": "A_LOST", "fabricated": False})

    capture_specs = (
        ("simultaneous_storage_finger_capture.mp4", _condition_row(
            rows, CaptureStrategy.SIMULTANEOUS.value, success=True), CaptureStrategy.SIMULTANEOUS),
        ("wrist_assisted_capture.mp4", _condition_row(
            rows, CaptureStrategy.WRIST.value, success=True, command=(5.0, -5.0)), CaptureStrategy.WRIST),
        ("failed_wrist_orientation.mp4", _condition_row(
            rows, CaptureStrategy.WRIST.value, success=False, command=(-5.0, 5.0)), CaptureStrategy.WRIST),
    )
    for name, stored, strategy in capture_specs:
        trial = capture_trial(
            build_shadow_scene(), states[stored["state_id"]], tuple(stored["subset"]), strategy,
            wrist_delta_deg=tuple(stored["wrist_delta_command_deg"]), record_state=True,
        )
        if bool(trial["A_retained"] and trial["persistence_first_reached"]["0.10/25"] is not None) != bool(stored["coordinated_capture"]):
            raise RuntimeError(f"representative capture replay mismatch for {stored['state_id']}")
        path = video_dir / name
        _render_qpos(trial["samples"], path)
        records.append({
            "path": str(path), "source": "deterministic replay of frozen formal condition",
            "state_id": stored["state_id"], "strategy": stored["strategy"],
            "subset": stored["subset"], "wrist_delta_command_deg": stored["wrist_delta_command_deg"],
            "outcome": "COORDINATED_CAPTURE_ESTABLISHED" if stored["coordinated_capture"] else "A_LOST",
            "fabricated": False,
        })

    release_specs = (
        ("successful_one_finger_recovery.mp4", "C05_STATE_00021", "thumb", 100),
        ("failed_early_release.mp4", "C05_STATE_00021", "index", 25),
        ("representative_CC-A_success.mp4", "C05_STATE_00006", "thumb", 25),
    )
    for name, state_id, finger, ramp in release_specs:
        scene = build_shadow_scene()
        capture = capture_trial(
            scene, states[state_id], ("middle", "ring", "little"),
            CaptureStrategy.WRIST_LOAD_TRANSFER, wrist_delta_deg=(-5.0, -5.0), record_state=True,
        )
        release = release_trial(scene, capture, finger, ramp, record_state=True)
        path = video_dir / name
        _render_qpos(capture["samples"] + release["samples"], path)
        records.append({
            "path": str(path), "source": "deterministic replay of frozen load-transfer and release condition",
            "state_id": state_id, "strategy": CaptureStrategy.WRIST_LOAD_TRANSFER.value,
            "subset": ["middle", "ring", "little"], "wrist_delta_command_deg": [-5.0, -5.0],
            "released_finger": finger, "ramp_steps": ramp,
            "outcome": "ONE_RESOURCE_RECOVERED" if release["one_resource_recovered"] else "A_LOST",
            "post_release_steps": 1000, "fabricated": False,
        })

    manifest = {
        "videos": records,
        "physics_changed": False,
        "success_videos_are_actual_deterministic_replays": True,
    }
    (output / "video_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
