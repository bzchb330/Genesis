"""Render only stored states from the executed Phase 3C-1.0 B03 holds."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from seqgrasp.config import ROOT
from seqgrasp.phase3c08 import build_forearm_scene


OUTPUT = ROOT / "outputs/phase3C10"


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0.35, -0.02, 0.02]
    camera.distance = 0.42; camera.azimuth = 135; camera.elevation = -18
    return camera


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(632, 26 + 7 * len(label)), 38), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255)); return np.asarray(image)


def render_stored(row: dict, destination: Path, label: str, stride: int) -> None:
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    series = np.load(row["timeseries_path"], allow_pickle=False); qpos = series["qpos"]
    renderer = mujoco.Renderer(scene.model, 480, 640); camera = _camera(); destination.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(destination, fps=20, codec="libx264", quality=7, macro_block_size=None) as writer:
        for index in range(0, len(qpos), stride):
            scene.data.qpos[:] = qpos[index]; mujoco.mj_forward(scene.model, scene.data)
            renderer.update_scene(scene.data, camera=camera)
            text = f"{label} | stored step {index + 1} | B03={bool(series['inside_B03'][index])}"
            writer.append_data(_annotate(renderer.render(), text))
    renderer.close()


def main() -> None:
    result = json.loads((OUTPUT / "B03_validation_results.json").read_text(encoding="utf-8"))
    longest = max(result["rows"], key=lambda row: max([int(step) for step, passed in row["survival"].items() if passed] or [0]))
    shortest = min(result["rows"], key=lambda row: max([int(step) for step, passed in row["survival"].items() if passed] or [0]))
    videos = OUTPUT / "videos"
    choices = (
        (longest, videos / "direct_B03_hold_validation.mp4", "executed direct hold; longest-retained B03-C trial", 10),
        (shortest, videos / "B03_dynamic_instability_failure.mp4", "executed B03 dynamic-instability failure", 5),
    )
    generated = []
    for row, destination, label, stride in choices:
        render_stored(row, destination, label, stride); generated.append(str(destination))
    summary_path = OUTPUT / "phase3c10_summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["videos"] = generated
    summary["video_omissions"] = {
        "scripted_receiver_first_transfer": "not executed because B03-C failed the prerequisite target gate",
        "successful_handoff": "no success occurred",
        "representative_support_loss": "no scripted handoff trial was executed",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"generated": generated, "dynamics_replays": 0, "fabricated_success_videos": 0}, indent=2))


if __name__ == "__main__":
    main()
