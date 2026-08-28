"""Render truthful Phase 3C-0.8 videos from stored qpos, with no dynamics replay."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from seqgrasp.config import ROOT
from seqgrasp.phase3c08 import build_forearm_scene


OUTPUT = ROOT / "outputs/phase3C08"


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0.35, -0.02, 0.02]
    camera.distance = 0.42; camera.azimuth = 135; camera.elevation = -18
    return camera


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(632, 24 + 7 * len(label)), 36), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255)); return np.asarray(image)


def render_stored(row: dict, path: Path, label: str) -> None:
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    series = np.load(row["timeseries_path"], allow_pickle=False); qpos = series["qpos"]; stages = series["stage"]
    renderer = mujoco.Renderer(scene.model, 480, 640); camera = _camera(); path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=None) as writer:
        for index in range(0, len(qpos), 5):
            scene.data.qpos[:] = qpos[index]; mujoco.mj_forward(scene.model, scene.data)
            renderer.update_scene(scene.data, camera=camera)
            writer.append_data(_annotate(renderer.render(), f"{label} | {stages[index]} | stored state {index}"))
    renderer.close()


def main() -> None:
    result = json.loads((OUTPUT / "targeted_dynamics_results.json").read_text(encoding="utf-8")); rows = result["rows"]
    retained = min((row for row in rows if row["mode"] == "F0_STATIC" and not row["sphere_loss"]), key=lambda row: row["closest_pocket_distance_m"])
    best = min(rows, key=lambda row: row["closest_pocket_distance_m"])
    failure = min((row for row in rows if row["sphere_loss"]), key=lambda row: row["closest_pocket_distance_m"])
    choices = (
        (retained, "forearm_reorientation_retaining_sphere.mp4", "forearm reorientation; sphere retained"),
        (best, "best_targeted_pocket_transport.mp4", "best approach; no pocket entry"),
        (failure, "representative_failure.mp4", "representative measured sphere loss"),
    )
    generated = []
    for row, filename, label in choices:
        path = OUTPUT / "videos" / filename; render_stored(row, path, label); generated.append(str(path))
    summary_path = OUTPUT / "phase3c08_summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8")); summary["videos"] = generated
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (ROOT / "docs/PHASE3C08_RESULTS.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Truthful render-only videos\n\n" + "\n".join(f"- `{Path(path).relative_to(ROOT).as_posix()}`" for path in generated) + "\n")
    print(json.dumps({"generated": generated, "dynamics_replays": 0, "successful_entry_video": None}, indent=2))


if __name__ == "__main__":
    main()
