#!/usr/bin/env python
from __future__ import annotations

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.object_scale import compiled_object_geometry
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.scene_builder import build_scene


def _render(scene_filename: str, focus: str, hand_view: bool) -> tuple[np.ndarray, dict]:
    cfg = load_configs(scene_filename=scene_filename)
    model, data = build_scene(cfg)
    other = "object_b" if focus == "object_a" else "object_a"
    other_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{other}_free")
    other_qpos = model.jnt_qposadr[other_joint]
    data.qpos[other_qpos:other_qpos + 3] = (2.0, 2.0, 2.0)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if hand_view:
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
        camera.lookat[:] = 0.55 * data.xpos[palm] + 0.45 * data.xpos[body]
        camera.distance, camera.azimuth, camera.elevation = 0.34, 140, -22
    else:
        camera.lookat[:] = data.xpos[body]
        camera.distance, camera.azimuth, camera.elevation = 0.16, 135, -18
    renderer = mujoco.Renderer(model, height=420, width=560)
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy()
    renderer.close()
    return image, compiled_object_geometry(model, focus)


def main() -> int:
    phase2s, _ = load_phase2s_config()
    columns = (("scene_two_object.yaml", "LARGE - Phase 2R"), (phase2s.scene_filename, "HALF-SCALE - Phase 2S"))
    rows = (("object_a", False, "Default object A"), ("object_b", False, "Default object B"), ("object_a", True, "Allegro hand + A"), ("object_b", True, "Allegro hand + B"))
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0))
    for row_index, (focus, hand_view, row_label) in enumerate(rows):
        for column_index, (scene_filename, column_label) in enumerate(columns):
            image, geometry = _render(scene_filename, focus, hand_view)
            ax = axes[row_index, column_index]
            ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
            dimensions = " x ".join(f"{value:.3f}" for value in geometry["physical_dimensions_m"])
            ax.set_title(f"{column_label}\n{row_label}", fontsize=9, weight="bold")
            ax.text(
                .02, .02, f"physical xyz: {dimensions} m\nmass: {geometry['mass_kg']:.2f} kg",
                transform=ax.transAxes, fontsize=7, color="white",
                bbox={"facecolor": "black", "alpha": .65, "pad": 3},
            )
    fig.suptitle("Phase 2S object-scale compilation check", fontsize=14, weight="bold")
    fig.text(.5, .008, "Same camera within each row; only object geometry and geometry-derived resting height differ.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    output = ROOT / "docs" / "figures" / "phase2S"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "object_scale_check.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
