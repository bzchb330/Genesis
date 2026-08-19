#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.resource_components import FINGER_ORDER, reconstruct_grasp
from seqgrasp.phase2s_config import load_phase2s_config


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _largest(group):
    paths = list((ROOT / "outputs" / "phase2S" / f"{group}_states").rglob("accepted_states.jsonl"))
    if not paths:
        raise FileNotFoundError(group)
    return max((_jsonl(path) for path in paths), key=len)


def _render(row, scene_cfg, azimuth, elevation):
    enriched = {
        **row,
        "initial_palm_position_m": row.get("initial_palm_position_m", list(scene_cfg.hand.mount_pos)),
        "initial_palm_quaternion": row.get("initial_palm_quaternion", list(scene_cfg.hand.mount_quat)),
    }
    _, model, data, _ = reconstruct_grasp(enriched, scene_cfg)
    renderer = mujoco.Renderer(model, height=420, width=520)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    camera.lookat[:] = data.xpos[object_id]
    camera.distance, camera.azimuth, camera.elevation = 0.26, azimuth, elevation
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy()
    renderer.close()
    return image


def _annotate(ax, row, image, title):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2)
    ax.scatter(
        [image.shape[1] / 2], [image.shape[0] / 2], marker="o",
        facecolors="none", edgecolors="#D55E00", s=38, linewidths=1.2,
    )
    occupied = [finger for finger, flag in zip(FINGER_ORDER, row["occupied_finger_mask"]) if flag]
    free = [finger for finger, flag in zip(FINGER_ORDER, row["occupied_finger_mask"]) if not flag]
    ax.text(
        .02, .02, f"occupied: {','.join(occupied)}\nfree: {','.join(free)}",
        fontsize=5.5, color="white", bbox={"facecolor": "black", "alpha": .55, "pad": 2},
        transform=ax.transAxes,
    )


def main() -> int:
    phase2s, _ = load_phase2s_config()
    scene_cfg = load_configs(scene_filename=phase2s.scene_filename)
    selected = {}
    for group in ("fingertip", "palmar"):
        rows = sorted(_largest(group), key=lambda row: (row["COM_to_palm_origin_distance_m"], row["grasp_state_id"]))
        indices = np.linspace(0, len(rows) - 1, 10).round().astype(int)
        selected[group.upper()] = [rows[index] for index in indices]
    figures = ROOT / "docs" / "figures" / "phase2S"
    figures.mkdir(parents=True, exist_ok=True)
    diagnostics = ROOT / phase2s.output_dir / "diagnostics" / "rendered_states"
    diagnostics.mkdir(parents=True, exist_ok=True)
    views = ((90, 0, "side view"), (180, 0, "palm-normal view"), (140, -22, "3-D view"))
    with PdfPages(figures / "grasp_state_examples.pdf") as pdf:
        for group, rows in selected.items():
            for page in range(2):
                fig, axes = plt.subplots(5, 3, figsize=(10.5, 13.0))
                for row_index, row in enumerate(rows[page * 5:(page + 1) * 5]):
                    for column, (azimuth, elevation, label) in enumerate(views):
                        _annotate(axes[row_index, column], row, _render(row, scene_cfg, azimuth, elevation), label)
                        if column == 0:
                            axes[row_index, column].set_ylabel(row["grasp_state_id"].replace("phase2S_", ""), fontsize=6)
                fig.suptitle(f"Phase 2S {group} half-scale endpoint states (page {page + 1}/2)")
                fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.97))
                pdf.savefig(fig)
                plt.close(fig)
    for group, filename in (("FINGERTIP", "main_fingertip.png"), ("PALMAR", "main_palmar.png")):
        row = selected[group][len(selected[group]) // 2]
        plt.imsave(diagnostics / filename, _render(row, scene_cfg, 140, -22))
    manifest = {group: [row["grasp_state_id"] for row in rows] for group, rows in selected.items()}
    (diagnostics / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
