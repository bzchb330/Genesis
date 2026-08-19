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


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _largest(group: str) -> list[dict]:
    paths = list((ROOT / "outputs" / "phase2R" / f"{group}_states").rglob("accepted_states.jsonl"))
    if not paths:
        raise FileNotFoundError(group)
    return max((_jsonl(path) for path in paths), key=len)


def _reconstruct(row: dict):
    cfg = load_configs()
    enriched = {
        **row,
        "initial_palm_position_m": row.get("initial_palm_position_m", list(cfg.hand.mount_pos)),
        "initial_palm_quaternion": row.get("initial_palm_quaternion", list(cfg.hand.mount_quat)),
    }
    return reconstruct_grasp(enriched)


def _render(row: dict, azimuth: float, elevation: float) -> np.ndarray:
    _, model, data, _ = _reconstruct(row)
    renderer = mujoco.Renderer(model, height=420, width=520)
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    camera.lookat[:] = data.xpos[object_id]
    camera.distance, camera.azimuth, camera.elevation = 0.30, azimuth, elevation
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy(); renderer.close()
    return image


def _annotate(ax, row: dict, image: np.ndarray, title: str):
    ax.imshow(image); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8, pad=2)
    # The camera targets A COM, so the central marker is a rendered A-COM overlay.
    ax.scatter([image.shape[1] / 2], [image.shape[0] / 2], marker="o", facecolors="none", edgecolors="#D55E00", s=38, linewidths=1.2)
    ax.text(.51, .47, "A COM", color="#D55E00", fontsize=6, transform=ax.transAxes)
    ax.annotate("palm +x", xy=(.19, .86), xytext=(.05, .86), arrowprops={"arrowstyle": "->", "color": "#009E73"}, color="#009E73", fontsize=6, xycoords=ax.transAxes)
    occupied = [finger for finger, flag in zip(FINGER_ORDER, row["occupied_finger_mask"]) if flag]
    free = [finger for finger, flag in zip(FINGER_ORDER, row["occupied_finger_mask"]) if not flag]
    ax.text(.02, .02, f"occupied: {','.join(occupied)}\nfree: {','.join(free)}", fontsize=5.5, color="white", bbox={"facecolor": "black", "alpha": .55, "pad": 2}, transform=ax.transAxes)


def main() -> int:
    selected = {}
    for group in ("fingertip", "palmar"):
        rows = sorted(_largest(group), key=lambda row: (row["COM_to_palm_origin_distance_m"], row["grasp_state_id"]))
        indices = np.linspace(0, len(rows) - 1, 10).round().astype(int)
        selected[group.upper()] = [rows[index] for index in indices]
    figures = ROOT / "docs" / "figures" / "phase2R"; figures.mkdir(parents=True, exist_ok=True)
    diagnostics = ROOT / "outputs" / "phase2R" / "diagnostics" / "rendered_states"; diagnostics.mkdir(parents=True, exist_ok=True)
    views = ((90, 0, "side view"), (180, 0, "palm-normal view"), (140, -22, "3-D view"))
    with PdfPages(figures / "grasp_state_examples.pdf") as pdf:
        for group, rows in selected.items():
            for page in range(2):
                fig, axes = plt.subplots(5, 3, figsize=(10.5, 13.0))
                for row_index, row in enumerate(rows[page * 5:(page + 1) * 5]):
                    for column, (azimuth, elevation, label) in enumerate(views):
                        _annotate(axes[row_index, column], row, _render(row, azimuth, elevation), label)
                        if column == 0:
                            short_id = row["grasp_state_id"].replace("phase2R_fingertip_", "").replace("phase2R_palmar_", "")
                            axes[row_index, column].set_ylabel(short_id, fontsize=6.5, labelpad=3)
                fig.suptitle(f"Phase 2R {group} endpoint states (page {page + 1}/2)")
                fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.97)); pdf.savefig(fig); plt.close(fig)
    for group, filename in (("FINGERTIP", "main_fingertip.png"), ("PALMAR", "main_palmar.png")):
        row = selected[group][len(selected[group]) // 2]
        plt.imsave(diagnostics / filename, _render(row, 140, -22))
    manifest = {
        group: [row["grasp_state_id"] for row in rows] for group, rows in selected.items()
    }
    (diagnostics / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
