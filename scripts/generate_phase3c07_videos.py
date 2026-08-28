"""Render only Phase 3C-0.7 behaviors that actually occurred."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from seqgrasp.config import ROOT
from seqgrasp.phase3c07 import (
    PreshapeCondition,
    TransportStrategy,
    build_c07_scene,
    load_acquisition_states,
    plan_transport,
    pocket_volume_from_audit,
    restore_acquisition_state,
    run_transport_trial,
)


OUTPUT = ROOT / "outputs/phase3C07"


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0.35, -0.02, 0.02]
    camera.distance = 0.42; camera.azimuth = 135; camera.elevation = -18
    return camera


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(632, 20 + 7 * len(label)), 35), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255))
    return np.asarray(image)


def _write(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


def acquisition_video(state) -> list[np.ndarray]:
    scene = build_c07_scene(); restore_acquisition_state(scene, state)
    renderer = mujoco.Renderer(scene.model, 480, 640); camera = _camera(); frames = []
    for step in range(120):
        mujoco.mj_step(scene.model, scene.data)
        if step % 4 == 0:
            renderer.update_scene(scene.data, camera=camera)
            frames.append(_annotate(renderer.render(), "25-mm thumb/index acquired state | fixture off"))
    renderer.close(); return frames


def replay(row, state, pocket) -> list[np.ndarray]:
    scene = build_c07_scene(); strategy = TransportStrategy(row["strategy"])
    plan = plan_transport(scene, state, pocket, strategy)
    renderer = mujoco.Renderer(scene.model, 480, 640); camera = _camera(); frames = []
    def callback(current, step, stage):
        if step % 5 == 0:
            renderer.update_scene(current.data, camera=camera)
            frames.append(_annotate(renderer.render(), f"{row['strategy']} | {row['wrist_delta_command_deg']} | {stage}"))
    run_transport_trial(
        scene, state, pocket, strategy,
        wrist_delta_deg=tuple(row["wrist_delta_command_deg"]),
        preshape=None if row["preshape"] is None else PreshapeCondition(row["preshape"]),
        transport_plan=plan, frame_callback=callback,
    )
    renderer.close(); return frames


def main() -> None:
    audit = json.loads((OUTPUT / "static_reachability.json").read_text(encoding="utf-8"))
    result = json.loads((OUTPUT / "phase3c07_results.json").read_text(encoding="utf-8"))
    states = {row.state_id: row for row in load_acquisition_states(OUTPUT / "matched_states")}
    pocket = pocket_volume_from_audit(audit); rows = result["trials"]
    video_dir = OUTPUT / "videos"; generated = []
    path = video_dir / "25mm_thumb_index_acquisition.mp4"; _write(path, acquisition_video(next(iter(states.values())))); generated.append(str(path))
    choices = [
        ("fixed_wrist_failed_pocket_transport.mp4", lambda r: r["strategy"] == "T1_POCKET_DIRECTED" and r["first_pocket_entry_step"] is None),
        ("successful_pocket_directed_transport.mp4", lambda r: r["strategy"] == "T1_POCKET_DIRECTED" and r["first_pocket_entry_step"] is not None),
        ("wrist_assisted_pocket_entry.mp4", lambda r: r["strategy"] == "T2_WRIST_ASSISTED" and r["first_pocket_entry_step"] is not None and r["preshape"] is None),
        ("ring_little_preshape.mp4", lambda r: r["preshape"] is not None),
        ("mechanical_cage_formation.mp4", lambda r: r["cage_formed"]),
        ("1000_step_cage_hold.mp4", lambda r: r["hold_survival"].get("1000", False)),
        ("pocket_not_reached_failure.mp4", lambda r: "POCKET_NOT_REACHED" in r["failures"]),
        ("sphere_rollout_failure.mp4", lambda r: "SPHERE_ROLLED_OUT" in r["failures"]),
    ]
    used = set()
    for filename, predicate in choices:
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            continue
        row = min(candidates, key=lambda value: value["closest_approach_m"])
        identity = (row["state_id"], row["strategy"], tuple(row["wrist_delta_command_deg"]), row["preshape"])
        # Distinct labels may truthfully point to the same observed replay, but
        # render it only once and do not fabricate missing behaviors.
        if identity in used and filename != "pocket_not_reached_failure.mp4":
            continue
        used.add(identity)
        path = video_dir / filename; _write(path, replay(row, states[row["state_id"]], pocket)); generated.append(str(path))
    summary_path = OUTPUT / "phase3c07_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")); summary["videos"] = generated
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (ROOT / "docs/PHASE3C07_RESULTS.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Truthful videos\n\n"
            + "\n".join(f"- `{Path(path).relative_to(ROOT).as_posix()}`" for path in generated)
            + "\n"
        )
    print(json.dumps({"generated": generated, "note": "Only behaviors present in measured results were rendered."}, indent=2))


if __name__ == "__main__":
    main()
