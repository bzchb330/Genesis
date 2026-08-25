from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import mujoco
import numpy as np

from ..config import ROOT
from .config import FINGERS, SUPPORT_SURFACES
from .model import ShadowScene, build_shadow_scene


def _name(model, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def compiled_shadow_audit(scene: ShadowScene | None = None) -> dict:
    scene = scene or build_shadow_scene()
    model = scene.model
    joints = {}
    for joint_id in range(24):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joints[name] = {
            "range": model.jnt_range[joint_id].tolist(),
            "damping": float(model.dof_damping[model.jnt_dofadr[joint_id]]),
            "armature": float(model.dof_armature[model.jnt_dofadr[joint_id]]),
            "frictionloss": float(model.dof_frictionloss[model.jnt_dofadr[joint_id]]),
        }
    actuators = {}
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        actuators[name] = {
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange": model.actuator_forcerange[actuator_id].tolist(),
            "kp": float(model.actuator_gainprm[actuator_id, 0]),
            "transmission_type": int(model.actuator_trntype[actuator_id]),
        }
    surfaces = {}
    for surface in SUPPORT_SURFACES:
        geom_records = []
        names = scene.fingertip_geoms[surface] if surface in FINGERS else scene.collision_geoms[surface]
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            geom_records.append(
                {
                    "name": name,
                    "type": int(model.geom_type[geom_id]),
                    "condim": int(model.geom_condim[geom_id]),
                    "friction": model.geom_friction[geom_id].tolist(),
                    "solref": model.geom_solref[geom_id].tolist(),
                    "solimp": model.geom_solimp[geom_id].tolist(),
                }
            )
        surfaces[surface] = geom_records
    return {
        "model": scene.config.hand.model_name,
        "source_repository": scene.config.hand.source_repository,
        "source_commit": scene.config.hand.source_commit,
        "compiled": {
            "bodies": int(model.nbody),
            "hand_bodies_excluding_world": 25,
            "hand_joints": 24,
            "scene_joints_including_object_freejoint": int(model.njnt),
            "scene_dofs_including_object_freejoint": int(model.nv),
            "actuators": int(model.nu),
            "tendons": int(model.ntendon),
            "geoms": int(model.ngeom),
        },
        "solver": {
            "timestep": float(model.opt.timestep),
            "cone": int(model.opt.cone),
            "impratio": float(model.opt.impratio),
            "integrator": int(model.opt.integrator),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
        },
        "semantic": {
            "palm_body": scene.config.hand.palm_body,
            "wrist_joints": list(scene.config.hand.wrist_joints),
            "finger_bodies": {key: list(value) for key, value in scene.config.hand.finger_bodies.items()},
            "finger_joints": {key: list(value) for key, value in scene.config.hand.finger_joints.items()},
            "fingertip_bodies": dict(scene.config.hand.fingertip_bodies),
            "actuator_groups": {key: list(value) for key, value in scene.config.hand.actuator_groups.items()},
        },
        "joint_properties": joints,
        "actuator_properties": actuators,
        "contact_surfaces": surfaces,
    }


def write_shadow_audit(
    markdown_path: Path | None = None, json_path: Path | None = None
) -> dict:
    audit = compiled_shadow_audit()
    markdown_path = markdown_path or ROOT / "docs/PHASE3A_SHADOW_HAND_AUDIT.md"
    json_path = json_path or ROOT / "outputs/phase3A/shadow_hand_audit.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    semantic = audit["semantic"]
    lines = [
        "# Phase 3A Shadow Hand Audit",
        "",
        "## Provenance",
        "",
        f"- Model: {audit['model']}",
        f"- Official source: `{audit['source_repository']}`",
        f"- Vendored commit: `{audit['source_commit']}`",
        "- Upstream license: Apache-2.0 (vendored alongside the model)",
        "- The upstream MJCF and assets are unchanged. Semantic collision names are added to the in-memory runtime XML.",
        "",
        "## Compiled structure",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in audit["compiled"].items())
    lines.extend(
        [
            f"- Palm body: `{semantic['palm_body']}`",
            f"- Wrist joints: {', '.join(f'`{name}`' for name in semantic['wrist_joints'])}",
            "",
            "## Semantic finger chains",
            "",
        ]
    )
    for finger in FINGERS:
        lines.append(
            f"- {finger}: bodies {', '.join(f'`{name}`' for name in semantic['finger_bodies'][finger])}; "
            f"joints {', '.join(f'`{name}`' for name in semantic['finger_joints'][finger])}; "
            f"tip `{semantic['fingertip_bodies'][finger]}`"
        )
    lines.extend(["", "## Joint limits and passive parameters", "", "| Joint | Range (rad) | Damping | Armature | Friction loss |", "|---|---:|---:|---:|---:|"])
    for name, record in audit["joint_properties"].items():
        lines.append(
            f"| `{name}` | {record['range']} | {record['damping']:.6g} | {record['armature']:.6g} | {record['frictionloss']:.6g} |"
        )
    lines.extend(["", "## Actuator limits", "", "| Actuator | Control range | Force range | Position gain |", "|---|---:|---:|---:|"])
    for name, record in audit["actuator_properties"].items():
        lines.append(f"| `{name}` | {record['ctrlrange']} | {record['forcerange']} | {record['kp']:.6g} |")
    lines.extend(["", "## Collision/contact representation", ""])
    for surface, records in audit["contact_surfaces"].items():
        lines.append(f"### {surface}")
        lines.append("")
        for record in records:
            lines.append(
                f"- `{record['name']}`: compiled geom type {record['type']}, condim {record['condim']}, "
                f"friction {record['friction']}, solref {record['solref']}, solimp {record['solimp']}"
            )
        lines.append("")
    solver = audit["solver"]
    lines.extend(
        [
            "## Solver",
            "",
            f"- timestep: {solver['timestep']} s",
            f"- cone enum: {solver['cone']} (elliptic in upstream MJCF)",
            f"- impratio: {solver['impratio']}",
            f"- integrator enum: {solver['integrator']}",
            f"- iterations: {solver['iterations']}",
            f"- line-search iterations: {solver['ls_iterations']}",
            "",
            "No Phase 2 Allegro code path or historical physics parameter is modified by this integration.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit
