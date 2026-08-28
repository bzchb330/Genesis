from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..config import ROOT
from .config import FINGERS, Phase3Config, load_phase3_config


@dataclass(frozen=True)
class ShadowScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: Phase3Config
    collision_geoms: dict[str, tuple[str, ...]]
    fingertip_geoms: dict[str, tuple[str, ...]]
    actuator_ids: dict[str, np.ndarray]
    joint_ids: dict[str, np.ndarray]
    object_body_id: int
    object_joint_id: int
    fixture_eq_id: int
    fixture_mocap_id: int


def _vec(values) -> str:
    return " ".join(str(value) for value in values)


def _collision_geoms(body: ET.Element) -> list[ET.Element]:
    return [geom for geom in body.findall("geom") if geom.get("class") == "plastic_collision"]


def _name_runtime_collision_geoms(
    root: ET.Element, cfg: Phase3Config
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Shadow Hand MJCF has no worldbody")
    collision: dict[str, tuple[str, ...]] = {}
    fingertips: dict[str, tuple[str, ...]] = {}
    semantic_bodies = {"palm": (cfg.hand.palm_body,), **cfg.hand.finger_bodies}
    for semantic, body_names in semantic_bodies.items():
        names: list[str] = []
        for body_name in body_names:
            body = world.find(f".//body[@name='{body_name}']")
            if body is None:
                raise ValueError(f"missing configured Shadow body {body_name}")
            for ordinal, geom in enumerate(_collision_geoms(body)):
                name = f"phase3_{semantic}_{body_name}_collision_{ordinal}"
                geom.set("name", name)
                names.append(name)
        collision[semantic] = tuple(names)
    for finger in FINGERS:
        tip_body = cfg.hand.fingertip_bodies[finger]
        tip_prefix = f"phase3_{finger}_{tip_body}_collision_"
        fingertips[finger] = tuple(name for name in collision[finger] if name.startswith(tip_prefix))
        if not fingertips[finger]:
            raise ValueError(f"no collision geom found for {finger} fingertip body {tip_body}")
    return collision, fingertips


def build_shadow_scene(
    config: Phase3Config | None = None,
    *,
    model_transform: Callable[[ET.Element, Phase3Config], None] | None = None,
) -> ShadowScene:
    cfg = config or load_phase3_config()
    model_path = ROOT / cfg.hand.model_path
    root = ET.parse(model_path).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Shadow Hand MJCF has no worldbody")
    forearm = world.find(f".//body[@name='{cfg.hand.forearm_body}']")
    if forearm is None:
        raise ValueError("configured Shadow forearm body is missing")
    forearm.set("pos", _vec(cfg.hand.mount_pos))
    forearm.set("quat", _vec(cfg.hand.mount_quat))
    if model_transform is not None:
        # Transform only the parsed runtime composition. The vendored source
        # XML remains immutable and historical callers retain identical output.
        model_transform(root, cfg)
    collision_geoms, fingertip_geoms = _name_runtime_collision_geoms(root, cfg)

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(cfg.raw["timestep"]))
    option.set("gravity", "0 0 -9.81")

    ET.SubElement(world, "light", name="phase3_key", pos="0 -0.5 0.8", dir="0 0.4 -0.8")
    floor = cfg.raw["floor"]
    ET.SubElement(
        world,
        "geom",
        name=floor["name"],
        type="plane",
        size="1 1 0.05",
        pos=f"0 0 {floor['z']}",
        rgba="0.35 0.35 0.38 1",
    )
    obj = cfg.object
    body = ET.SubElement(world, "body", name=obj["name"], pos=_vec(obj["initial_pos"]), quat=_vec(obj["initial_quat"]))
    ET.SubElement(body, "freejoint", name=f"{obj['name']}_free")
    object_attributes = {
        "name": f"{obj['name']}_geom",
        "type": obj["shape"],
        "size": _vec(obj["size"]),
        "friction": _vec(obj["friction"]),
        "rgba": _vec(obj["rgba"]),
        "condim": "6",
        "priority": "1",
    }
    if "density" in obj:
        object_attributes["density"] = str(obj["density"])
    ET.SubElement(
        body,
        "geom",
        **object_attributes,
    )
    fixture_body = ET.SubElement(
        world,
        "body",
        name="phase3_fixture_anchor",
        mocap="true",
        pos=_vec(obj["initial_pos"]),
        quat=_vec(obj["initial_quat"]),
    )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name=obj["fixture_name"],
        body1=obj["name"],
        body2="phase3_fixture_anchor",
        relpose="0 0 0 1 0 0 0",
    )

    assets = {
        str(path.relative_to(model_path.parent)).replace("\\", "/"): path.read_bytes()
        for path in (model_path.parent / "assets").rglob("*")
        if path.is_file()
    }
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    data = mujoco.MjData(model)
    actuator_ids = {
        group: np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names], dtype=int
        )
        for group, names in cfg.hand.actuator_groups.items()
    }
    joint_ids = {
        finger: np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names], dtype=int
        )
        for finger, names in cfg.hand.finger_joints.items()
    }
    if any(np.any(ids < 0) for ids in (*actuator_ids.values(), *joint_ids.values())):
        raise ValueError("configured Shadow semantic mapping contains a missing compiled name")
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj["name"])
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj['name']}_free")
    fixture_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, obj["fixture_name"])
    fixture_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phase3_fixture_anchor")
    fixture_mocap_id = int(model.body_mocapid[fixture_body_id])
    return ShadowScene(
        model=model,
        data=data,
        config=cfg,
        collision_geoms=collision_geoms,
        fingertip_geoms=fingertip_geoms,
        actuator_ids=actuator_ids,
        joint_ids=joint_ids,
        object_body_id=object_body_id,
        object_joint_id=object_joint_id,
        fixture_eq_id=fixture_eq_id,
        fixture_mocap_id=fixture_mocap_id,
    )


def set_fixture(scene: ShadowScene, active: bool) -> None:
    scene.data.eq_active[scene.fixture_eq_id] = int(active)


def set_object_pose(scene: ShadowScene, position, quaternion=(1.0, 0.0, 0.0, 0.0)) -> None:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    address = scene.model.jnt_qposadr[scene.object_joint_id]
    scene.data.qpos[address : address + 3] = position
    scene.data.qpos[address + 3 : address + 7] = quaternion
    # Pair the initial free-body pose with its mocap fixture anchor. This setup
    # is performed only before fixture release; the object is never kinematically
    # moved afterward.
    scene.data.mocap_pos[scene.fixture_mocap_id] = position
    scene.data.mocap_quat[scene.fixture_mocap_id] = quaternion
    velocity_address = scene.model.jnt_dofadr[scene.object_joint_id]
    scene.data.qvel[velocity_address : velocity_address + 6] = 0.0
    mujoco.mj_forward(scene.model, scene.data)
