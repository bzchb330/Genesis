from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .config import ConfigBundle, ROOT

def _vec(xs): return " ".join(str(x) for x in xs)

def build_scene(cfg: ConfigBundle) -> tuple[mujoco.MjModel, mujoco.MjData]:
    hand_path = ROOT / cfg.hand.model_path
    hand = ET.parse(hand_path).getroot()
    world = hand.find("worldbody")
    palm = world.find(f".//body[@name='{cfg.hand.palm_body}']") if world is not None else None
    if palm is None: raise ValueError("hand MJCF has no root body")
    palm.set("pos", _vec(cfg.hand.mount_pos)); palm.set("quat", _vec(cfg.hand.mount_quat))
    # The upstream collision geoms are intentionally anonymous. Give configured
    # fingertip geoms stable identities so sensing remains hand-config driven.
    if len(cfg.hand.fingertip_bodies) != len(cfg.hand.finger_geom_mapping):
        raise ValueError("fingertip body and finger mapping counts differ")
    for body_name, (_, geom_names) in zip(cfg.hand.fingertip_bodies, cfg.hand.finger_geom_mapping.items()):
        body = world.find(f".//body[@name='{body_name}']")
        if body is None: raise ValueError(f"missing configured fingertip body {body_name}")
        collision_geoms = [g for g in body.findall("geom") if "collision" in g.get("class", "")]
        if len(collision_geoms) != len(geom_names):
            raise ValueError(f"cannot bind configured fingertip geoms for {body_name}")
        for geom, geom_name in zip(collision_geoms, geom_names): geom.set("name", geom_name)
    option = hand.find("option")
    if option is None: option = ET.SubElement(hand, "option")
    option.set("timestep", str(cfg.scene.timestep)); option.set("gravity", "0 0 -9.81")
    # Menagerie ships position servos; the task controller computes physical torque.
    actuator = hand.find("actuator")
    if actuator is None: raise ValueError("hand MJCF has no actuators")
    for old in list(actuator):
        motor = ET.Element("motor", name=old.get("name", ""), joint=old.get("joint", ""), ctrllimited="false")
        actuator.remove(old); actuator.append(motor)
    ET.SubElement(world, "light", name="key", pos="0 -0.4 0.8", dir="0 0.4 -0.8")
    ET.SubElement(world, "geom", name="table", type="box", size=_vec(cfg.scene.table_size), pos=_vec(cfg.scene.table_pos), rgba="0.5 0.5 0.5 1")
    for obj in cfg.scene.objects:
        body = ET.SubElement(world, "body", name=obj.name, pos=_vec(obj.pos))
        ET.SubElement(body, "freejoint", name=f"{obj.name}_free")
        typ = "box" if obj.shape == "cube" else obj.shape
        ET.SubElement(body, "geom", name=f"{obj.name}_geom", type=typ, size=_vec(obj.size), mass=str(obj.mass), friction=_vec(obj.friction), rgba=_vec(obj.rgba))
    xml = ET.tostring(hand, encoding="unicode")
    assets = {str(p.relative_to(hand_path.parent)).replace("\\", "/"): p.read_bytes() for p in (hand_path.parent / "assets").rglob("*") if p.is_file()}
    model = mujoco.MjModel.from_xml_string(xml, assets)
    if model.nu != cfg.hand.dof_count: raise ValueError(f"configured DoF {cfg.hand.dof_count}, model actuators {model.nu}")
    for name in cfg.hand.actuator_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0: raise ValueError(f"missing actuator {name}")
    return model, mujoco.MjData(model)

def randomize_objects(model, data, cfg: ConfigBundle, rng: np.random.Generator) -> None:
    for obj in cfg.scene.objects:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.name}_free")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr+3] = np.asarray(obj.pos) + np.r_[rng.uniform(-cfg.scene.placement_jitter_xy, cfg.scene.placement_jitter_xy, 2), 0.0]
        data.qpos[adr+3:adr+7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
