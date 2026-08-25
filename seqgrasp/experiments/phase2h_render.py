from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import ConfigBundle
from .resource_components import _finger_prefixes
from .static_wrist import palm_normal_world


FINGER_COLORS = {
    "index": (0.12, 0.47, 0.71, 1.0),
    "middle": (1.0, 0.50, 0.05, 1.0),
    "ring": (0.17, 0.63, 0.17, 1.0),
    "thumb": (0.58, 0.40, 0.74, 1.0),
}


@dataclass(frozen=True)
class CameraSpec:
    name: str
    azimuth_deg: float
    elevation_deg: float
    distance_m: float


DEFAULT_CAMERAS = (
    CameraSpec("palm-normal", 68.0, -12.0, 0.30),
    CameraSpec("side", 158.0, -10.0, 0.30),
    CameraSpec("oblique", 128.0, -28.0, 0.34),
)


def _camera(spec: CameraSpec, lookat: Iterable[float]) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(tuple(lookat), dtype=float)
    camera.distance = spec.distance_m
    camera.azimuth = spec.azimuth_deg
    camera.elevation = spec.elevation_deg
    return camera


def color_fingers(model: mujoco.MjModel, cfg: ConfigBundle) -> None:
    prefixes = _finger_prefixes(cfg)
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        for finger, prefix in prefixes.items():
            if body_name.startswith(prefix + "_"):
                model.geom_rgba[geom_id] = FINGER_COLORS[finger]


def _add_box(scene, center: np.ndarray, half_size: np.ndarray, rgba=(0.1, 0.7, 0.7, 0.18)) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_size, dtype=float),
        np.asarray(center, dtype=float),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_arrow(scene, start: np.ndarray, end: np.ndarray, rgba, width=0.002) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1


def _add_sphere(scene, position: np.ndarray, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius]),
        np.asarray(position, dtype=float),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


class MuJoCoDiagnosticRenderer:
    def __init__(self, cfg: ConfigBundle, width: int = 640, height: int = 480):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.renderer = None
        self.model = None

    def _ensure(self, model: mujoco.MjModel) -> None:
        if self.renderer is None:
            color_fingers(model, self.cfg)
            self.renderer = mujoco.Renderer(model, width=self.width, height=self.height, max_geom=2000)
            self.model = model
        elif model is not self.model:
            raise ValueError("one diagnostic renderer cannot span different MuJoCo models")

    def render(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_spec: CameraSpec,
        *,
        lookat: Iterable[float],
        candidate_region: dict | None = None,
        row: dict | None = None,
        contact_overlay: bool = False,
        show_vectors: bool = False,
        markers: list[tuple[np.ndarray, tuple[float, float, float, float]]] | None = None,
    ) -> np.ndarray:
        self._ensure(model)
        self.renderer.update_scene(data, camera=_camera(camera_spec, lookat))
        scene = self.renderer.scene
        if candidate_region is not None:
            bounds = candidate_region["center_bounds_m"]
            low = np.asarray([bounds[axis][0] for axis in "xyz"], dtype=float)
            high = np.asarray([bounds[axis][1] for axis in "xyz"], dtype=float)
            object_b = next(item for item in self.cfg.scene.objects if item.name == "object_b")
            if object_b.shape == "cylinder":
                object_extent = np.asarray([object_b.size[0], object_b.size[0], object_b.size[1]], dtype=float)
            else:
                object_extent = np.asarray(object_b.size[:3], dtype=float)
            # The bounds describe candidate center positions. Show the swept
            # physical B envelope so the small center interval is not hidden
            # inside the rendered cylinder.
            _add_box(scene, (low + high) / 2.0, (high - low) / 2.0 + object_extent)
        if show_vectors:
            palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.hand.palm_body)
            origin = np.asarray(data.xpos[palm_id], dtype=float)
            normal = palm_normal_world(self.cfg.hand.mount_quat)
            _add_arrow(scene, origin, origin + 0.055 * normal, (0.9, 0.1, 0.2, 1.0), 0.0024)
            _add_arrow(scene, origin, origin + np.asarray([0.0, 0.0, -0.065]), (0.1, 0.1, 0.1, 1.0), 0.0024)
        if contact_overlay and row is not None:
            flags = np.asarray(row["B_per_finger_contact_flag"], dtype=bool)
            positions = np.asarray(row["B_contact_positions_m"], dtype=float)
            normals = np.asarray(row["B_contact_normals"], dtype=float)
            forces = np.asarray(row["B_per_finger_normal_force_N"], dtype=float)
            overlay_colors = {
                "index": (1.0, 0.85, 0.05, 1.0),
                "middle": (1.0, 0.45, 0.05, 1.0),
                "ring": (0.25, 1.0, 0.25, 1.0),
                "thumb": (1.0, 0.10, 0.80, 1.0),
            }
            for index, finger in enumerate(("index", "middle", "ring", "thumb")):
                if not flags[index]:
                    continue
                color = overlay_colors[finger]
                _add_sphere(scene, positions[index], 0.0050, color)
                arrow_length = 0.018 + 0.012 * min(float(forces[index]), 2.0)
                # Stored normals point inward toward B. Start outside and end
                # at the actual contact point so the true direction remains
                # visible instead of being occluded inside the cylinder.
                _add_arrow(scene, positions[index] - arrow_length * normals[index], positions[index], color, 0.0035)
        for position, color in markers or []:
            _add_sphere(scene, np.asarray(position, dtype=float), 0.003, color)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.model = None


def annotate_frame(frame: np.ndarray, lines: list[str], *, released: bool) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    line_height = 24
    width = min(image.width - 24, max(360, max((draw.textlength(line, font=font) for line in lines), default=0) + 24))
    height = 16 + line_height * len(lines)
    draw.rounded_rectangle((10, 10, 10 + width, 10 + height), radius=8, fill=(255, 255, 255, 205))
    for index, line in enumerate(lines):
        draw.text((22, 18 + line_height * index), line, fill=(10, 25, 45, 255), font=font)
    label = "FIXTURE RELEASED" if released else "FIXTURE ACTIVE"
    color = (180, 35, 35, 235) if released else (35, 100, 180, 235)
    bbox = draw.textbbox((0, 0), label, font=font)
    label_width = bbox[2] - bbox[0] + 20
    draw.rounded_rectangle((image.width - label_width - 10, 10, image.width - 10, 46), radius=8, fill=color)
    draw.text((image.width - label_width, 18), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
