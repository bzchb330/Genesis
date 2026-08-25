from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..sensing.contact import ContactRecord, extract_contacts
from .config import SUPPORT_SURFACES
from .model import ShadowScene


@dataclass(frozen=True)
class ShadowContactState:
    records: tuple[ContactRecord, ...]
    object_records: tuple[ContactRecord, ...]
    records_by_surface: dict[str, tuple[ContactRecord, ...]]
    contact_flags: np.ndarray
    normal_forces: np.ndarray
    tangential_forces: np.ndarray
    support_vector: np.ndarray
    support_load_fraction: np.ndarray
    penetration_by_surface: np.ndarray
    maximum_penetration: float
    maximum_penetration_pair: tuple[str, str] | None


def extract_shadow_contacts(scene: ShadowScene) -> ShadowContactState:
    records = tuple(extract_contacts(scene.model, scene.data))
    object_name = scene.config.object["name"]
    object_records = tuple(
        record for record in records if object_name in {record.body1_name, record.body2_name}
    )
    by_surface: dict[str, tuple[ContactRecord, ...]] = {}
    for surface in SUPPORT_SURFACES:
        geom_names = set(scene.collision_geoms[surface])
        by_surface[surface] = tuple(
            record
            for record in object_records
            if geom_names.intersection((record.geom1_name, record.geom2_name))
        )
    flags = np.asarray([bool(by_surface[surface]) for surface in SUPPORT_SURFACES], dtype=np.float64)
    normal = np.asarray(
        [sum(record.normal_force for record in by_surface[surface]) for surface in SUPPORT_SURFACES],
        dtype=np.float64,
    )
    tangential = np.asarray(
        [sum(record.tangential_force for record in by_surface[surface]) for surface in SUPPORT_SURFACES],
        dtype=np.float64,
    )
    total = float(normal.sum())
    loads = normal / total if total > 0.0 else np.zeros(len(SUPPORT_SURFACES), dtype=np.float64)
    penetration = np.asarray(
        [max((-record.distance for record in by_surface[surface]), default=0.0) for surface in SUPPORT_SURFACES],
        dtype=np.float64,
    )
    deepest = min(object_records, key=lambda record: record.distance, default=None)
    max_penetration = max(0.0, -deepest.distance) if deepest is not None else 0.0
    pair = (deepest.geom1_name, deepest.geom2_name) if deepest is not None else None
    return ShadowContactState(
        records=records,
        object_records=object_records,
        records_by_surface=by_surface,
        contact_flags=flags,
        normal_forces=normal,
        tangential_forces=tangential,
        support_vector=normal.copy(),
        support_load_fraction=loads,
        penetration_by_surface=penetration,
        maximum_penetration=float(max_penetration),
        maximum_penetration_pair=pair,
    )


def fingertip_object_penetration(scene: ShadowScene, finger: str) -> float:
    tip_geoms = set(scene.fingertip_geoms[finger])
    return max(
        (
            -record.distance
            for record in extract_shadow_contacts(scene).object_records
            if tip_geoms.intersection((record.geom1_name, record.geom2_name))
        ),
        default=0.0,
    )


def contact_force_for(scene: ShadowScene, surface: str) -> float:
    state = extract_shadow_contacts(scene)
    return float(state.normal_forces[SUPPORT_SURFACES.index(surface)])


def object_velocity(scene: ShadowScene) -> tuple[np.ndarray, np.ndarray]:
    dof_address = scene.model.jnt_dofadr[scene.object_joint_id]
    velocity = scene.data.qvel[dof_address : dof_address + 6].copy()
    return velocity[:3], velocity[3:]
