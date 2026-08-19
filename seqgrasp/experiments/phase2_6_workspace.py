from __future__ import annotations

import math

import numpy as np


def world_to_frame(points: np.ndarray, origin: np.ndarray, rotation_world: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - np.asarray(origin)) @ np.asarray(rotation_world)


def cylinder_surface_geometry(points: np.ndarray, center: np.ndarray, radius: float, half_height: float):
    """Return unsigned surface distance, nearest points, and outward normals for a vertical cylinder."""

    points = np.asarray(points, dtype=float)
    local = points - np.asarray(center, dtype=float)
    radial_norm = np.linalg.norm(local[:, :2], axis=1)
    safe = np.maximum(radial_norm, np.finfo(float).eps)
    radial_unit = local[:, :2] / safe[:, None]
    side_point = np.c_[radial_unit * radius, np.clip(local[:, 2], -half_height, half_height)]
    cap_point = np.c_[np.where((radial_norm <= radius)[:, None], local[:, :2], radial_unit * radius), np.sign(local[:, 2]) * half_height]
    side_distance = np.linalg.norm(local - side_point, axis=1)
    cap_distance = np.linalg.norm(local - cap_point, axis=1)
    use_side = side_distance <= cap_distance
    nearest_local = np.where(use_side[:, None], side_point, cap_point)
    normals = np.zeros_like(local)
    normals[use_side, :2] = radial_unit[use_side]
    normals[~use_side, 2] = np.where(local[~use_side, 2] >= 0, 1.0, -1.0)
    return np.minimum(side_distance, cap_distance), nearest_local + center, normals


def contact_opposition_angle_deg(inward_normals: np.ndarray) -> float:
    normals = np.asarray(inward_normals, dtype=float)
    if len(normals) < 2:
        return 0.0
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), np.finfo(float).eps)
    dots = np.clip(normals @ normals.T, -1.0, 1.0)
    upper = dots[np.triu_indices(len(normals), 1)]
    return float(np.degrees(np.max(np.arccos(upper))))


def accessible_surface_samples(points, center, radius, half_height, fingertip_radius, tolerance):
    distance, contact_points, outward = cylinder_surface_geometry(points, center, radius, half_height)
    mask = np.abs(distance - fingertip_radius) <= tolerance
    return mask, contact_points[mask], -outward[mask]


def pairwise_envelope_boxes(point_clouds: dict[str, np.ndarray], fingertip_radii: dict[str, float], radius: float, half_height: float):
    boxes = {}
    names = tuple(point_clouds)
    per_finger = {}
    for finger in names:
        extent = np.asarray([radius + fingertip_radii[finger], radius + fingertip_radii[finger], half_height + fingertip_radii[finger]])
        per_finger[finger] = (np.min(point_clouds[finger], axis=0) - extent, np.max(point_clouds[finger], axis=0) + extent)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            low = np.maximum(per_finger[left][0], per_finger[right][0])
            high = np.minimum(per_finger[left][1], per_finger[right][1])
            if np.all(low < high):
                boxes[f"{left}+{right}"] = (low, high)
    return boxes


def lexicographic_pose_key(record: dict) -> tuple:
    return (
        int(record["valid_initial_geometry"]),
        int(record["accessible_finger_count"] >= 2),
        int(record["opposition_available"]),
        int(record["ferrari_canny_epsilon"] > 0),
        int(record["accessible_finger_count"] >= 3),
        int(record["palm_support_available"]),
        -float(record["predicted_penetration_m"]),
        float(record["minimum_joint_margin_rad"]),
    )
