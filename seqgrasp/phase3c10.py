"""Phase 3C-1.0 support-gated metrics and B03 validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc
import yaml

from .config import ROOT
from .phase3.model import set_fixture
from .phase3.control import actuator_target_from_qpos
from .phase3.contacts import object_velocity
from .phase3c0 import gravity_in_palm_frame, object_pose_in_palm, palm_transform
from .phase3c07 import SPHERE_RADIUS_M, _geom_ids, _object_geom_id, _set_object_palm, contact_geometry, floor_contact
from .phase3c08 import FOREARM_JOINT_NAME, build_forearm_scene
from .phase3c09 import OUTPUT as C09_OUTPUT, SURFACES, _finger_interpolation, cluster_storage_mask, exact_clearance_grid


OUTPUT = ROOT / "outputs/phase3C10"
STORAGE_SURFACES = ("middle", "ring", "little", "palm")


def load_phase3c10_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C10_b03_handoff.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def support_gated_progress(
    pocket_distance_m: Iterable[float], hand_support_force_n: Iterable[float], relative_speed_mps: Iterable[float], speed_gate_mps: float,
    *, non_hand_support: Iterable[bool] | None = None, support_force_tolerance_n: float = 1e-9,
) -> dict[str, Any]:
    distance = np.asarray(tuple(pocket_distance_m), dtype=float); force = np.asarray(tuple(hand_support_force_n), dtype=float); speed = np.asarray(tuple(relative_speed_mps), dtype=float)
    raw_progress = np.maximum(0.0, distance[0] - distance)
    external = np.zeros(len(distance), dtype=bool) if non_hand_support is None else np.asarray(tuple(non_hand_support), dtype=bool)
    if not (len(distance) == len(force) == len(speed) == len(external)):
        raise ValueError("distance, force, speed, and non-hand-support series must have equal length")
    eligible = (force > support_force_tolerance_n) & ~external & (speed <= speed_gate_mps)
    valid_progress = np.where(eligible, raw_progress, 0.0)
    return {"raw_progress_m": raw_progress, "eligible": eligible, "valid_progress_m": valid_progress,
            "valid_supported_minimum_distance_m": float(np.min(distance[eligible])) if np.any(eligible) else None,
            "maximum_valid_progress_m": float(np.max(valid_progress)), "speed_gate_mps": float(speed_gate_mps),
            "speed_gate_label": "ENGINEERING_DIAGNOSTIC_ONLY"}


def transfer_clearance(clearances_m: dict[str, float]) -> dict[str, Any]:
    values = {surface: float(clearances_m[surface]) for surface in ("middle", "ring", "little")}
    return {"minimum_storage_finger_clearance_m": min(values.values()), "clearance_by_surface_m": values,
            "corridor_collision_free": all(value >= 0.0 for value in values.values())}


def receiver_ready(current_qpos: np.ndarray, receiver_qpos: np.ndarray, receiver_joint_addresses: np.ndarray, cspace_clear: bool, tolerance_rad: float) -> dict[str, Any]:
    error = float(np.sqrt(np.mean((np.asarray(current_qpos)[receiver_joint_addresses] - np.asarray(receiver_qpos)[receiver_joint_addresses]) ** 2)))
    return {"ready": bool(error <= tolerance_rad and cspace_clear), "receiver_qpos_rms_error_rad": error,
            "receiver_qpos_tolerance_rad": tolerance_rad, "tolerance_label": "ENGINEERING_GEOMETRY_DIAGNOSTIC_ONLY",
            "cspace_clear": bool(cspace_clear)}


def receiver_ready_with_contact_opportunity(
    current_qpos: np.ndarray,
    receiver_qpos: np.ndarray,
    receiver_joint_addresses: np.ndarray,
    cspace_clear: bool,
    storage_contact_opportunities: int,
    tolerance_rad: float,
) -> dict[str, Any]:
    """Receiver readiness requires geometry *and* storage-side contact opportunity."""
    result = receiver_ready(current_qpos, receiver_qpos, receiver_joint_addresses, cspace_clear, tolerance_rad)
    result["storage_contact_opportunities"] = int(storage_contact_opportunities)
    result["ready"] = bool(result["ready"] and storage_contact_opportunities >= 2)
    result["definition"] = (
        "receiver joint state is within the configurable engineering tolerance, "
        "the sphere C-space remains clear, and at least two middle/ring/little/palm "
        "surfaces provide contact opportunity"
    )
    return result


def workspace_descriptor(
    reachable_points_palm_m: np.ndarray,
    candidate_count: int,
    blocked_by_object: int,
    blocked_by_storage_fingers: int,
    minimum_joint_margin_rad: Iterable[float],
) -> dict[str, Any]:
    """Summarize an already sampled finger workspace without a sufficiency threshold."""
    points = np.asarray(reachable_points_palm_m, dtype=float)
    volume = 0.0
    if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 3:
        volume = float(ConvexHull(points).volume)
    ranges = np.ptp(points, axis=0).tolist() if len(points) else [0.0, 0.0, 0.0]
    return {
        "reachable_point_count": int(len(points)),
        "candidate_count": int(candidate_count),
        "reachable_volume_m3": volume,
        "palm_frame_range_m": ranges,
        "blocked_by_stored_A": int(blocked_by_object),
        "blocked_by_storage_fingers": int(blocked_by_storage_fingers),
        "blocked_fraction": float((blocked_by_object + blocked_by_storage_fingers) / candidate_count) if candidate_count else 0.0,
        "minimum_joint_margin_rad": [float(value) for value in minimum_joint_margin_rad],
        "scientific_sufficiency_threshold": None,
    }


def joint_acquisition_workspace(
    thumb_points_palm_m: np.ndarray,
    index_points_palm_m: np.ndarray,
    opposition_distance_m: float,
) -> dict[str, Any]:
    """Describe paired thumb/index opposition samples for a hypothetical object."""
    thumb = np.asarray(thumb_points_palm_m, dtype=float)
    index = np.asarray(index_points_palm_m, dtype=float)
    if len(thumb) != len(index):
        raise ValueError("paired thumb/index samples must have equal length")
    apertures = np.linalg.norm(thumb - index, axis=1) if len(thumb) else np.asarray([])
    useful = apertures <= opposition_distance_m
    return {
        "paired_sample_count": int(len(apertures)),
        "opposition_sample_count": int(np.count_nonzero(useful)),
        "opposition_fraction": float(np.mean(useful)) if len(useful) else 0.0,
        "minimum_aperture_m": float(np.min(apertures)) if len(apertures) else None,
        "maximum_aperture_m": float(np.max(apertures)) if len(apertures) else None,
        "diagnostic_opposition_distance_m": float(opposition_distance_m),
        "publication_threshold": None,
    }


def m2_contact_mapping() -> dict[str, str]:
    return {"guide": "thumb", "unload_and_migrate": "index", "mode": "M2"}


def scripted_stage_specification() -> list[dict[str, Any]]:
    """Frozen receiver-first ordering; execution remains gated on validated B03."""
    return [
        {"stage": 1, "name": "STABLE_ACQUISITION", "index_unload": False, "receiver_required": False},
        {"stage": 2, "name": "WHOLE_HAND_REORIENTATION", "index_unload": False, "receiver_required": False},
        {"stage": 3, "name": "B03_RECEIVER_PRESHAPE", "index_unload": False, "receiver_required": True},
        {"stage": 4, "name": "INDEX_UNLOAD_THUMB_GUIDE", "index_unload": True, "receiver_required": True},
        {"stage": 5, "name": "CONTROLLED_TRANSPORT_INTO_B03", "index_unload": True, "receiver_required": True},
        {"stage": 6, "name": "STORAGE_SUPPORT_TAKEOVER", "index_unload": True, "receiver_required": True},
    ]


def premature_support_loss(acquisition_force_n: float, storage_force_n: float, tolerance_n: float = 1e-9) -> bool:
    return bool(acquisition_force_n <= tolerance_n and storage_force_n <= tolerance_n)


def b03_entry_success(inside_b03: bool, hand_supported: bool, storage_force_n: float, controlled_speed: bool) -> bool:
    return bool(inside_b03 and hand_supported and storage_force_n > 1e-9 and controlled_speed)


def phase_gate(classification: str) -> dict[str, Any]:
    approved = classification in ("B03-A", "B03-B")
    return {
        "B03_approved": approved,
        "workspace_executed": approved,
        "scripted_handoff_executed": approved,
        "reason": None if approved else "B03 is not dynamically validated; the frozen protocol forbids transport redesign around it",
    }


def metric_repair_audit() -> dict[str, Any]:
    c08 = json.loads((ROOT / "outputs/phase3C08/targeted_dynamics_results.json").read_text(encoding="utf-8")); best = min(c08["rows"], key=lambda row: row["closest_pocket_distance_m"])
    series = np.load(best["timeseries_path"], allow_pickle=False); centers = np.asarray(series["sphere_center_palm_m"])
    wrapper = build_forearm_scene(with_actuator=True); dt = float(wrapper.scene.model.opt.timestep)
    speed = np.linalg.norm(np.gradient(centers, dt, axis=0), axis=1)
    contacts = [json.loads(str(value)) for value in series["contact_geometry_json"]]
    hand_force = np.asarray([sum(record["normal_force_n"] for record in value["records"] if record["surface"] in SURFACES) for value in contacts])
    non_hand_support = np.asarray(series["floor_contact"], dtype=bool)
    gates = load_phase3c10_config()["metrics"]["relative_speed_gates_mps"]
    sensitivity = {}
    for gate in gates:
        result = support_gated_progress(series["pocket_distance_m"], hand_force, speed, float(gate), non_hand_support=non_hand_support)
        sensitivity[str(gate)] = {key: value for key, value in result.items() if key not in ("raw_progress_m", "eligible", "valid_progress_m")}
    minimum_index = int(np.argmin(series["pocket_distance_m"])); rejected = all(not support_gated_progress(series["pocket_distance_m"], hand_force, speed, float(gate), non_hand_support=non_hand_support)["eligible"][minimum_index] for gate in gates)
    return {"trajectory_id": best["trial_id"], "old_raw_minimum_distance_m": float(np.min(series["pocket_distance_m"])),
            "old_raw_minimum_index": minimum_index, "raw_speed_at_minimum_mps": float(speed[minimum_index]),
            "hand_force_at_minimum_n": float(hand_force[minimum_index]), "diagnostic_speed_gates_mps": gates,
            "non_hand_support_at_minimum": bool(non_hand_support[minimum_index]),
            "sensitivity": sensitivity, "flyby_minimum_rejected": rejected,
            "raw_minimum_distance_role": "DESCRIPTIVE_GEOMETRIC_QUANTITY_ONLY",
            "previous_timing_ablations": "previously non-discriminative under failed transport conditions"}


def _storage_surface_clearance(scene, qpos: np.ndarray, points: np.ndarray) -> np.ndarray:
    scene.data.qpos[:] = qpos; mujoco.mj_forward(scene.model, scene.data); origin, rotation = palm_transform(scene); world = origin + points @ rotation.T
    object_geom = _object_geom_id(scene); old = scene.data.geom_xpos[object_geom].copy(); result = np.empty((len(STORAGE_SURFACES), len(points)))
    for si, surface in enumerate(STORAGE_SURFACES):
        geoms = _geom_ids(scene, surface)
        for pi, center in enumerate(world):
            scene.data.geom_xpos[object_geom] = center
            result[si, pi] = min(float(mujoco.mj_geomDistance(scene.model, scene.data, object_geom, geom, .25, None)) for geom in geoms)
    scene.data.geom_xpos[object_geom] = old; return result


def reconstruct_b03_candidates() -> dict[str, Any]:
    c09 = json.loads((C09_OUTPUT / "phase3c09_results.json").read_text(encoding="utf-8")); trajectory = c09["trajectory"]["rows"][0]
    source_index = max(0, trajectory["minimum_index"] - 20); base_qpos = np.asarray(trajectory["series"]["qpos"][source_index])
    c09_cfg = yaml.safe_load((ROOT / "configs/phase3C09_storage_reachability.yaml").read_text(encoding="utf-8"))["storage_manifold"]
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    axes = tuple(np.arange(c09_cfg["center_bounds_palm_m"][axis][0], c09_cfg["center_bounds_palm_m"][axis][1] + c09_cfg["center_spacing_m"]*.5, c09_cfg["center_spacing_m"]) for axis in ("x","y","z"))
    mesh=np.meshgrid(*axes,indexing="ij"); points=np.column_stack([value.ravel() for value in mesh]); shape=tuple(len(axis) for axis in axes)
    pairs=[]; location_valid=np.zeros(len(points),dtype=bool); config_id=0
    for middle in c09_cfg["middle_flexion_fractions"]:
     for ring in c09_cfg["ring_flexion_fractions"]:
      for little in c09_cfg["little_flexion_fractions"]:
       for wrist_offset in c09_cfg["wrist2_offsets_deg"]:
        for forearm in c09_cfg["forearm_PS_deg"]:
         scene.data.qpos[:]=base_qpos
         for finger,value in (("middle",middle),("ring",ring),("little",little)): _finger_interpolation(scene,base_qpos,finger,float(value))
         for name,value in (("rh_WRJ2",scene.data.qpos[scene.model.jnt_qposadr[mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,"rh_WRJ2")]]+np.deg2rad(wrist_offset)),(FOREARM_JOINT_NAME,np.deg2rad(forearm))):
             joint=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,name); lo,hi=scene.model.jnt_range[joint]; scene.data.qpos[scene.model.jnt_qposadr[joint]]=np.clip(value,lo,hi)
         qpos=scene.data.qpos.copy(); clearance,_=exact_clearance_grid(scene,qpos,points); storage_clearance=_storage_surface_clearance(scene,qpos,points)
         near=storage_clearance <= float(c09_cfg["near_surface_numerical_tolerance_m"]); valid=(clearance>=-float(c09_cfg["occupancy_numerical_tolerance_m"])) & (np.sum(near,axis=0)>=2)
         location_valid |= valid
         for point_index in np.flatnonzero(valid):
             pairs.append({"config_id":config_id,"point_index":int(point_index),"center_palm_m":points[point_index].tolist(),"qpos":qpos.tolist(),"configuration":{"middle_fraction":middle,"ring_fraction":ring,"little_fraction":little,"WRJ2_offset_deg":wrist_offset,"forearm_PS_deg":forearm},"storage_clearance_m":storage_clearance[:,point_index].tolist()})
         config_id+=1
    labels,count=cluster_storage_mask(location_valid.reshape(shape)); sizes=[np.count_nonzero(labels==label) for label in range(1,count+1)]; b03_label=int(np.argmax(sizes)+1)
    b03_points_flat=set(np.flatnonzero(labels.ravel()==b03_label).tolist()); b03_pairs=[pair for pair in pairs if pair["point_index"] in b03_points_flat]
    unique=np.asarray([points[index] for index in sorted(b03_points_flat)]); centroid=unique.mean(axis=0)
    medoid_index=int(np.argmin(np.linalg.norm(unique-centroid,axis=1))); selected_points=[unique[medoid_index]]
    while len(selected_points)<int(load_phase3c10_config()["B03_validation"]["candidate_count"]):
        distances=np.min(np.linalg.norm(unique[:,None,:]-np.asarray(selected_points)[None,:,:],axis=2),axis=1); selected_points.append(unique[int(np.argmax(distances))])
    selected=[]
    for number,center in enumerate(selected_points):
        candidates=[pair for pair in b03_pairs if np.allclose(pair["center_palm_m"],center,atol=1e-12)]; pair=min(candidates,key=lambda value:value["config_id"]); pair={**pair,"candidate_id":f"B03_CANDIDATE_{number:02d}"}; selected.append(pair)
    return {"constructed_before_dynamic_outcomes":True,"selection":"medoid then deterministic farthest-point coverage of the largest C09 basin","B03_label":b03_label,"B03_unique_center_count":len(unique),"B03_pair_count":len(b03_pairs),"B03_centroid_palm_m":centroid.tolist(),"B03_centers_palm_m":unique.tolist(),"selected":selected}


def predeclare_gravity_orientations(candidate_manifest: dict[str,Any]) -> list[dict[str,Any]]:
    c08=json.loads((ROOT/"outputs/phase3C08/kinematic_audit.json").read_text(encoding="utf-8")); reach=c08["reachable_gravity_audit"]["augmented"]
    directions=np.asarray(reach["directions"]); configurations=np.asarray(reach["configurations_rad"]); center=np.asarray(candidate_manifest["B03_centroid_palm_m"]); receiver=-center; receiver/=np.linalg.norm(receiver); escape=-receiver
    tangent=np.cross(receiver,np.asarray([0,0,1.0])); tangent/=np.linalg.norm(tangent); intermediate=receiver+tangent; intermediate/=np.linalg.norm(intermediate)
    targets={"RECEIVER_BIASED":receiver,"ESCAPE_BIASED":escape,"TANGENTIAL":intermediate}
    orientations=[{"orientation_id":"NOMINAL","target_basis":"associated valid-candidate configuration","forearm_PS_rad":None,"WRJ1_rad":None,"WRJ2_rad":None,"gravity_direction_target_palm":None}]
    for name,target in targets.items():
        index=int(np.argmax(directions@target)); orientations.append({"orientation_id":name,"target_basis":"B03 centroid/support geometry and existing reachable gravity set","forearm_PS_rad":float(configurations[index,0]),"WRJ1_rad":float(configurations[index,1]),"WRJ2_rad":float(configurations[index,2]),"gravity_direction_target_palm":target.tolist(),"coarse_alignment_projection":float(directions[index]@target)})
    return orientations


def freeze_b03_manifest(output:Path|None=None)->dict[str,Any]:
    candidates=reconstruct_b03_candidates(); orientations=predeclare_gravity_orientations(candidates); trials=[]
    for candidate in candidates["selected"]:
        for orientation in orientations: trials.append({"trial_id":f"{candidate['candidate_id']}_{orientation['orientation_id']}","candidate_id":candidate["candidate_id"],"center_palm_m":candidate["center_palm_m"],"qpos":candidate["qpos"],"configuration":candidate["configuration"],"orientation":orientation})
    payload=json.dumps(trials,sort_keys=True,separators=(",",":")).encode(); manifest={"frozen_before_outcomes":True,"maximum_trials":12,"trial_count":len(trials),"sha256":hashlib.sha256(payload).hexdigest(),"candidates":candidates,"orientations":orientations,"trials":trials}
    destination=output or OUTPUT/"B03_validation_manifest.json"; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def _inside_b03(center:np.ndarray,centers:np.ndarray,spacing:float=.005)->bool:
    return bool(np.min(np.linalg.norm(centers-center,axis=1)) <= np.sqrt(3)*spacing/2+1e-12)


def initialize_b03_trial(wrapper,trial):
    scene=wrapper.scene; mujoco.mj_resetData(scene.model,scene.data); scene.data.qpos[:]=trial["qpos"]
    for name,key in ((FOREARM_JOINT_NAME,"forearm_PS_rad"),("rh_WRJ1","WRJ1_rad"),("rh_WRJ2","WRJ2_rad")):
        value=trial["orientation"][key]
        if value is not None:
            joint=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,name); scene.data.qpos[scene.model.jnt_qposadr[joint]]=value
    mujoco.mj_forward(scene.model,scene.data); _set_object_palm(scene,np.asarray(trial["center_palm_m"])); scene.data.qvel[:]=0; set_fixture(scene,False); scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); mujoco.mj_forward(scene.model,scene.data)


def run_b03_hold_trial(wrapper,trial,b03_centers:np.ndarray)->dict[str,Any]:
    initialize_b03_trial(wrapper,trial); scene=wrapper.scene; initial=object_pose_in_palm(scene,scene.object_body_id)[0]; samples=[]
    initial_contact=contact_geometry(scene); initial_penetration=float(initial_contact["maximum_penetration_m"])
    for step in range(1,1001):
        mujoco.mj_step(scene.model,scene.data); center=object_pose_in_palm(scene,scene.object_body_id)[0]; linear,angular=object_velocity(scene); contacts=contact_geometry(scene); topology=contacts["contact_topology"]
        samples.append({"step":step,"center_palm_m":center.tolist(),"displacement_m":float(np.linalg.norm(center-initial)),"linear_speed_mps":float(np.linalg.norm(linear)),"angular_speed_radps":float(np.linalg.norm(angular)),"inside_B03":_inside_b03(center,b03_centers),"middle_contact":"middle" in topology,"ring_contact":"ring" in topology,"little_contact":"little" in topology,"palm_contact":"palm" in topology,"support_topology":topology,"load_bearing_topology":contacts["load_bearing_topology"],"acquisition_force_n":contacts["acquisition_force_n"],"storage_force_n":contacts["storage_force_n"],"penetration_by_surface_m":contacts["penetration_by_surface_m"],"maximum_penetration_m":contacts["maximum_penetration_m"],"floor_contact":floor_contact(scene),"finite":bool(np.all(np.isfinite(scene.data.qpos)) and np.all(np.isfinite(scene.data.qvel))),"qpos":scene.data.qpos.tolist()})
    checkpoints=load_phase3c10_config()["B03_validation"]["checkpoints"]
    survival={str(checkpoint):bool(all(sample["inside_B03"] and not sample["floor_contact"] and sample["finite"] for sample in samples[:checkpoint])) for checkpoint in checkpoints}
    escaped=next((sample for sample in samples if not sample["inside_B03"]),None); gross_overlap=initial_penetration>1e-9
    return {**{key:trial[key] for key in ("trial_id","candidate_id","center_palm_m","configuration","orientation")},"initial_maximum_penetration_m":initial_penetration,"gross_overlap":gross_overlap,"survival":survival,"maximum_displacement_m":max(sample["displacement_m"] for sample in samples),"middle_contact":any(s["middle_contact"] for s in samples),"ring_contact":any(s["ring_contact"] for s in samples),"little_contact":any(s["little_contact"] for s in samples),"palm_contact":any(s["palm_contact"] for s in samples),"dominant_support_topology":max((tuple(s["load_bearing_topology"]) for s in samples),key=lambda topology:sum(tuple(x["load_bearing_topology"])==topology for x in samples)),"maximum_penetration_m":max(s["maximum_penetration_m"] for s in samples),"escape_direction_palm":None if escaped is None else (np.asarray(escaped["center_palm_m"])-initial).tolist(),"samples":samples}


def run_b03_validation()->dict[str,Any]:
    manifest=json.loads((OUTPUT/"B03_validation_manifest.json").read_text(encoding="utf-8")); wrapper=build_forearm_scene(with_actuator=True); centers=np.asarray(manifest["candidates"]["B03_centers_palm_m"]); rows=[]; series_dir=OUTPUT/"B03_hold_series"; series_dir.mkdir(parents=True,exist_ok=True)
    for index,trial in enumerate(manifest["trials"]):
        row=run_b03_hold_trial(wrapper,trial,centers); samples=row.pop("samples"); np.savez_compressed(series_dir/f"trial_{index:02d}.npz",step=[s["step"] for s in samples],center_palm_m=[s["center_palm_m"] for s in samples],displacement_m=[s["displacement_m"] for s in samples],linear_speed_mps=[s["linear_speed_mps"] for s in samples],angular_speed_radps=[s["angular_speed_radps"] for s in samples],inside_B03=[s["inside_B03"] for s in samples],storage_force_n=[s["storage_force_n"] for s in samples],acquisition_force_n=[s["acquisition_force_n"] for s in samples],maximum_penetration_m=[s["maximum_penetration_m"] for s in samples],floor_contact=[s["floor_contact"] for s in samples],qpos=[s["qpos"] for s in samples]); row["timeseries_path"]=str(series_dir/f"trial_{index:02d}.npz"); rows.append(row); print(f"completed B03 hold {index+1}/{len(manifest['trials'])}",flush=True)
    successes=[row for row in rows if row["survival"]["1000"] and not row["gross_overlap"]]; candidate_success=len({r["candidate_id"] for r in successes}); orientation_success=len({r["orientation"]["orientation_id"] for r in successes}); classification="B03-D" if any(r["gross_overlap"] for r in rows) else ("B03-A" if candidate_success>=2 and orientation_success>=2 else ("B03-B" if successes else "B03-C"))
    result={"manifest_sha256":manifest["sha256"],"trial_count":len(rows),"rows":rows,"survival_counts":{str(cp):sum(r["survival"][str(cp)] for r in rows) for cp in load_phase3c10_config()["B03_validation"]["checkpoints"]},"classification":classification,"approved_as_transport_target":classification in ("B03-A","B03-B")}
    (OUTPUT/"B03_validation_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def phase3c10_contract()->dict[str,Any]:
    return {"large_batch":False,"optimizer":False,"rl":False,"object_B":False,"friction_changed":False,"contact_changed":False,"skin_added":False,"world_gravity_changed":False,"joint_limits_changed":False,"actuator_limits_changed":False}
