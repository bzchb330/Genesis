"""Phase 3C-1.1 preload, shape, workspace, and storage-role diagnostics."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares, minimize
from scipy.spatial import ConvexHull
from scipy.stats import qmc
import yaml

from .config import ROOT
from .phase3.config import SUPPORT_SURFACES
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.control import actuator_target_from_qpos
from .phase3.model import build_shadow_scene, set_fixture
from .phase3c0 import object_pose_in_palm, palm_transform, world_to_palm
from .phase3c07 import SPHERE_RADIUS_M, _geom_ids, _object_geom_id, _set_object_palm, contact_geometry, floor_contact, phase3c07_scene_config
from .phase3c08 import FOREARM_JOINT_NAME, _forearm_transform, build_forearm_scene
from .phase3c09 import _finger_interpolation


OUTPUT = ROOT / "outputs/phase3C11"
ROLE_MRL = {"storage": ("middle", "ring", "little", "palm"), "preserved": ("thumb", "index")}
ROLE_T = {"storage": ("thumb", "ring", "little", "palm"), "preserved": ("index", "middle")}


def load_phase3c11_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C11_preload_shape_resource_storage.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def phase3c11_contract() -> dict[str, Any]:
    return {
        "object_B": False, "handoff": False, "optimizer": False, "contact_implicit_optimizer": False,
        "rl": False, "friction_changed": False, "solref_changed": False, "solimp_changed": False,
        "solver_changed": False, "timestep_changed": False, "joint_limits_changed": False,
        "actuator_limits_changed": False, "skin_added": False,
    }


def shape_specifications() -> dict[str, dict[str, Any]]:
    density = float(load_phase3c11_config()["shapes"]["density_kg_m3"])
    radius = .0125
    return {
        "S0": {"shape": "sphere", "size": [radius], "dimensions_m": {"diameter": .025},
               "volume_m3": 4/3*np.pi*radius**3, "mass_kg": density*4/3*np.pi*radius**3},
        "S1": {"shape": "box", "size": [radius, radius, radius], "dimensions_m": {"side": .025},
               "volume_m3": .025**3, "mass_kg": density*.025**3},
        "S2": {"shape": "cylinder", "size": [radius, .010], "dimensions_m": {"diameter": .025, "height": .020},
               "volume_m3": np.pi*radius**2*.020, "mass_kg": density*np.pi*radius**2*.020},
    }


def build_shape_scene(shape_id: str):
    spec = shape_specifications()[shape_id]
    base = phase3c07_scene_config(); raw = dict(base.raw); obj = dict(base.object)
    obj.update({"shape": spec["shape"], "size": list(spec["size"]), "density": load_phase3c11_config()["shapes"]["density_kg_m3"]})
    raw["object"] = obj; cfg = replace(base, raw=raw)
    return build_shadow_scene(cfg, model_transform=_forearm_transform(with_actuator=True))


def _geom_record(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    return {
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "type": int(model.geom_type[geom_id]), "type_name": mujoco.mjtGeom(int(model.geom_type[geom_id])).name,
        "condim": int(model.geom_condim[geom_id]), "friction": model.geom_friction[geom_id].tolist(),
        "margin_m": float(model.geom_margin[geom_id]), "gap_m": float(model.geom_gap[geom_id]),
        "solref": model.geom_solref[geom_id].tolist(), "solimp": model.geom_solimp[geom_id].tolist(),
    }


def compiled_contact_parameter_audit() -> dict[str, Any]:
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene; model = scene.model
    object_geom = _object_geom_id(scene); surfaces = {}
    for surface in SUPPORT_SURFACES:
        surfaces[surface] = [_geom_record(model, geom) for geom in _geom_ids(scene, surface)]
    return {
        "object": _geom_record(model, object_geom), "surfaces": surfaces,
        "pair_combination_note": "object geom priority=1 and condim=6; runtime contact dimensions are measured in calibration rather than inferred",
        "solver": {"timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
                   "cone": int(model.opt.cone), "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
                   "ls_iterations": int(model.opt.ls_iterations), "tolerance": float(model.opt.tolerance),
                   "impratio": float(model.opt.impratio), "gravity_mps2": model.opt.gravity.tolist()},
        "contract": phase3c11_contract(),
    }


def _manifest() -> dict[str, Any]:
    return json.loads((ROOT / "outputs/phase3C10/B03_validation_manifest.json").read_text(encoding="utf-8"))


def _assign_object_world(scene, center_world: np.ndarray, quaternion=(1., 0., 0., 0.)) -> None:
    address = scene.model.jnt_qposadr[scene.object_joint_id]
    scene.data.qpos[address:address+3] = center_world; scene.data.qpos[address+3:address+7] = quaternion
    scene.data.mocap_pos[scene.fixture_mocap_id] = center_world; scene.data.mocap_quat[scene.fixture_mocap_id] = quaternion
    velocity = scene.model.jnt_dofadr[scene.object_joint_id]; scene.data.qvel[velocity:velocity+6] = 0.0
    mujoco.mj_forward(scene.model, scene.data)


def _pair_distance(scene, object_geom: int, surface_geom: int) -> tuple[float, np.ndarray]:
    fromto = np.zeros(6)
    distance = mujoco.mj_geomDistance(scene.model, scene.data, object_geom, surface_geom, .25, fromto)
    return float(distance), fromto


def representative_calibration_pairs() -> dict[str, dict[str, Any]]:
    manifest = _manifest(); wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene; object_geom = _object_geom_id(scene)
    selected = {}
    for surface in SUPPORT_SURFACES:
        best = None
        eligible_geoms = (_geom_ids(scene, surface) if surface == "palm" else tuple(
            mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in scene.fingertip_geoms[surface]
        ))
        for candidate in manifest["candidates"]["selected"]:
            scene.data.qpos[:] = candidate["qpos"]; mujoco.mj_forward(scene.model, scene.data); _set_object_palm(scene, np.asarray(candidate["center_palm_m"]))
            for geom in eligible_geoms:
                distance, fromto = _pair_distance(scene, object_geom, geom)
                row = (distance, candidate, geom, fromto.copy())
                if best is None or distance < best[0]: best = row
        assert best is not None
        calibration_qpos=np.asarray(best[1]["qpos"],dtype=float).copy(); scene.data.qpos[:]=calibration_qpos
        for finger in ("thumb","index","middle","ring","little"):
            if finger != surface: _finger_interpolation(scene,calibration_qpos,finger,0.0)
        calibration_qpos=scene.data.qpos.copy()
        selected[surface] = {"candidate_id": best[1]["candidate_id"], "qpos": calibration_qpos.tolist(),
                             "candidate_center_palm_m": best[1]["center_palm_m"], "surface_geom_id": int(best[2]),
                             "surface_geom_name": mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, best[2]),
                             "candidate_signed_distance_m": float(best[0]),
                             "representative_rule":"compiled fingertip collision geometry for digits; nearest compiled palm collision geom for palm/root",
                             "isolation": "all non-target fingers set to their existing phase3C09 zero-flexion configuration; palm calibration opens all fingers"}
    return selected


def _tangent_setup(scene, pair: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Find a target-pair tangent with positive clearance to every other hand geom."""
    scene.data.qpos[:] = pair["qpos"]; mujoco.mj_forward(scene.model, scene.data)
    object_geom = _object_geom_id(scene); surface_geom = pair["surface_geom_id"]
    geom_center = scene.data.geom_xpos[surface_geom].copy()
    other_geoms = tuple(
        geom for surface in SUPPORT_SURFACES for geom in _geom_ids(scene, surface)
        if geom != surface_geom
    )
    radial_grid = np.linspace(0.0, 0.08, 81); best = None
    # Deterministic Fibonacci sphere directions make the fixture independent of
    # later retention outcomes while avoiding adjacent links/palm patches.
    count = 100
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
        direction = np.asarray([np.sqrt(1-z*z)*np.cos(azimuth), np.sqrt(1-z*z)*np.sin(azimuth), z])
        def signed_distance(radius: float) -> float:
            _assign_object_world(scene, geom_center + direction*radius)
            return _pair_distance(scene, object_geom, surface_geom)[0]
        values = [signed_distance(float(radius)) for radius in radial_grid]
        bracket = next(((float(lo), float(hi)) for lo, hi, vlo, vhi in zip(radial_grid[:-1], radial_grid[1:], values[:-1], values[1:]) if vlo <= 0.0 <= vhi), None)
        if bracket is None:
            continue
        radius = brentq(signed_distance, *bracket, xtol=1e-12)
        tangent = geom_center + direction*radius
        _assign_object_world(scene, tangent + direction*1e-6)
        clearance = min((_pair_distance(scene, object_geom, geom)[0] for geom in other_geoms), default=np.inf)
        candidate = (float(clearance), tangent.copy(), direction.copy())
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None or best[0] <= 0.0:
        raise RuntimeError(f"could not isolate calibration pair {pair['surface_geom_name']}")
    return best[1], -best[2]


def _selected_pair_contact(scene, geom_id: int):
    object_geom = _object_geom_id(scene)
    return next((record for record in extract_shadow_contacts(scene).object_records
                 if {record.geom1_id, record.geom2_id} == {object_geom, geom_id}), None)


def force_approach_calibration() -> dict[str, Any]:
    cfg = load_phase3c11_config()["calibration"]
    principal = [float(value) for value in cfg["principal_approach_mm"]]
    extension = [float(value) for value in cfg["explicit_extension_approach_mm"]]
    approaches = principal + extension
    pairs = representative_calibration_pairs(); rows = []; frames = {}
    for surface, pair in pairs.items():
        frame_scene = build_forearm_scene(with_actuator=True).scene
        frame_scene.data.qpos[:] = pair["qpos"]; mujoco.mj_forward(frame_scene.model, frame_scene.data)
        frames[surface] = _tangent_setup(frame_scene, pair)
    for surface in (*cfg["storage_surfaces"], *cfg["comparison_surfaces"]):
        pair = pairs[surface]
        for approach_mm in approaches:
            wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene; scene.data.qpos[:] = pair["qpos"]
            scene.data.ctrl[:] = actuator_target_from_qpos(scene, scene.data.qpos); set_fixture(scene, True); mujoco.mj_forward(scene.model, scene.data)
            tangent, direction = frames[surface]; center = tangent + direction*(approach_mm/1000.0)
            _assign_object_world(scene, center); scene.data.ctrl[:] = actuator_target_from_qpos(scene, scene.data.qpos); mujoco.mj_forward(scene.model, scene.data)
            distance, _ = _pair_distance(scene, _object_geom_id(scene), pair["surface_geom_id"]); contact = _selected_pair_contact(scene, pair["surface_geom_id"])
            contact_index = None if contact is None else next(i for i in range(scene.data.ncon) if {int(scene.data.contact[i].geom1),int(scene.data.contact[i].geom2)}=={_object_geom_id(scene),pair["surface_geom_id"]})
            runtime_dim = None if contact_index is None else int(scene.data.contact[contact_index].dim)
            runtime_efc_address = None if contact_index is None else int(scene.data.contact[contact_index].efc_address)
            runtime_friction = None if contact_index is None else scene.data.contact[contact_index].friction.tolist()
            runtime_solref = None if contact_index is None else scene.data.contact[contact_index].solref.tolist()
            runtime_solimp = None if contact_index is None else scene.data.contact[contact_index].solimp.tolist()
            object_records=extract_shadow_contacts(scene).object_records
            isolated_pair = all({record.geom1_id, record.geom2_id} == {_object_geom_id(scene), pair["surface_geom_id"]} for record in object_records)
            velocity_address = scene.model.jnt_dofadr[scene.object_joint_id]
            set_fixture(scene, False); mujoco.mj_forward(scene.model, scene.data)
            free_state=extract_shadow_contacts(scene); surface_index=SUPPORT_SURFACES.index(surface); semantic_contact_active=bool(free_state.contact_flags[surface_index])
            initial_force=float(free_state.normal_forces[surface_index]); tangential=float(free_state.tangential_forces[surface_index])
            free_acceleration = scene.data.qacc[velocity_address:velocity_address+3].copy()
            forces=[]
            for _ in range(int(cfg["stability_steps"])):
                mujoco.mj_step(scene.model, scene.data); state=extract_shadow_contacts(scene); forces.append(float(state.normal_forces[surface_index]))
            rows.append({"surface":surface,"surface_geom_name":pair["surface_geom_name"],"candidate_id":pair["candidate_id"],
                         "approach_mm":approach_mm,"sweep_role":"PRINCIPAL" if approach_mm in principal else "EXPLICIT_EXTENSION",
                         "signed_geom_separation_m":distance,"contact_active":semantic_contact_active,
                         "selected_pair_contact_active":contact is not None,
                         "penetration_m":max(0.0,-distance),"normal_force_n":initial_force,"tangential_force_n":tangential,
                         "normal_impulse_n_s":initial_force*float(scene.model.opt.timestep),"runtime_contact_dim":runtime_dim,
                         "runtime_friction":runtime_friction,"runtime_solref":runtime_solref,"runtime_solimp":runtime_solimp,
                         "runtime_efc_address":runtime_efc_address,"object_contact_count":len(object_records),
                         "isolated_representative_pair":isolated_pair,
                         "selected_pair_normal_force_with_fixture_n":0.0 if contact is None else float(contact.normal_force),
                         "total_object_normal_force_with_fixture_n":float(sum(record.normal_force for record in object_records)),
                         "free_sphere_acceleration_mps2":free_acceleration.tolist(),"stability_force_mean_n":float(np.mean(forces)),
                         "stability_force_std_n":float(np.std(forces)),"stability_force_min_n":float(np.min(forces)),
                         "stability_force_max_n":float(np.max(forces))})
    selection = freeze_preload_selection(rows)
    payload={"principal_sweep_mm":principal,"explicit_extension_sweep_mm":extension,"pairs":pairs,"rows":rows,"selection":selection,
             "selection_frozen_before_B03_outcomes":True}
    encoded=json.dumps(selection,sort_keys=True,separators=(",",":")).encode(); payload["selection_sha256"]=hashlib.sha256(encoded).hexdigest()
    return payload


def freeze_preload_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cfg=load_phase3c11_config()["calibration"]["selection"]; preferred=float(cfg["preferred_approach_mm"]); lo,hi=map(float,cfg["region_of_interest_mm"]); zero=float(cfg["numerical_force_zero_n"])
    targets={}
    for surface in load_phase3c11_config()["calibration"]["storage_surfaces"]:
        candidates=[row for row in rows if row["surface"]==surface and lo<=row["approach_mm"]<=hi]
        chosen=min(candidates,key=lambda row:(abs(row["approach_mm"]-preferred),row["approach_mm"]))
        if chosen["normal_force_n"]<=zero:
            active=[row for row in candidates if row["normal_force_n"]>zero]
            if active: chosen=min(active,key=lambda row:row["approach_mm"])
            else:
                extension=[row for row in rows if row["surface"]==surface and row["sweep_role"]=="EXPLICIT_EXTENSION" and row["normal_force_n"]>zero]
                if extension: chosen=min(extension,key=lambda row:row["approach_mm"])
        targets[surface]={"target_approach_mm":chosen["approach_mm"],"target_normal_force_n":chosen["normal_force_n"],
                          "calibration_geom":chosen["surface_geom_name"],"force_above_numerical_zero":chosen["normal_force_n"]>zero}
    return {"rule":cfg["rule"],"targets":targets,"retention_outcomes_inspected":False}


def write_contact_and_calibration_audit() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True,exist_ok=True); audit=compiled_contact_parameter_audit(); calibration=force_approach_calibration()
    (OUTPUT/"compiled_contact_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (OUTPUT/"preload_calibration.json").write_text(json.dumps(calibration,indent=2),encoding="utf-8")
    return {"audit":audit,"calibration":calibration}


def _storage_qpos_addresses(scene,digits:Iterable[str]=("middle","ring","little")) -> np.ndarray:
    return np.concatenate([scene.model.jnt_qposadr[scene.joint_ids[surface]] for surface in digits]).astype(int)


def _set_object_palm_quaternion(scene,center_palm:np.ndarray,quaternion_palm:Iterable[float]) -> None:
    origin,rotation=palm_transform(scene); center_world=origin+rotation@np.asarray(center_palm); palm_quat=np.zeros(4); mujoco.mju_mat2Quat(palm_quat,rotation.reshape(-1)); world_quat=np.zeros(4); mujoco.mju_mulQuat(world_quat,palm_quat,np.asarray(tuple(quaternion_palm),dtype=float)); _assign_object_world(scene,center_world,world_quat)


def _apply_trial_pose(scene, trial: dict[str, Any], qpos: np.ndarray | None=None) -> None:
    scene.data.qpos[:] = np.asarray(trial["qpos"] if qpos is None else qpos,dtype=float)
    for name,key in ((FOREARM_JOINT_NAME,"forearm_PS_rad"),("rh_WRJ1","WRJ1_rad"),("rh_WRJ2","WRJ2_rad")):
        value=trial["orientation"][key]
        if value is not None:
            joint=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,name); scene.data.qpos[scene.model.jnt_qposadr[joint]]=value
    mujoco.mj_forward(scene.model,scene.data); _set_object_palm_quaternion(scene,np.asarray(trial["center_palm_m"]),trial.get("object_quaternion_palm_wxyz",(1,0,0,0))); scene.data.qvel[:]=0.0
    scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); set_fixture(scene,True); mujoco.mj_forward(scene.model,scene.data)


def _surface_distances(scene, surfaces: Iterable[str]) -> dict[str,float]:
    object_geom=_object_geom_id(scene)
    return {surface:float(min(mujoco.mj_geomDistance(scene.model,scene.data,object_geom,geom,.25,None) for geom in _geom_ids(scene,surface))) for surface in surfaces}


def local_preload_closure(trial: dict[str,Any], selection: dict[str,Any],shape_id:str="S0",storage_digits:tuple[str,...]=("middle","ring","little")) -> dict[str,Any]:
    scene=build_forearm_scene(with_actuator=True).scene if shape_id=="S0" else build_shape_scene(shape_id); _apply_trial_pose(scene,trial)
    cfg=load_phase3c11_config()["preload_closure"]; addresses=_storage_qpos_addresses(scene,storage_digits); original=scene.data.qpos.copy(); x0=original[addresses].copy()
    original_distances=_surface_distances(scene,storage_digits); target_surfaces=tuple(sorted(original_distances,key=original_distances.get)[:2])
    target_approach={surface:float(selection["targets"][surface]["target_approach_mm"])/1000 for surface in target_surfaces}
    target_force={surface:float(selection["targets"][surface]["target_normal_force_n"]) for surface in target_surfaces}
    joint_ids=np.concatenate([scene.joint_ids[surface] for surface in storage_digits]); joint_lo=scene.model.jnt_range[joint_ids,0]; joint_hi=scene.model.jnt_range[joint_ids,1]
    delta=float(cfg["maximum_joint_displacement_rad"]); lower=np.maximum(joint_lo,x0-delta); upper=np.minimum(joint_hi,x0+delta)
    def assign(x:np.ndarray,fixture:bool=True):
        scene.data.qpos[:]=original; scene.data.qpos[addresses]=x; scene.data.qvel[:]=0; scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); set_fixture(scene,fixture); mujoco.mj_forward(scene.model,scene.data)
    def geometric_residual(x):
        assign(x,True); distances=_surface_distances(scene,target_surfaces)
        contact=[(distances[surface]+target_approach[surface])/float(cfg["distance_residual_scale_m"]) for surface in target_surfaces]
        regularization=((x-x0)/float(cfg["regularization_scale_rad"])*.02).tolist(); return np.asarray(contact+regularization)
    geometric=least_squares(geometric_residual,x0,bounds=(lower,upper),max_nfev=int(cfg["maximum_function_evaluations"]))
    def full_residual(x):
        assign(x,False); contacts=extract_shadow_contacts(scene); distances=_surface_distances(scene,target_surfaces)
        force=[(contacts.normal_forces[SUPPORT_SURFACES.index(surface)]-target_force[surface])/max(target_force[surface],.05) for surface in target_surfaces]
        geometry=[(distances[surface]+target_approach[surface])/float(cfg["distance_residual_scale_m"]) for surface in target_surfaces]
        regularization=((x-x0)/float(cfg["regularization_scale_rad"])*.01).tolist(); return np.asarray(force+geometry+regularization)
    refined=least_squares(full_residual,geometric.x,bounds=(lower,upper),max_nfev=int(cfg["maximum_function_evaluations"]))
    storage_surfaces=(*storage_digits,"palm"); assign(refined.x,False); contacts=extract_shadow_contacts(scene); distances=_surface_distances(scene,storage_surfaces); forces={surface:float(contacts.normal_forces[SUPPORT_SURFACES.index(surface)]) for surface in storage_surfaces}; tangential={surface:float(contacts.tangential_forces[SUPPORT_SURFACES.index(surface)]) for surface in storage_surfaces}; penetrations={surface:float(contacts.penetration_by_surface[SUPPORT_SURFACES.index(surface)]) for surface in storage_surfaces}
    active=[surface for surface in storage_surfaces if contacts.contact_flags[SUPPORT_SURFACES.index(surface)]]; load=[surface for surface in storage_surfaces if forces[surface]>float(load_phase3c11_config()["calibration"]["selection"]["numerical_force_zero_n"])]
    maximum_target=max(target_approach.values()); excess=max([penetrations[surface]-maximum_target for surface in penetrations],default=0.0)
    feasible=len(load)>=int(cfg["minimum_storage_contacts"]) and excess<=float(cfg["gross_overlap_numerical_tolerance_m"])
    velocity=scene.model.jnt_dofadr[scene.object_joint_id]; acceleration=scene.data.qacc[velocity:velocity+6].copy()
    return {"trial_id":trial["trial_id"],"candidate_id":trial["candidate_id"],"shape_id":shape_id,"storage_digits":list(storage_digits),"object_quaternion_palm_wxyz":trial.get("object_quaternion_palm_wxyz",[1,0,0,0]),"target_surfaces":list(target_surfaces),"target_approach_m":target_approach,"target_normal_force_n":target_force,
            "original_surface_distance_m":original_distances,"qpos":scene.data.qpos.copy().tolist(),"original_qpos":original.tolist(),"storage_joint_addresses":addresses.tolist(),
            "joint_displacement_l2_rad":float(np.linalg.norm(refined.x-x0)),"joint_displacement_max_rad":float(np.max(np.abs(refined.x-x0))),
            "geometric_solver":{"success":bool(geometric.success),"nfev":int(geometric.nfev)},"force_solver":{"success":bool(refined.success),"nfev":int(refined.nfev)},
            "actual_surface_distance_m":distances,"active_storage_contacts":active,"load_bearing_storage_topology":load,"actual_normal_force_n":forces,"actual_tangential_force_n":tangential,
            "penetration_by_surface_m":penetrations,"maximum_penetration_m":float(contacts.maximum_penetration),"sphere_wrench_acceleration":acceleration.tolist(),
            "initializer_feasible":bool(feasible),"failure":None if feasible else "PRELOAD_INITIALIZATION_INFEASIBLE"}


def freeze_preloaded_b03_manifest() -> dict[str,Any]:
    original=_manifest(); calibration=json.loads((OUTPUT/"preload_calibration.json").read_text(encoding="utf-8")); rows=[]
    for index,trial in enumerate(original["trials"]):
        row=local_preload_closure(trial,calibration["selection"]); row["original_manifest_sha256"]=original["sha256"]; rows.append(row); print(f"preload initializer {index+1}/12",flush=True)
    encoded=json.dumps(rows,sort_keys=True,separators=(",",":")).encode(); result={"frozen_before_hold_outcomes":True,"original_manifest_sha256":original["sha256"],"preload_selection_sha256":calibration["selection_sha256"],"trial_count":len(rows),"sha256":hashlib.sha256(encoded).hexdigest(),"rows":rows}
    (OUTPUT/"preloaded_B03_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def run_preloaded_b03_holds() -> dict[str,Any]:
    frozen=json.loads((OUTPUT/"preloaded_B03_manifest.json").read_text(encoding="utf-8")); original=_manifest(); trials={trial["trial_id"]:trial for trial in original["trials"]}; centers=np.asarray(original["candidates"]["B03_centers_palm_m"]); cfg=load_phase3c11_config()["preload_closure"]; rows=[]; series_dir=OUTPUT/"preloaded_B03_series"; series_dir.mkdir(parents=True,exist_ok=True)
    from .phase3c10 import _inside_b03
    for index,preload in enumerate(frozen["rows"]):
        if not preload["initializer_feasible"]:
            rows.append({"trial_id":preload["trial_id"],"initializer_feasible":False,"failure":"PRELOAD_INITIALIZATION_INFEASIBLE","survival":{str(cp):False for cp in cfg["checkpoints"]}}); continue
        scene=build_forearm_scene(with_actuator=True).scene; trial=trials[preload["trial_id"]]; _apply_trial_pose(scene,trial,np.asarray(preload["qpos"])); set_fixture(scene,False); scene.data.qvel[:]=0; scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); mujoco.mj_forward(scene.model,scene.data); initial=object_pose_in_palm(scene,scene.object_body_id)[0]; samples=[]; first_loss=None
        for step in range(1,int(cfg["hold_steps"])+1):
            mujoco.mj_step(scene.model,scene.data); center=object_pose_in_palm(scene,scene.object_body_id)[0]; velocity,angular=object_velocity(scene); state=extract_shadow_contacts(scene); storage_indices=[SUPPORT_SURFACES.index(s) for s in ("middle","ring","little","palm")]; topology=[SUPPORT_SURFACES[i] for i in storage_indices if state.contact_flags[i]]; load=[SUPPORT_SURFACES[i] for i in storage_indices if state.normal_forces[i]>1e-9]
            if first_loss is None and not load:first_loss=step
            samples.append({"step":step,"center_palm_m":center.tolist(),"inside_B03":_inside_b03(center,centers),"displacement_m":float(np.linalg.norm(center-initial)),"linear_speed_mps":float(np.linalg.norm(velocity)),"angular_speed_radps":float(np.linalg.norm(angular)),"storage_topology":topology,"load_topology":load,"normal_forces_n":state.normal_forces.tolist(),"tangential_forces_n":state.tangential_forces.tolist(),"penetration_by_surface_m":state.penetration_by_surface.tolist(),"maximum_penetration_m":state.maximum_penetration,"floor_contact":floor_contact(scene),"qpos":scene.data.qpos.tolist(),"qvel":scene.data.qvel.tolist()})
        survival={str(cp):bool(all(s["inside_B03"] and not s["floor_contact"] and np.all(np.isfinite(s["qpos"])) for s in samples[:int(cp)])) for cp in cfg["checkpoints"]}; escaped=next((s for s in samples if not s["inside_B03"]),None); path=series_dir/f"trial_{index:02d}.npz"
        np.savez_compressed(path,step=[s["step"] for s in samples],center_palm_m=[s["center_palm_m"] for s in samples],inside_B03=[s["inside_B03"] for s in samples],displacement_m=[s["displacement_m"] for s in samples],linear_speed_mps=[s["linear_speed_mps"] for s in samples],angular_speed_radps=[s["angular_speed_radps"] for s in samples],storage_topology_json=[json.dumps(s["storage_topology"]) for s in samples],load_topology_json=[json.dumps(s["load_topology"]) for s in samples],normal_forces_n=[s["normal_forces_n"] for s in samples],tangential_forces_n=[s["tangential_forces_n"] for s in samples],penetration_by_surface_m=[s["penetration_by_surface_m"] for s in samples],maximum_penetration_m=[s["maximum_penetration_m"] for s in samples],floor_contact=[s["floor_contact"] for s in samples],qpos=[s["qpos"] for s in samples],qvel=[s["qvel"] for s in samples])
        rows.append({"trial_id":preload["trial_id"],"candidate_id":preload["candidate_id"],"initializer_feasible":True,"initialization":preload,"survival":survival,"first_contact_loss_step":first_loss,"maximum_displacement_m":max(s["displacement_m"] for s in samples),"maximum_penetration_m":max(s["maximum_penetration_m"] for s in samples),"escape_direction_palm":None if escaped is None else (np.asarray(escaped["center_palm_m"])-initial).tolist(),"timeseries_path":str(path)}); print(f"preloaded hold {index+1}/12",flush=True)
    counts={str(cp):sum(row["survival"][str(cp)] for row in rows) for cp in cfg["checkpoints"]}; original_counts={"10":9,"25":4,"50":1,"100":0,"200":0,"500":0,"1000":0}; feasible=sum(row["initializer_feasible"] for row in rows)
    if feasible==0: classification="PR-E"
    elif counts["1000"]>0: classification="PR-A"
    elif counts["200"]>0: classification="PR-B"
    elif any(counts[key]>original_counts[key] for key in original_counts): classification="PR-C"
    else: classification="PR-D"
    result={"manifest_sha256":frozen["sha256"],"trial_count":len(rows),"feasible_initializers":feasible,"rows":rows,"original_survival_counts":original_counts,"preloaded_survival_counts":counts,"classification":classification,"initialization_materially_affected_B03_C":classification in ("PR-A","PR-B")}
    (OUTPUT/"preloaded_B03_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def _all_hand_collision_geom_ids(scene) -> tuple[int,...]:
    names={name for values in scene.collision_geoms.values() for name in values}
    return tuple(mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in sorted(names))


def freeze_shape_candidates() -> dict[str,Any]:
    cfg=load_phase3c11_config(); original=_manifest(); calibration=json.loads((OUTPUT/"preload_calibration.json").read_text(encoding="utf-8")); c09=json.loads((ROOT/"outputs/phase3C09/phase3c09_results.json").read_text(encoding="utf-8"))
    cloud=np.asarray(original["candidates"]["B03_centers_palm_m"],dtype=float); basin_centers=np.asarray([row["centroid_palm_m"] for row in c09["storage_manifold"]["basins"]]); indices=np.linspace(0,len(cloud)-1,int(cfg["role_search"]["centers_sampled"]),dtype=int); centers=np.unique(np.vstack([cloud[indices],basin_centers]),axis=0); qpos_sources=original["candidates"]["selected"]; result={"frozen_before_dynamic_outcomes":True,"search_centers":len(centers),"shapes":{}}
    for shape_id in ("S1","S2"):
        scene=build_shape_scene(shape_id); object_geom=_object_geom_id(scene); hand_geoms=_all_hand_collision_geom_ids(scene); candidates=[]
        for source in qpos_sources:
            for orientation in cfg["shapes"]["orientations"][shape_id]:
                for center in centers:
                    trial={"qpos":source["qpos"],"center_palm_m":center.tolist(),"orientation":{"forearm_PS_rad":None,"WRJ1_rad":None,"WRJ2_rad":None},"object_quaternion_palm_wxyz":orientation["quaternion_wxyz"]}; _apply_trial_pose(scene,trial)
                    all_dist=[float(mujoco.mj_geomDistance(scene.model,scene.data,object_geom,geom,.25,None)) for geom in hand_geoms]; surface=_surface_distances(scene,("middle","ring","little","palm")); near=sum(value<=float(cfg["role_search"]["fast_near_surface_m"]) for value in surface.values()); clearance=min(all_dist)
                    if clearance>=-float(cfg["preload_closure"]["gross_overlap_numerical_tolerance_m"]) and near>=2:
                        score=sum(sorted(max(value,0.0) for value in surface.values())[:2]); candidates.append({"source_candidate_id":source["candidate_id"],"qpos":source["qpos"],"center_palm_m":center.tolist(),"orientation_id":orientation["id"],"object_quaternion_palm_wxyz":orientation["quaternion_wxyz"],"initial_clearance_m":clearance,"near_surface_count":near,"surface_distance_m":surface,"prefilter_order_quantity_m":score})
        candidates.sort(key=lambda row:(row["prefilter_order_quantity_m"],-row["near_surface_count"],row["orientation_id"],row["center_palm_m"])); selected=[]; used=set()
        for row in candidates:
            key=(row["orientation_id"],tuple(row["center_palm_m"]))
            if key in used:continue
            row={**row,"candidate_id":f"{shape_id}_STORAGE_{len(selected):02d}","trial_id":f"{shape_id}_STORAGE_{len(selected):02d}","orientation":{"forearm_PS_rad":None,"WRJ1_rad":None,"WRJ2_rad":None}}; selected.append(row); used.add(key)
            if len(selected)>=6:break
        initialized=[]
        for row in selected: initialized.append(local_preload_closure(row,calibration["selection"],shape_id))
        result["shapes"][shape_id]={"static_combinations_evaluated":len(qpos_sources)*len(cfg["shapes"]["orientations"][shape_id])*len(centers),"prefilter_passed":len(candidates),"selected":selected,"initialized":initialized}
    encoded=json.dumps(result["shapes"],sort_keys=True,separators=(",",":")).encode(); result["sha256"]=hashlib.sha256(encoded).hexdigest(); (OUTPUT/"shape_candidate_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def run_shape_holds() -> dict[str,Any]:
    manifest=json.loads((OUTPUT/"shape_candidate_manifest.json").read_text(encoding="utf-8")); cfg=load_phase3c11_config()["preload_closure"]; result={"manifest_sha256":manifest["sha256"],"shapes":{}}; series_dir=OUTPUT/"shape_hold_series"; series_dir.mkdir(parents=True,exist_ok=True)
    for shape_id in ("S1","S2"):
        selected={row["candidate_id"]:row for row in manifest["shapes"][shape_id]["selected"]}; rows=[]
        for index,initialized in enumerate(manifest["shapes"][shape_id]["initialized"]):
            if not initialized["initializer_feasible"]:
                rows.append({"trial_id":initialized["trial_id"],"initializer_feasible":False,"failure":"PRELOAD_INITIALIZATION_INFEASIBLE","survival":{str(cp):False for cp in cfg["checkpoints"]}}); continue
            scene=build_shape_scene(shape_id); trial=selected[initialized["candidate_id"]]; _apply_trial_pose(scene,trial,np.asarray(initialized["qpos"])); set_fixture(scene,False); scene.data.qvel[:]=0; scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); mujoco.mj_forward(scene.model,scene.data); initial_center=object_pose_in_palm(scene,scene.object_body_id)[0]; address=scene.model.jnt_qposadr[scene.object_joint_id]; initial_quat=scene.data.qpos[address+3:address+7].copy(); samples=[]
            for step in range(1,int(cfg["hold_steps"])+1):
                mujoco.mj_step(scene.model,scene.data); center=object_pose_in_palm(scene,scene.object_body_id)[0]; velocity,angular=object_velocity(scene); state=extract_shadow_contacts(scene); storage_indices=[SUPPORT_SURFACES.index(s) for s in ("middle","ring","little","palm")]; load=[SUPPORT_SURFACES[i] for i in storage_indices if state.normal_forces[i]>1e-9]; quat=scene.data.qpos[address+3:address+7].copy(); orientation_change=2*np.arccos(np.clip(abs(float(np.dot(initial_quat,quat))),0,1)); displacement=float(np.linalg.norm(center-initial_center)); in_voxel=displacement<=np.sqrt(3)*.0025+1e-12
                samples.append({"step":step,"center_palm_m":center.tolist(),"displacement_m":displacement,"orientation_change_rad":float(orientation_change),"linear_speed_mps":float(np.linalg.norm(velocity)),"angular_speed_radps":float(np.linalg.norm(angular)),"load_topology":load,"normal_forces_n":state.normal_forces.tolist(),"tangential_forces_n":state.tangential_forces.tolist(),"maximum_penetration_m":state.maximum_penetration,"floor_contact":floor_contact(scene),"retained":bool(in_voxel and load and not floor_contact(scene)),"qpos":scene.data.qpos.tolist()})
            survival={str(cp):bool(all(s["retained"] for s in samples[:int(cp)])) for cp in cfg["checkpoints"]}; path=series_dir/f"{shape_id}_{index:02d}.npz"; np.savez_compressed(path,step=[s["step"] for s in samples],center_palm_m=[s["center_palm_m"] for s in samples],displacement_m=[s["displacement_m"] for s in samples],orientation_change_rad=[s["orientation_change_rad"] for s in samples],linear_speed_mps=[s["linear_speed_mps"] for s in samples],angular_speed_radps=[s["angular_speed_radps"] for s in samples],load_topology_json=[json.dumps(s["load_topology"]) for s in samples],normal_forces_n=[s["normal_forces_n"] for s in samples],tangential_forces_n=[s["tangential_forces_n"] for s in samples],maximum_penetration_m=[s["maximum_penetration_m"] for s in samples],floor_contact=[s["floor_contact"] for s in samples],retained=[s["retained"] for s in samples],qpos=[s["qpos"] for s in samples]); rows.append({"trial_id":initialized["trial_id"],"initializer_feasible":True,"initialization":initialized,"survival":survival,"maximum_displacement_m":max(s["displacement_m"] for s in samples),"maximum_orientation_change_rad":max(s["orientation_change_rad"] for s in samples),"maximum_penetration_m":max(s["maximum_penetration_m"] for s in samples),"dominant_load_topology":max((tuple(s["load_topology"]) for s in samples),key=lambda t:sum(tuple(x["load_topology"])==t for x in samples)),"timeseries_path":str(path)}); print(f"shape hold {shape_id} {index+1}/{len(manifest['shapes'][shape_id]['initialized'])}",flush=True)
        result["shapes"][shape_id]={"candidate_count":len(rows),"feasible_initializers":sum(r["initializer_feasible"] for r in rows),"survival_counts":{str(cp):sum(r["survival"][str(cp)] for r in rows) for cp in cfg["checkpoints"]},"rows":rows}
    (OUTPUT/"shape_hold_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def _workspace_sample(scene,base_qpos:np.ndarray,finger:str,samples:np.ndarray,center_palm:np.ndarray|None,storage_surfaces:tuple[str,...]=("middle","ring","little")) -> dict[str,Any]:
    joint_ids=scene.joint_ids[finger]; addresses=scene.model.jnt_qposadr[joint_ids]; lower=scene.model.jnt_range[joint_ids,0]; upper=scene.model.jnt_range[joint_ids,1]; values=lower+samples*(upper-lower); object_geom=_object_geom_id(scene); finger_geoms=_geom_ids(scene,finger); storage_geoms=tuple(geom for surface in storage_surfaces if surface!="palm" for geom in _geom_ids(scene,surface)); body=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,scene.config.hand.fingertip_bodies[finger]); points=[]; orientations=[]; margins=[]; blocked_object=0; blocked_storage=0; blocked_any=0
    for value in values:
        scene.data.qpos[:]=base_qpos; scene.data.qpos[addresses]=value; mujoco.mj_forward(scene.model,scene.data)
        _set_object_palm_quaternion(scene,np.asarray([.3,.3,.3]) if center_palm is None else center_palm,(1,0,0,0))
        object_collision=center_palm is not None and min(float(mujoco.mj_geomDistance(scene.model,scene.data,object_geom,geom,.25,None)) for geom in finger_geoms)<0
        storage_collision=bool(storage_geoms) and min(float(mujoco.mj_geomDistance(scene.model,scene.data,first,second,.25,None)) for first in finger_geoms for second in storage_geoms)<0
        blocked_object+=int(object_collision); blocked_storage+=int(storage_collision); blocked_any+=int(object_collision or storage_collision)
        if object_collision or storage_collision:continue
        origin,rotation=palm_transform(scene); points.append((rotation.T@(scene.data.xpos[body]-origin)).tolist()); tip_rotation=scene.data.xmat[body].reshape(3,3); orientations.append((rotation.T@tip_rotation[:,2]).tolist()); margins.append(float(np.min(np.minimum(value-lower,upper-value))))
    points_array=np.asarray(points,dtype=float).reshape(-1,3); volume=0.0
    if len(points_array)>=4 and np.linalg.matrix_rank(points_array-points_array.mean(axis=0))==3:volume=float(ConvexHull(points_array).volume)
    bounds={"lower":np.min(points_array,axis=0).tolist(),"upper":np.max(points_array,axis=0).tolist()} if len(points_array) else {"lower":None,"upper":None}
    return {"finger":finger,"candidate_count":len(values),"reachable_count":len(points_array),"reachable_points_palm_m":points_array.tolist(),"representative_orientation_axes_palm":orientations,"reachable_volume_m3":volume,"palm_frame_bounds_m":bounds,"minimum_joint_margin_rad":None if not margins else float(min(margins)),"blocked_by_stored_A":blocked_object,"blocked_by_storage_fingers":blocked_storage,"collision_limited_fraction":float(blocked_any/len(values))}


def _opposition_descriptor(thumb:dict[str,Any],index:dict[str,Any],distance_m:float)->dict[str,Any]:
    t=np.asarray(thumb["reachable_points_palm_m"]); i=np.asarray(index["reachable_points_palm_m"])
    if not len(t) or not len(i):return {"pair_count":0,"opposition_pair_count":0,"minimum_aperture_m":None,"maximum_aperture_m":None,"opposition_midpoint_volume_m3":0.0,"representative_pairs":[]}
    distances=np.linalg.norm(t[:,None,:]-i[None,:,:],axis=2); mask=distances<=distance_m; pairs=np.argwhere(mask); midpoints=(t[pairs[:,0]]+i[pairs[:,1]])/2 if len(pairs) else np.empty((0,3)); volume=0.0
    if len(midpoints)>=4 and np.linalg.matrix_rank(midpoints-midpoints.mean(axis=0))==3:volume=float(ConvexHull(midpoints).volume)
    flat=np.argsort(distances,axis=None); representatives=[]
    for flat_index in flat[:min(5,len(flat))]:
        a,b=np.unravel_index(flat_index,distances.shape); representatives.append({"thumb_point_palm_m":t[a].tolist(),"index_point_palm_m":i[b].tolist(),"aperture_m":float(distances[a,b]),"thumb_orientation_axis_palm":thumb["representative_orientation_axes_palm"][a],"index_orientation_axis_palm":index["representative_orientation_axes_palm"][b]})
    return {"pair_count":int(distances.size),"opposition_pair_count":int(np.count_nonzero(mask)),"diagnostic_opposition_distance_m":distance_m,"minimum_aperture_m":float(np.min(distances)),"maximum_aperture_m":float(np.max(distances)),"opposition_midpoint_volume_m3":volume,"representative_pairs":representatives}


def resource_workspace_audit() -> dict[str,Any]:
    cfg=load_phase3c11_config()["workspace"]; manifest=_manifest(); candidate=manifest["candidates"]["selected"][0]; scene=build_forearm_scene(with_actuator=True).scene; base=np.asarray(candidate["qpos"],dtype=float); baseline=base.copy(); scene.data.qpos[:]=baseline
    for finger in ("middle","ring","little"):_finger_interpolation(scene,baseline,finger,0.0)
    baseline=scene.data.qpos.copy(); count=int(cfg["samples_per_finger"]); power=int(np.log2(count)); sample_sets={finger:qmc.Sobol(d=len(scene.joint_ids[finger]),scramble=True,seed=int(cfg["seed"])+(0 if finger=="thumb" else 1)).random_base2(power) for finger in ("thumb","index")}
    results={"baseline":{},"geometric_B03":{},"dynamically_supported":None}
    for finger in ("thumb","index"):
        results["baseline"][finger]=_workspace_sample(scene,baseline,finger,sample_sets[finger],None)
        results["geometric_B03"][finger]=_workspace_sample(scene,base,finger,sample_sets[finger],np.asarray(candidate["center_palm_m"]))
    results["baseline"]["opposition"]=_opposition_descriptor(results["baseline"]["thumb"],results["baseline"]["index"],float(cfg["opposition_distance_m"])); results["geometric_B03"]["opposition"]=_opposition_descriptor(results["geometric_B03"]["thumb"],results["geometric_B03"]["index"],float(cfg["opposition_distance_m"]))
    preloaded=json.loads((OUTPUT/"preloaded_B03_results.json").read_text(encoding="utf-8")); supported=[row for row in preloaded["rows"] if row["initializer_feasible"] and row["survival"]["200"] and row["initialization"]["load_bearing_storage_topology"]]
    results["dynamic_workspace_gate"]={"required_survival_steps":200,"eligible_state_count":len(supported),"status":"AVAILABLE" if supported else "NOT_AVAILABLE"}
    if supported:
        row=supported[0]; qpos=np.load(row["timeseries_path"],allow_pickle=False)["qpos"][199]; center=np.load(row["timeseries_path"],allow_pickle=False)["center_palm_m"][199]; dynamic={finger:_workspace_sample(scene,qpos,finger,sample_sets[finger],center) for finger in ("thumb","index")}; dynamic["opposition"]=_opposition_descriptor(dynamic["thumb"],dynamic["index"],float(cfg["opposition_distance_m"])); dynamic["source_trial_id"]=row["trial_id"]; results["dynamically_supported"]=dynamic
    results["retained_fraction"]={finger:(results["geometric_B03"][finger]["reachable_volume_m3"]/results["baseline"][finger]["reachable_volume_m3"] if results["baseline"][finger]["reachable_volume_m3"]>0 else None) for finger in ("thumb","index")}; base_opp=results["baseline"]["opposition"]["opposition_midpoint_volume_m3"]; results["retained_fraction"]["opposition"]=results["geometric_B03"]["opposition"]["opposition_midpoint_volume_m3"]/base_opp if base_opp>0 else None; results["geometric_candidate_dynamic_stability_claim"]=False; (OUTPUT/"resource_workspace_audit.json").write_text(json.dumps(results,indent=2),encoding="utf-8"); return results


def _selection_with_thumb() -> dict[str,Any]:
    calibration=json.loads((OUTPUT/"preload_calibration.json").read_text(encoding="utf-8")); selection=json.loads(json.dumps(calibration["selection"])); rows=calibration["rows"]
    for surface in ("thumb","index"):
        row=min((r for r in rows if r["surface"]==surface and r["sweep_role"]=="PRINCIPAL"),key=lambda r:(abs(r["approach_mm"]-.2),r["approach_mm"])); selection["targets"][surface]={"target_approach_mm":row["approach_mm"],"target_normal_force_n":row["normal_force_n"],"calibration_geom":row["surface_geom_name"],"force_above_numerical_zero":row["normal_force_n"]>1e-9}
    return selection


def _role_configurations(scene,role_name:str,base_qpos:np.ndarray) -> list[np.ndarray]:
    fractions=load_phase3c11_config()["role_search"]["flexion_fractions"]; digits=("middle","ring","little") if role_name=="ROLE-MRL" else ("thumb","ring","little"); values=[]
    for first in fractions:
     for second in fractions:
      for third in fractions:
       scene.data.qpos[:]=base_qpos
       if role_name=="ROLE-T":
           _finger_interpolation(scene,base_qpos,"index",0.0); _finger_interpolation(scene,base_qpos,"middle",0.0)
       for finger,fraction in zip(digits,(first,second,third)):_finger_interpolation(scene,base_qpos,finger,float(fraction))
       values.append(scene.data.qpos.copy())
    return values


def _surface_for_record(scene,record,allowed:Iterable[str])->str|None:
    names={record.geom1_name,record.geom2_name}
    for surface in allowed:
        if names.intersection(scene.collision_geoms[surface]):return surface
    return None


def static_wrench_equilibrium(scene,storage_surfaces:tuple[str,...]) -> dict[str,Any]:
    set_fixture(scene,False); scene.data.qvel[:]=0; mujoco.mj_forward(scene.model,scene.data); contacts=extract_shadow_contacts(scene); center=scene.data.xpos[scene.object_body_id].copy(); records=[]
    for record in contacts.object_records:
        surface=_surface_for_record(scene,record,storage_surfaces)
        if surface is None:continue
        radial=center-record.position; norm=float(np.linalg.norm(radial))
        if norm<1e-12:continue
        normal=radial/norm; helper=np.asarray([1.,0,0]) if abs(normal[0])<.8 else np.asarray([0,1.,0]); tangent1=np.cross(normal,helper); tangent1/=np.linalg.norm(tangent1); tangent2=np.cross(normal,tangent1); contact_index=next(i for i in range(scene.data.ncon) if {int(scene.data.contact[i].geom1),int(scene.data.contact[i].geom2)}=={record.geom1_id,record.geom2_id}); mu=float(scene.data.contact[contact_index].friction[0]); records.append((surface,record.position.copy(),normal,tangent1,tangent2,mu))
    if not records:return {"feasible":False,"contact_count":0,"failure":"NO_ACTIVE_STORAGE_CONTACT"}
    matrix=np.zeros((6,3*len(records)))
    for index,(_,position,normal,t1,t2,_) in enumerate(records):
        for component,basis in enumerate((normal,t1,t2)):
            matrix[:3,3*index+component]=basis; matrix[3:,3*index+component]=np.cross(position-center,basis)
    mass=float(scene.model.body_mass[scene.object_body_id]); target=-np.r_[mass*scene.model.opt.gravity,np.zeros(3)]; x0=np.zeros(3*len(records)); x0[::3]=mass*np.linalg.norm(scene.model.opt.gravity)/len(records)
    constraints=[{"type":"ineq","fun":lambda x,i=i,mu=record[5]:mu*x[3*i]-np.hypot(x[3*i+1],x[3*i+2])} for i,record in enumerate(records)]; bounds=[(0,20) if i%3==0 else (-20,20) for i in range(3*len(records))]
    solved=minimize(lambda x:float(np.sum((matrix@x-target)**2)+1e-10*np.sum(x*x)),x0,bounds=bounds,constraints=constraints,method="SLSQP",options={"maxiter":500,"ftol":1e-14}); residual=matrix@solved.x-target; cfg=load_phase3c11_config()["role_search"]; feasible=bool(solved.success and np.linalg.norm(residual[:3])<=float(cfg["wrench_force_residual_tolerance_n"]) and np.linalg.norm(residual[3:])<=float(cfg["wrench_torque_residual_tolerance_nm"]))
    forces=[]
    for index,record in enumerate(records):
        n,t1,t2=solved.x[3*index:3*index+3]; forces.append({"surface":record[0],"normal_force_n":float(n),"tangential_components_n":[float(t1),float(t2)],"friction":record[5],"friction_utilization":float(np.hypot(t1,t2)/(record[5]*n)) if n>0 and record[5]>0 else None,"normal_torque_nm":np.cross(record[1]-center,n*record[2]).tolist(),"tangential_torque_nm":np.cross(record[1]-center,t1*record[3]+t2*record[4]).tolist()})
    return {"feasible":feasible,"solver_success":bool(solved.success),"contact_count":len(records),"active_support_topology":sorted(set(r[0] for r in records)),"force_residual_n":residual[:3].tolist(),"torque_residual_nm":residual[3:].tolist(),"forces":forces,"required_normal_force_n":float(sum(f["normal_force_n"] for f in forces)),"maximum_friction_utilization":max((f["friction_utilization"] or 0) for f in forces)}


def storage_role_mechanics_search() -> dict[str,Any]:
    cfg=load_phase3c11_config(); manifest=_manifest(); base=np.asarray(manifest["candidates"]["selected"][0]["qpos"],dtype=float); cloud=np.asarray(manifest["candidates"]["B03_centers_palm_m"]); indices=np.linspace(0,len(cloud)-1,int(cfg["role_search"]["centers_sampled"]),dtype=int); centers=cloud[indices]; selection=_selection_with_thumb(); result={"roles":{},"no_scalar_J":True}
    for role_name,definition in role_definitions().items():
        scene=build_forearm_scene(with_actuator=True).scene; configurations=_role_configurations(scene,role_name,base); storage_digits=tuple(s for s in definition["storage"] if s!="palm"); preserved=definition["preserved"]; candidates=[]; object_geom=_object_geom_id(scene); hand_geoms=_all_hand_collision_geom_ids(scene)
        for config_index,qpos_value in enumerate(configurations):
            for center in centers:
                trial={"qpos":qpos_value.tolist(),"center_palm_m":center.tolist(),"orientation":{"forearm_PS_rad":None,"WRJ1_rad":None,"WRJ2_rad":None}}; _apply_trial_pose(scene,trial); clearance=min(float(mujoco.mj_geomDistance(scene.model,scene.data,object_geom,geom,.25,None)) for geom in hand_geoms); distances=_surface_distances(scene,definition["storage"]); preserved_distances=_surface_distances(scene,preserved); near=sum(value<=float(cfg["role_search"]["fast_near_surface_m"]) for value in distances.values()); free=all(value>=0 for value in preserved_distances.values())
                if clearance>=-float(cfg["preload_closure"]["gross_overlap_numerical_tolerance_m"]) and near>=2 and free:
                    gaps=sorted(max(value,0.0) for value in distances.values())[:2]; candidates.append({"config_index":config_index,"qpos":qpos_value.tolist(),"center_palm_m":center.tolist(),"near_surface_count":near,"two_smallest_surface_gaps_m":gaps,"preserved_surface_distance_m":preserved_distances,"orientation":{"forearm_PS_rad":None,"WRJ1_rad":None,"WRJ2_rad":None}})
        candidates.sort(key=lambda row:(-row["near_surface_count"],row["two_smallest_surface_gaps_m"],row["config_index"],row["center_palm_m"])); selected=[]; used=set()
        for row in candidates:
            key=tuple(row["center_palm_m"])
            if key in used:continue
            identifier=f"{role_name.replace('-','_')}_{len(selected):02d}"; selected.append({**row,"candidate_id":identifier,"trial_id":identifier}); used.add(key)
            if len(selected)>=int(cfg["role_search"]["promising_candidates_per_role"]):break
        initialized=[local_preload_closure(row,selection,"S0",storage_digits) for row in selected]; mechanics=[]
        for row,init in zip(selected,initialized):
            if not init["initializer_feasible"]:mechanics.append({"candidate_id":row["candidate_id"],"nominal":{"feasible":False,"failure":"PRELOAD_INITIALIZATION_INFEASIBLE"},"perturbations":[],"translation_perturbation_feasible_count":0}); continue
            audit_scene=build_forearm_scene(with_actuator=True).scene; _apply_trial_pose(audit_scene,row,np.asarray(init["qpos"])); nominal=static_wrench_equilibrium(audit_scene,definition["storage"]); perturbations=[]
            if nominal["feasible"]:
                for perturb in translation_perturbations(float(cfg["role_search"]["translation_perturbation_m"])):
                    shifted={**row,"trial_id":row["trial_id"]+"_PERTURB","center_palm_m":(np.asarray(row["center_palm_m"])+perturb).tolist()}; shifted_init=local_preload_closure(shifted,selection,"S0",storage_digits)
                    if shifted_init["initializer_feasible"]:
                        shifted_scene=build_forearm_scene(with_actuator=True).scene; _apply_trial_pose(shifted_scene,shifted,np.asarray(shifted_init["qpos"])); equilibrium=static_wrench_equilibrium(shifted_scene,definition["storage"])
                    else:equilibrium={"feasible":False,"failure":"PRELOAD_INITIALIZATION_INFEASIBLE"}
                    perturbations.append({"translation_m":perturb.tolist(),"initializer_feasible":shifted_init["initializer_feasible"],"equilibrium":equilibrium})
            mechanics.append({"candidate_id":row["candidate_id"],"nominal":nominal,"perturbations":perturbations,"translation_perturbation_feasible_count":sum(p["equilibrium"]["feasible"] for p in perturbations)})
        # Role-specific preserved-resource workspace is descriptive and uses the first frozen geometry.
        role_workspace={}
        if selected:
            work_scene=build_forearm_scene(with_actuator=True).scene; work_qpos=np.asarray(initialized[0]["qpos"] if initialized[0]["initializer_feasible"] else selected[0]["qpos"]); work_center=np.asarray(selected[0]["center_palm_m"])
            for offset,finger in enumerate(preserved):
                samples=qmc.Sobol(d=len(work_scene.joint_ids[finger]),scramble=True,seed=int(cfg["workspace"]["seed"])+20+offset).random_base2(int(np.log2(cfg["workspace"]["samples_per_finger"])))
                role_workspace[finger]=_workspace_sample(work_scene,work_qpos,finger,samples,work_center,storage_digits)
        result["roles"][role_name]={"definition":definition,"search_size":len(configurations)*len(centers),"prefilter_count":len(candidates),"selected":selected,"initialized":initialized,"mechanics":mechanics,"mechanically_feasible_count":sum(m["nominal"]["feasible"] for m in mechanics),"disturbance_robust_count":sum(m["translation_perturbation_feasible_count"]==6 for m in mechanics),"preserved_workspace":role_workspace}
    encoded=json.dumps(result["roles"],sort_keys=True,separators=(",",":")).encode(); result["sha256"]=hashlib.sha256(encoded).hexdigest(); (OUTPUT/"storage_role_mechanics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def run_role_T_holds() -> dict[str,Any]:
    mechanics=json.loads((OUTPUT/"storage_role_mechanics.json").read_text(encoding="utf-8")); role=mechanics["roles"]["ROLE-T"]; robust={m["candidate_id"] for m in role["mechanics"] if m["translation_perturbation_feasible_count"]==6}; chosen=[(row,init) for row,init in zip(role["selected"],role["initialized"]) if row["candidate_id"] in robust][:int(load_phase3c11_config()["role_search"]["maximum_role_T_dynamic_trials"])]
    cfg=load_phase3c11_config()["preload_closure"]; rows=[]; series_dir=OUTPUT/"role_T_hold_series"; series_dir.mkdir(parents=True,exist_ok=True)
    for index,(trial,init) in enumerate(chosen):
        scene=build_forearm_scene(with_actuator=True).scene; _apply_trial_pose(scene,trial,np.asarray(init["qpos"])); set_fixture(scene,False); scene.data.qvel[:]=0; scene.data.ctrl[:]=actuator_target_from_qpos(scene,scene.data.qpos); mujoco.mj_forward(scene.model,scene.data); initial=object_pose_in_palm(scene,scene.object_body_id)[0]; samples=[]
        for step in range(1,1001):
            mujoco.mj_step(scene.model,scene.data); center=object_pose_in_palm(scene,scene.object_body_id)[0]; state=extract_shadow_contacts(scene); load=[surface for surface in ROLE_T["storage"] if state.normal_forces[SUPPORT_SURFACES.index(surface)]>1e-9]; displacement=float(np.linalg.norm(center-initial)); retained=displacement<=np.sqrt(3)*.0025+1e-12 and bool(load) and not floor_contact(scene); samples.append({"step":step,"retained":retained,"center_palm_m":center.tolist(),"load_topology":load,"normal_forces_n":state.normal_forces.tolist(),"maximum_penetration_m":state.maximum_penetration,"qpos":scene.data.qpos.tolist()})
        survival={str(cp):all(s["retained"] for s in samples[:cp]) for cp in cfg["checkpoints"]}; path=series_dir/f"trial_{index:02d}.npz"; np.savez_compressed(path,step=[s["step"] for s in samples],retained=[s["retained"] for s in samples],center_palm_m=[s["center_palm_m"] for s in samples],load_topology_json=[json.dumps(s["load_topology"]) for s in samples],normal_forces_n=[s["normal_forces_n"] for s in samples],maximum_penetration_m=[s["maximum_penetration_m"] for s in samples],qpos=[s["qpos"] for s in samples]); rows.append({"trial_id":trial["trial_id"],"survival":survival,"timeseries_path":str(path)})
    result={"frozen_candidate_ids":[row[0]["candidate_id"] for row in chosen],"selection_rule":"nominal wrench feasible and all six predeclared translation perturbations feasible","dynamic_trial_count":len(rows),"survival_counts":{str(cp):sum(r["survival"][str(cp)] for r in rows) for cp in cfg["checkpoints"]},"rows":rows}; (OUTPUT/"role_T_hold_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def role_definitions() -> dict[str, Any]: return {"ROLE-MRL": ROLE_MRL, "ROLE-T": ROLE_T}


def translation_perturbations(distance_m: float=.0005) -> np.ndarray:
    return np.asarray([[distance_m,0,0],[-distance_m,0,0],[0,distance_m,0],[0,-distance_m,0],[0,0,distance_m],[0,0,-distance_m]])


def sphere_normal_torque(contact_position: Iterable[float], center: Iterable[float], normal_force: Iterable[float]) -> np.ndarray:
    return np.cross(np.asarray(contact_position)-np.asarray(center),np.asarray(normal_force))


def tangential_contact_torque(contact_position: Iterable[float], center: Iterable[float], tangential_force: Iterable[float]) -> np.ndarray:
    return np.cross(np.asarray(contact_position)-np.asarray(center),np.asarray(tangential_force))
