"""Frozen-protocol checks for Phase 3C-1.1 (one test per requested topic)."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import mujoco
import numpy as np
from seqgrasp.config import ROOT
import seqgrasp.phase3c11 as c11

OUT=ROOT/"outputs/phase3C11"
def read(name): return json.loads((OUT/name).read_text(encoding="utf-8"))

def test_exact_compiled_contact_parameter_extraction():
    d=read("compiled_contact_audit.json"); assert d["object"]["condim"]==6 and d["object"]["friction"]==[.5,.01,.003]; assert d["solver"]["timestep_s"]==.002

def test_force_approach_calibration():
    d=read("preload_calibration.json"); assert all(r["isolated_representative_pair"] for r in d["rows"]); assert all(any(r["contact_active"] and r["normal_force_n"]>0 for r in d["rows"] if r["surface"]==s) for s in ("middle","ring","little","palm"))

def test_frozen_calibration_sweep_values():
    d=read("preload_calibration.json"); assert d["principal_sweep_mm"]==[0,.05,.1,.15,.2,.25,.3,.4,.5] and d["explicit_extension_sweep_mm"]==[.75,1.0]

def test_preload_selection_independent_of_retention_outcomes():
    d=read("preload_calibration.json"); assert d["selection_frozen_before_B03_outcomes"] and not d["selection"]["retention_outcomes_inspected"]

def test_original_12_state_identity_preservation():
    d=read("preloaded_B03_manifest.json"); ids=[r["trial_id"] for r in d["rows"]]; assert len(ids)==12 and len(set(ids))==12 and {r["candidate_id"] for r in d["rows"]}=={f"B03_CANDIDATE_0{i}" for i in range(3)}

def test_local_preload_closure():
    d=read("preloaded_B03_manifest.json"); assert all(r["joint_displacement_max_rad"]<=.45+1e-9 for r in d["rows"]); assert sum(r["initializer_feasible"] for r in d["rows"])==0

def test_target_vs_actual_normal_preload():
    rows=read("preloaded_B03_manifest.json")["rows"]; assert all(set(r["target_surfaces"])<=set(r["target_normal_force_n"]) for r in rows); assert all(set(r["target_surfaces"])<=set(r["actual_normal_force_n"]) for r in rows)

def test_original_preloaded_survival_comparison():
    d=read("preloaded_B03_results.json"); assert d["original_survival_counts"]=={"10":9,"25":4,"50":1,"100":0,"200":0,"500":0,"1000":0}; assert not any(d["preloaded_survival_counts"].values()) and d["classification"]=="PR-E"

def test_shape_dimensions():
    s=c11.shape_specifications(); assert s["S0"]["dimensions_m"]=={"diameter":.025}; assert s["S1"]["dimensions_m"]=={"side":.025}; assert s["S2"]["dimensions_m"]=={"diameter":.025,"height":.020}

def test_shape_density_consistent_masses():
    s=c11.shape_specifications(); assert np.isclose(s["S0"]["mass_kg"],1000*4/3*np.pi*.0125**3); assert np.isclose(s["S1"]["mass_kg"],1000*.025**3); assert np.isclose(s["S2"]["mass_kg"],1000*np.pi*.0125**2*.020)

def test_shape_contact_physics_unchanged():
    for sid in ("S0","S1","S2"):
        scene=c11.build_shape_scene(sid) if sid!="S0" else c11.build_forearm_scene(with_actuator=True).scene
        gid=c11._object_geom_id(scene); assert scene.model.geom_friction[gid].tolist()==[.5,.01,.003] and int(scene.model.geom_condim[gid])==6

def test_shape_orientation_handling():
    o=c11.load_phase3c11_config()["shapes"]["orientations"]; assert len(o["S1"])==2 and len(o["S2"])==2; assert all(np.isclose(np.linalg.norm(x["quaternion_wxyz"]),1) for s in o.values() for x in s)

def test_geometric_b03_workspace_allowed_without_dynamic_claim():
    d=read("resource_workspace_audit.json"); assert d["geometric_B03"] and d["geometric_candidate_dynamic_stability_claim"] is False

def test_dynamically_supported_workspace_gate():
    d=read("resource_workspace_audit.json"); assert d["dynamic_workspace_gate"]=={"required_survival_steps":200,"eligible_state_count":0,"status":"NOT_AVAILABLE"} and d["dynamically_supported"] is None

def test_thumb_workspace():
    d=read("resource_workspace_audit.json"); assert d["baseline"]["thumb"]["reachable_volume_m3"]>0 and d["geometric_B03"]["thumb"]["reachable_count"]>0

def test_index_workspace():
    d=read("resource_workspace_audit.json"); assert d["baseline"]["index"]["reachable_volume_m3"]>0 and d["geometric_B03"]["index"]["reachable_count"]>0

def test_thumb_index_opposition_workspace():
    d=read("resource_workspace_audit.json"); assert d["baseline"]["opposition"]["opposition_pair_count"]>0 and d["geometric_B03"]["opposition"]["minimum_aperture_m"]>0

def test_role_mrl_definition():
    assert c11.role_definitions()["ROLE-MRL"]=={"storage":("middle","ring","little","palm"),"preserved":("thumb","index")}

def test_role_t_definition():
    assert c11.role_definitions()["ROLE-T"]=={"storage":("thumb","ring","little","palm"),"preserved":("index","middle")}

def test_thumb_assisted_storage_search():
    d=read("storage_role_mechanics.json")["roles"]["ROLE-T"]; assert d["search_size"]==1728 and d["prefilter_count"]==234 and len(d["selected"])==6

def test_static_wrench_equilibrium():
    source=inspect.getsource(c11.static_wrench_equilibrium); assert "matrix[:3" in source and "matrix[3:" in source and "Coulomb" not in source; assert "friction_utilization" in source

def test_sphere_normal_force_torque_property():
    assert np.allclose(c11.sphere_normal_torque([1,0,0],[0,0,0],[3,0,0]),0)

def test_tangential_friction_torque_contribution():
    assert np.allclose(c11.tangential_contact_torque([1,0,0],[0,0,0],[0,2,0]),[0,0,2])

def test_plus_minus_half_mm_perturbation_generation():
    p=c11.translation_perturbations(); assert p.shape==(6,3) and set(map(tuple,p))=={(.0005,0,0),(-.0005,0,0),(0,.0005,0),(0,-.0005,0),(0,0,.0005),(0,0,-.0005)}

def test_role_specific_resource_preservation():
    d=read("storage_role_mechanics.json")["roles"]; assert set(d["ROLE-MRL"]["preserved_workspace"])=={"thumb","index"}; assert set(d["ROLE-T"]["preserved_workspace"])=={"index","middle"}

def test_no_object_b(): assert c11.phase3c11_contract()["object_B"] is False
def test_no_handoff(): assert c11.phase3c11_contract()["handoff"] is False and "handoff" not in inspect.getsource(c11.run_role_T_holds).lower()
def test_no_optimizer(): assert c11.phase3c11_contract()["optimizer"] is False and c11.phase3c11_contract()["contact_implicit_optimizer"] is False
def test_no_rl(): assert c11.phase3c11_contract()["rl"] is False and "stable_baselines" not in inspect.getsource(c11)

def test_no_contact_physics_changes():
    c=c11.phase3c11_contract(); assert not any(c[x] for x in ("friction_changed","solref_changed","solimp_changed","solver_changed","timestep_changed"))

def test_no_skin(): assert c11.phase3c11_contract()["skin_added"] is False

def test_phase3c10_backward_compatibility():
    d=json.loads((ROOT/"outputs/phase3C10/B03_validation_results.json").read_text()); assert d["survival_counts"]=={"10":9,"25":4,"50":1,"100":0,"200":0,"500":0,"1000":0}; assert d["classification"]=="B03-C"
