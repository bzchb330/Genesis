import numpy as np
from seqgrasp.experiments.phase2_6_workspace import accessible_surface_samples, contact_opposition_angle_deg, cylinder_surface_geometry, lexicographic_pose_key, pairwise_envelope_boxes
from seqgrasp.phase2_6_config import load_phase2_6_config

def test_phase2_6_dense_budgets_and_seed_isolation():
    cfg,_=load_phase2_6_config(); assert cfg.workspace.samples_per_finger==200000; assert cfg.workspace.candidate_pose_count==10000; assert cfg.dynamic_search.expanded_candidate_count==8192; assert len({cfg.seeds.workspace,cfg.seeds.candidate_poses,cfg.seeds.calibration_B_namespace,cfg.seeds.formal_v3_B_namespace})==4

def test_cylinder_surface_access_and_opposition_geometry():
    points=np.asarray([[.035,0,0],[-.035,0,0]]); distance,contacts,normals=cylinder_surface_geometry(points,np.zeros(3),.025,.04); assert np.allclose(distance,.01); mask,cps,inward=accessible_surface_samples(points,np.zeros(3),.025,.04,.01,1e-9); assert mask.tolist()==[True,True]; assert len(cps)==2; assert np.isclose(contact_opposition_angle_deg(inward),180)

def test_pairwise_envelope_and_lexicographic_ranking_are_deterministic():
    clouds={"index":np.asarray([[0,0,0],[1,1,1]]),"thumb":np.asarray([[.5,.5,.5],[1.5,1.5,1.5]])}; boxes=pairwise_envelope_boxes(clouds,{"index":.01,"thumb":.01},.02,.04); assert "index+thumb" in boxes
    base={"valid_initial_geometry":True,"accessible_finger_count":2,"opposition_available":True,"ferrari_canny_epsilon":.1,"palm_support_available":False,"predicted_penetration_m":0.,"minimum_joint_margin_rad":.1}; assert lexicographic_pose_key(base)==lexicographic_pose_key(dict(base))

def test_frozen_distribution_is_separate_from_historical_box():
    import yaml
    from seqgrasp.config import ROOT
    frozen=yaml.safe_load((ROOT/"configs"/"phase2_6_frozen_B_distribution.yaml").read_text()); cfg,_=load_phase2_6_config(); assert frozen["experiment_id"]=="phase2_6_formal_v3"; assert frozen["center_bounds_m"]["x"]!=cfg.object_B.old_center_x_bounds_m
