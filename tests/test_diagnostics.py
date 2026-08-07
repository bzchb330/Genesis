from dataclasses import replace
import json
import numpy as np
from seqgrasp import load_configs
from seqgrasp.diagnostics import run_scripted_grasp

def test_scripted_diagnostic_is_deterministic_and_writes_raw_outputs(tmp_path):
    cfg=load_configs(); diag=replace(cfg.diagnostic,output_dir=str(tmp_path),save_plots=True,render_video=False); cfg=replace(cfg,diagnostic=diag)
    first=run_scripted_grasp(cfg,seed=5,output_dir=tmp_path/"first"); second=run_scripted_grasp(cfg,seed=5,output_dir=tmp_path/"second",save_outputs=False)
    for key in first.arrays: np.testing.assert_array_equal(first.arrays[key],second.arrays[key])
    assert first.metadata["maximum_raw_finger_normal_force_N"]>0
    assert np.any(first.arrays["tactile_contact_flags"]>0); assert np.any(first.arrays["tactile_normal_force"]>0)
    assert first.metadata["support_release_event"] is not None; assert np.any(first.arrays["support_active"]==0); assert first.arrays["joint_limits"].shape[1:]==(cfg.hand.dof_count,2)
    assert (tmp_path/"first"/"timesteps.csv").is_file(); assert (tmp_path/"first"/"timesteps.npz").is_file(); assert (tmp_path/"first"/"resource_state.npz").is_file(); assert json.loads((tmp_path/"first"/"metadata.json").read_text())["scientific_success_assigned"] is False
    with np.load(tmp_path/"first"/"resource_state.npz") as resource:
        assert {"active_fingers","active_finger_count","contact_count","active_contact_count","finger_contact_count"}<=set(resource.files)
    expected={"object_a_height.png","object_a_vertical_velocity.png","object_a_displacement.png","object_a_orientation_change.png","total_contact_force.png","normal_force_per_finger.png","binary_finger_contact.png","active_finger_count.png","number_of_contacts.png","actuator_commands.png","joint_positions.png","episode_phase.png"}
    assert {p.name for p in (tmp_path/"first"/"plots").glob("*.png")}==expected

def test_named_diagnostic_profile_is_selectable_without_changing_physics():
    cfg=load_configs(); profile=cfg.diagnostic.profiles[cfg.diagnostic.active_profile]; profiles=dict(cfg.diagnostic.profiles); profiles["reference_copy"]=profile; cfg=replace(cfg,diagnostic=replace(cfg.diagnostic,profiles=profiles,save_plots=False))
    active=run_scripted_grasp(cfg,seed=2,save_outputs=False); copy=run_scripted_grasp(cfg,seed=2,profile_name="reference_copy",save_outputs=False)
    for key in active.arrays: np.testing.assert_array_equal(active.arrays[key],copy.arrays[key])
