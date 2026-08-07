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
    assert (tmp_path/"first"/"timesteps.csv").is_file(); assert (tmp_path/"first"/"timesteps.npz").is_file(); assert json.loads((tmp_path/"first"/"metadata.json").read_text())["scientific_success_assigned"] is False
    expected={"object_a_height.png","total_contact_force.png","normal_force_per_finger.png","number_of_contacts.png","actuator_commands.png","joint_positions.png","episode_phase.png"}
    assert {p.name for p in (tmp_path/"first"/"plots").glob("*.png")}==expected
