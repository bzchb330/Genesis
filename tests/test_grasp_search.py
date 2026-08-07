from dataclasses import replace
import numpy as np
from seqgrasp import load_configs
from seqgrasp.diagnostics import analyze_contact_sequence,run_scripted_grasp
from seqgrasp.diagnostics.grasp_search import candidate_bundle,comparison_plots,diagnostic_metrics,generate_candidates,latin_hypercube,load_search_config,select_distinct

def test_latin_hypercube_and_candidate_generation_are_deterministic_and_bounded():
    first=latin_hypercube(8,3,17); second=latin_hypercube(8,3,17); np.testing.assert_array_equal(first,second)
    for column in first.T: assert set(np.floor(column*8).astype(int))==set(range(8))
    cfg=load_configs(); search=load_search_config(); a=generate_candidates(cfg,search,4); b=generate_candidates(cfg,search,4); assert a==b
    for candidate in a:
        assert all(0<=value<=1 for value in candidate.closed_joint_fractions.values()); assert all(0<=value<=1 for value in candidate.hold_joint_fractions.values()); candidate_bundle(cfg,candidate)

def test_baseline_mechanics_and_engineering_score_remain_descriptive(tmp_path):
    cfg=load_configs(); cfg=replace(cfg,diagnostic=replace(cfg.diagnostic,save_plots=False)); run=run_scripted_grasp(cfg,seed=0,save_outputs=False); search=load_search_config(); metrics=diagnostic_metrics(run,cfg,search); mechanics=analyze_contact_sequence(run,cfg)
    assert metrics["engineering_search_only"] is True; assert metrics["scientific_label_assigned"] is False; assert np.isfinite(metrics["engineering_retention_score"]); assert mechanics["contacting_fingers_immediately_after_release"]
    assert len(comparison_plots(tmp_path,run,run,{"reference":[run]}))==6

def test_distinct_selection_uses_engineering_score_without_labels():
    cfg=load_configs(); search=load_search_config(); candidates=generate_candidates(cfg,search,3); by_id={candidate.candidate_id:candidate for candidate in candidates}; results=[{"candidate_id":candidate.candidate_id,"metrics":{"engineering_retention_score":float(index)}} for index,candidate in enumerate(candidates)]; selected=select_distinct(results,by_id,list(cfg.hand.actuator_names),2,0.0); assert [item["metrics"]["engineering_retention_score"] for item in selected]==[2.0,1.0]
