from dataclasses import replace
from seqgrasp import load_configs
from seqgrasp.diagnostics import aggregate_summaries,plot_aggregate,run_scripted_grasp,summarize_run

def test_characterization_is_descriptive_and_generates_eight_aggregate_plots(tmp_path):
    cfg=load_configs(); cfg=replace(cfg,diagnostic=replace(cfg.diagnostic,save_plots=False)); runs=[run_scripted_grasp(cfg,seed=seed,save_outputs=False) for seed in range(2)]; summaries=[summarize_run(run,cfg) for run in runs]; aggregate=aggregate_summaries(runs,summaries)
    assert aggregate["scientific_labels_assigned"] is False; assert aggregate["run_count"]==2; assert all(s["support_release_time_s"]>0 for s in summaries)
    plot_aggregate(runs,tmp_path); assert len(list(tmp_path.glob("*.png")))==8
