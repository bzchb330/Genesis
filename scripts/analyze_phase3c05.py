"""Analyze the frozen Phase 3C-0.5 experiment and create vector figures."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from seqgrasp.config import ROOT
from seqgrasp.phase3.config import SUPPORT_SURFACES


FIGURE_NAMES = (
    "phase3C0_failure_timeline.pdf",
    "coordinated_capture_concept.pdf",
    "acquisition_vs_alternate_load.pdf",
    "storage_subset_comparison.pdf",
    "fixed_vs_wrist_assisted_capture.pdf",
    "gravity_in_palm_during_capture.pdf",
    "wrist_pose_vs_support.pdf",
    "support_topology_timeline.pdf",
    "load_share_gate_analysis.pdf",
    "release_ramp_analysis.pdf",
    "thumb_first_vs_index_first.pdf",
    "post_release_survival.pdf",
    "serial_vs_simultaneous_capture.pdf",
    "serial_vs_wrist_coordinated_capture.pdf",
    "corridor_metric_audit.pdf",
    "representative_success_sequence.pdf",
)

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array):
        return {"count": 0, "minimum": None, "median": None, "mean": None,
                "p95": None, "p99": None, "maximum": None}
    return {
        "count": int(len(array)), "minimum": float(array.min()),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def _rate(rows: list[dict], key: str) -> dict[str, float | int]:
    successes = sum(bool(row[key]) for row in rows)
    return {"successes": successes, "N": len(rows), "rate": successes / len(rows) if rows else 0.0}


def _paired(control: list[dict], treatment: list[dict], key: str) -> dict[str, Any]:
    identity = lambda row: (row["state_id"], tuple(row["subset"]))
    left = {identity(row): bool(row[key]) for row in control}
    right = {identity(row): bool(row[key]) for row in treatment}
    if set(left) != set(right):
        raise ValueError("paired condition identities differ")
    treatment_only = sum(not left[item] and right[item] for item in left)
    control_only = sum(left[item] and not right[item] for item in left)
    discordant = treatment_only + control_only
    p = 1.0 if discordant == 0 else float(
        binomtest(min(treatment_only, control_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "N_pairs": len(left), "control_successes": sum(left.values()),
        "treatment_successes": sum(right.values()),
        "treatment_only": treatment_only, "control_only": control_only,
        "matched_risk_difference": (sum(right.values()) - sum(left.values())) / len(left),
        "exact_McNemar_p": p,
    }


def _series(row: dict) -> dict[str, np.ndarray]:
    with np.load(row["timeseries_path"]) as data:
        return {name: data[name].copy() for name in data.files}


def _style(axis: plt.Axes, title: str, subtitle: str = "") -> None:
    axis.set_title(title, loc="left", fontsize=13, weight="bold", pad=24)
    if subtitle:
        axis.text(0, 1.01, subtitle, transform=axis.transAxes, fontsize=8, color="#4b5563")
    axis.grid(alpha=0.18)


def _save(directory: Path, name: str, draw) -> None:
    figure = plt.figure(figsize=(9.4, 5.6), constrained_layout=True)
    draw(figure)
    figure.savefig(directory / name, format="pdf", bbox_inches="tight")
    plt.close(figure)


def analyze() -> dict[str, Any]:
    output = ROOT / "outputs/phase3C05"
    result = json.loads((output / "phase3c05_results.json").read_text(encoding="utf-8"))
    failure = json.loads((output / "failure_handoff_audit.json").read_text(encoding="utf-8"))
    corridor = json.loads((output / "corridor_metric_audit.json").read_text(encoding="utf-8"))
    rows = result["capture_trials"]
    releases = result["release_trials"]
    serial = [row for row in rows if row["strategy"] == "C05-SERIAL"]
    simultaneous = [row for row in rows if row["strategy"] == "C05-SIMULTANEOUS"]
    w1 = [row for row in rows if row["strategy"] == "C05-WRIST"
          and max(abs(value) for value in row["wrist_delta_command_deg"]) == 5.0]
    w2 = [row for row in rows if row["strategy"] == "C05-WRIST"
          and max(abs(value) for value in row["wrist_delta_command_deg"]) == 10.0]
    load = [row for row in rows if row["strategy"] == "C05-WRIST-LOAD-TRANSFER"]
    subsets = sorted({tuple(row["subset"]) for row in serial}, key=lambda x: (len(x), x))

    subset_summary = {}
    for subset in subsets:
        label = "+".join(subset)
        subset_summary[label] = {}
        for name, condition in (("serial", serial), ("simultaneous", simultaneous)):
            selected = [row for row in condition if tuple(row["subset"]) == subset]
            subset_summary[label][name] = {
                "alternate_support_10pct_25step": {
                    "successes": sum(row["persistence_first_reached"]["0.10/25"] is not None for row in selected),
                    "N": len(selected),
                },
                "coordinated_capture": _rate(selected, "coordinated_capture"),
                "A_retained": _rate(selected, "A_retained"),
                "storage_contact": _rate(selected, "storage_finger_contact"),
                "palm_contact": _rate(selected, "palm_contact"),
            }

    gate_summary = {}
    conditions = {"serial": serial, "simultaneous": simultaneous, "W1": w1, "W2_conditional": w2,
                  "wrist_load_transfer": load}
    for name, condition in conditions.items():
        gate_summary[name] = {
            gate: {"successes": sum(row["persistence_first_reached"][gate] is not None for row in condition),
                   "N": len(condition)}
            for gate in ("0.10/10", "0.10/25", "0.10/50", "0.25/10", "0.25/25",
                         "0.25/50", "0.50/10", "0.50/25", "0.50/50")
        }

    wrist_by_command = {}
    for command in ((-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)):
        selected = [row for row in w1 if tuple(row["wrist_delta_command_deg"]) == command]
        wrist_by_command[str(list(command))] = {
            "coordinated_capture": _rate(selected, "coordinated_capture"),
            "A_retained": _rate(selected, "A_retained"),
            "palm_contact": _rate(selected, "palm_contact"),
            "storage_contact": _rate(selected, "storage_finger_contact"),
            "paired_vs_simultaneous_coordinated_capture": _paired(simultaneous, selected, "coordinated_capture"),
        }

    gravity_by_strategy: dict[str, list[np.ndarray]] = defaultdict(list)
    topology = Counter()
    for row in rows:
        arrays = _series(row)
        gravity_by_strategy[row["strategy"]].append(arrays["gravity_in_palm_frame"][::10])
        for flags in arrays["contact_flags"]:
            topology["{" + ",".join(name for name, active in zip(SUPPORT_SURFACES, flags) if active) + "}"] += 1
    gravity_summary = {
        strategy: {
            axis: distribution(np.concatenate(values)[:, index])
            for index, axis in enumerate(("palm_x_mps2", "palm_y_mps2", "palm_z_mps2"))
        }
        for strategy, values in gravity_by_strategy.items()
    }

    executed = [row for row in releases if row.get("executed")]
    recovered = [row for row in executed if row.get("one_resource_recovered")]
    release_by_finger = {
        finger: _rate([row for row in executed if row["finger"] == finger], "one_resource_recovered")
        for finger in ("thumb", "index")
    }
    release_by_ramp = {
        str(ramp): _rate([row for row in executed if row["ramp_steps"] == ramp], "one_resource_recovered")
        for ramp in (25, 50, 100, 200)
    }
    survival = {
        checkpoint: {
            "successes": sum(bool(row["survival"][checkpoint]) for row in executed), "N": len(executed),
            "valid_recovery_successes": sum(bool(row["survival"][checkpoint]) for row in recovered),
            "valid_recovery_N": len(recovered),
        }
        for checkpoint in ("10", "25", "50", "100", "200", "300", "500", "750", "1000")
    }

    successful_capture = [row for row in rows if row["coordinated_capture"]]
    wrist_success = [row for row in successful_capture if row["strategy"] in {"C05-WRIST", "C05-WRIST-LOAD-TRANSFER"}]
    successful_topology = Counter()
    failed_topology = Counter()
    for row in rows:
        flags = _series(row)["contact_flags"]
        target = successful_topology if row["coordinated_capture"] else failed_topology
        for item in flags:
            target["{" + ",".join(name for name, active in zip(SUPPORT_SURFACES, item) if active) + "}"] += 1

    summary = {
        "phase": "3C-0.5",
        "branch": "codex/phase3C05-coordinated-palmar-capture",
        "base_commit": "b20cf473dd9bf524128c3a212626162caee27e7f",
        "matched_initial_state_N": result["matched_state_count"],
        "frozen_state_ids": result["matched_state_ids_frozen_before_formal_conditions"],
        "failure_handoff": {
            "N": len(failure["trials"]),
            "storage_entry_steps": [row["storage_entry_step"] for row in failure["trials"]],
            "acquisition_support_loss_steps": [row["first_acquisition_support_loss_step"] for row in failure["trials"]],
            "A_loss_steps": [row["A_loss_step"] for row in failure["trials"]],
            "commanded_release_start_steps": [row["acquisition_unloading_start_step"] for row in failure["trials"]],
            "loss_before_commanded_release_count": sum(not row["controlled_release_started_before_A_loss"] for row in failure["trials"]),
            "storage_contact_before_loss_count": sum(
                row["first_storage_finger_contact_step"] is not None and row["first_storage_finger_contact_step"] < row["A_loss_step"]
                for row in failure["trials"]
            ),
            "diagnosis": "Acquisition contact disappeared after storage entry and before any alternate support; A hit the floor before commanded release in all six trials.",
        },
        "storage_subsets": [list(value) for value in subsets],
        "subset_summary": subset_summary,
        "capture_success_definition": "A retained at capture endpoint and lambda_alt >= 0.10 persisted for 25 consecutive steps (engineering diagnostic)",
        "serial": _rate(serial, "coordinated_capture"),
        "simultaneous": _rate(simultaneous, "coordinated_capture"),
        "wrist_W1": _rate(w1, "coordinated_capture"),
        "wrist_W2_conditional": _rate(w2, "coordinated_capture"),
        "wrist_load_transfer": _rate(load, "coordinated_capture"),
        "serial_vs_simultaneous": _paired(serial, simultaneous, "coordinated_capture"),
        "wrist_by_command": wrist_by_command,
        "wrist_ranges_exercised_deg": {"W0": [0.0], "W1": [-5.0, 5.0], "W2_conditional": [-10.0, 10.0]},
        "actual_wrist_motion_successful_trials_deg": {
            "WRJ2": distribution(row["actual_wrist_motion_deg"][0] for row in wrist_success),
            "WRJ1": distribution(row["actual_wrist_motion_deg"][1] for row in wrist_success),
            "vector_magnitude": distribution(np.linalg.norm(row["actual_wrist_motion_deg"]) for row in wrist_success),
        },
        "gravity_in_palm": gravity_summary,
        "palm_contact_all_capture": _rate(rows, "palm_contact"),
        "storage_finger_contact_all_capture": _rate(rows, "storage_finger_contact"),
        "maximum_alternate_load_fraction": max(row["maximum_alternate_fraction"] for row in rows),
        "load_share_gate_results": gate_summary,
        "release": {
            "candidate_conditions": len(releases), "executed": len(executed),
            "support_gate_not_reached": len(releases) - len(executed),
            "by_finger": release_by_finger, "by_ramp": release_by_ramp,
            "overall_executed": _rate(executed, "one_resource_recovered"),
            "unique_recovered_states": sorted({row["state_id"] for row in recovered}),
            "unique_recovered_state_count": len({row["state_id"] for row in recovered}),
            "unique_recovered_states_over_matched_N": len({row["state_id"] for row in recovered}) / result["matched_state_count"],
            "both_resources_recovered": {"successes": 0, "N": 0, "rate": None, "reason": result["optional_second_release"]["reason"]},
            "post_release_survival": survival,
            "available_motion_rad": distribution(row["released_finger_available_motion_raw"] for row in executed),
            "all_valid_recoveries_retained_through_ramp": all(row["retained_A_during_ramp"] for row in recovered),
            "all_valid_recoveries_fixture_free": all(not row["fixture_active"] for row in recovered),
        },
        "successful_support_topologies_by_sample": successful_topology.most_common(15),
        "failed_support_topologies_by_sample": failed_topology.most_common(15),
        "penetration_m": distribution(row["maximum_penetration_m"] for row in rows),
        "successful_recovery_capture_penetration_m": distribution(
            next(item["maximum_penetration_m"] for item in load if item["state_id"] == state)
            for state in sorted({row["state_id"] for row in recovered})
        ),
        "minimum_joint_margin_rad": distribution(row["minimum_joint_margin_rad"] for row in rows),
        "actuator_clipping_count": distribution(row["maximum_actuator_clipping_count"] for row in rows),
        "corridor_metric_audit": corridor,
        "physics_changed": result["physics_changed"], "object_B_instantiated": result["object_B_instantiated"],
        "rl_training_performed": result["rl_training_performed"],
        "classification": "CC-A",
        "classification_basis": "One thumb resource was recovered in three matched states with fixture-free, floor-free retention through the ramp and all post-release checkpoints to 1000 steps.",
        "progression_decision": "PI_REVIEW_REQUIRED",
        "progression_reason": "The mechanism is demonstrated across multiple states, but the predefined progression text leaves 'penetration remains sane' and sufficient reproducibility without a hard threshold for PI judgment.",
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"result": result, "failure": failure, "corridor": corridor, "summary": summary,
            "rows": rows, "serial": serial, "simultaneous": simultaneous, "w1": w1,
            "w2": w2, "load": load, "executed": executed, "recovered": recovered}


def create_figures(data: dict[str, Any]) -> list[str]:
    figure_dir = ROOT / "docs/figures/phase3C05"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = data["summary"]
    failure = data["failure"]["trials"]
    serial, simultaneous, w1, w2 = data["serial"], data["simultaneous"], data["w1"], data["w2"]
    executed, recovered = data["executed"], data["recovered"]

    def fig1(fig):
        ax = fig.add_subplot(111); y = np.arange(len(failure))
        for i, row in enumerate(failure):
            ax.plot([row["storage_entry_step"], row["A_loss_step"]], [i, i], lw=6, color=COLORS[0], alpha=.7)
            ax.scatter(row["first_acquisition_support_loss_step"], i, marker="x", s=70, color=COLORS[4])
            ax.scatter(row["acquisition_unloading_start_step"], i, marker="|", s=170, color="black")
        ax.set(yticks=y, yticklabels=[f"C0 trial {i+1}" for i in y], xlabel="simulation step")
        ax.legend(handles=[plt.Line2D([],[],color=COLORS[0],lw=6,label="storage entry to A floor loss"),
                           plt.Line2D([],[],marker="x",ls="",color=COLORS[4],label="acquisition support lost"),
                           plt.Line2D([],[],marker="|",ls="",color="black",label="commanded release start")])
        _style(ax, "Phase 3C-0 failure timeline", "A was lost before commanded release in 6/6 trials")

    def fig2(fig):
        ax = fig.add_subplot(111); ax.axis("off")
        stages = [("u_acquisition(t)", .2, .72, COLORS[0]), ("u_storage(t)", .5, .72, COLORS[2]),
                  ("u_wrist(t)", .8, .72, COLORS[1]), ("COORDINATED\nCAPTURE", .5, .42, "#E5E7EB"),
                  ("one-finger\nrelease", .5, .14, "#FDE68A")]
        for label,x,y,color in stages: ax.text(x,y,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc=color,alpha=.35))
        for x in (.2,.5,.8): ax.annotate("",xy=(.5,.49),xytext=(x,.65),arrowprops=dict(arrowstyle="->",lw=1.7))
        ax.annotate("",xy=(.5,.23),xytext=(.5,.34),arrowprops=dict(arrowstyle="->",lw=1.7))
        ax.text(.5,.94,"Concurrent command channels",ha="center",fontsize=15,weight="bold")
        ax.text(.5,.03,"Release is gated by measured alternate support and persistence; first contact alone is insufficient.",ha="center",fontsize=9)

    def representative_release():
        preferred = next(row for row in recovered if row["state_id"] == "C05_STATE_00021" and row["finger"] == "thumb" and row["ramp_steps"] == 100)
        return preferred, _series(preferred)

    def fig3(fig):
        row, arrays = representative_release(); ax = fig.add_subplot(111)
        forces = arrays["normal_forces_n"]; steps = arrays["step"]
        ax.plot(steps, forces[:,:2].sum(1), label="thumb + index", color=COLORS[0])
        ax.plot(steps, forces[:,2:].sum(1), label="middle + ring + little + palm", color=COLORS[1])
        ax.set(xlabel="release-ramp step", ylabel="normal force (N)"); ax.legend()
        _style(ax, "Acquisition vs alternate load", f"Actual successful {row['finger']}-first replay, {row['ramp_steps']}-step ramp")

    def fig4(fig):
        ax = fig.add_subplot(111); labels=list(summary["subset_summary"]); x=np.arange(len(labels)); width=.36
        for j,condition in enumerate(("serial","simultaneous")):
            vals=[summary["subset_summary"][label][condition]["coordinated_capture"]["rate"] for label in labels]
            ax.bar(x+(j-.5)*width, vals, width, label=condition, color=COLORS[j])
        ax.set(xticks=x,xticklabels=labels,ylabel="capture fraction",ylim=(0,.25)); ax.tick_params(axis="x",rotation=24); ax.legend()
        _style(ax,"Storage subset comparison","Endpoint retention plus 10% alternate load persisted 25 steps; N=50 per bar")

    def fig5(fig):
        ax=fig.add_subplot(111); labels=["fixed"]+list(summary["wrist_by_command"]); vals=[summary["simultaneous"]["rate"]]+[summary["wrist_by_command"][k]["coordinated_capture"]["rate"] for k in summary["wrist_by_command"]]
        ax.bar(np.arange(len(vals)),vals,color=[COLORS[0],*([COLORS[1]]*4)]); ax.set(xticks=np.arange(len(vals)),xticklabels=labels,ylabel="capture fraction",ylim=(0,.2)); ax.tick_params(axis="x",rotation=20)
        _style(ax,"Fixed vs wrist-assisted capture","Four predeclared W1 diagonal commands; 350 matched state-subset pairs each")

    def fig6(fig):
        ax=fig.add_subplot(111); values=[]; labels=[]
        for strategy, rows in (("fixed",simultaneous),("W1",w1),("W2 conditional",w2)):
            blocks=[_series(row)["gravity_in_palm_frame"][::10] for row in rows]
            values.append(np.concatenate(blocks)[:,2]); labels.append(strategy)
        ax.boxplot(values,tick_labels=labels,showfliers=False); ax.set_ylabel("palm-frame gravity z (m/s2)")
        _style(ax,"Gravity in palm during capture","Descriptive component only; world gravity remained unchanged")

    def fig7(fig):
        ax=fig.add_subplot(111); mags=np.asarray([np.linalg.norm(row["actual_wrist_motion_deg"]) for row in w1]); support=np.asarray([row["maximum_alternate_fraction"] for row in w1]); success=np.asarray([row["coordinated_capture"] for row in w1])
        ax.scatter(mags[~success],support[~success],s=10,alpha=.25,label="not captured",color="#9CA3AF"); ax.scatter(mags[success],support[success],s=18,alpha=.7,label="captured",color=COLORS[2])
        ax.set(xlabel="actual wrist motion magnitude (deg)",ylabel="maximum alternate-load fraction"); ax.legend()
        _style(ax,"Wrist pose vs support","Motion alone is not counted as useful; outcome is shown separately")

    def fig8(fig):
        ax=fig.add_subplot(111); counts=Counter()
        row,_=representative_release(); capture=next(item for item in data["load"] if item["state_id"]==row["state_id"]); arrays=_series(capture)
        labels=[]
        for flags in arrays["contact_flags"]:
            labels.append("{"+",".join(name for name,active in zip(SUPPORT_SURFACES,flags) if active)+"}")
        codes={name:i for i,name in enumerate(dict.fromkeys(labels))}; ax.step(arrays["step"],[codes[x] for x in labels],where="post",color=COLORS[3]); ax.set(xlabel="capture step",ylabel="topology code",yticks=list(codes.values()),yticklabels=list(codes))
        _style(ax,"Support topology timeline",f"Representative successful state {row['state_id']}")

    def fig9(fig):
        ax=fig.add_subplot(111); gates=["0.10/10","0.10/25","0.10/50","0.25/10","0.25/25","0.25/50","0.50/10","0.50/25","0.50/50"]
        for i,name in enumerate(("serial","simultaneous","wrist_load_transfer")):
            values=[summary["load_share_gate_results"][name][g]["successes"]/summary["load_share_gate_results"][name][g]["N"] for g in gates]
            ax.plot(gates,values,marker="o",label=name,color=COLORS[i])
        ax.set(xlabel="lambda_alt gate / persistence steps",ylabel="fraction reached",ylim=(-.01,.14)); ax.tick_params(axis="x",rotation=35); ax.legend()
        _style(ax,"Load-share gate analysis","Engineering diagnostics only; no publication gate was selected")

    def fig10(fig):
        ax=fig.add_subplot(111); labels=["25","50","100","200"]; vals=[summary["release"]["by_ramp"][x]["rate"] for x in labels]; ns=[summary["release"]["by_ramp"][x]["N"] for x in labels]
        bars=ax.bar(labels,vals,color=COLORS[1]); ax.bar_label(bars,labels=[f"{round(v*n)}/{n}" for v,n in zip(vals,ns)]); ax.set(xlabel="release ramp (steps)",ylabel="valid recovery fraction",ylim=(0,.8))
        _style(ax,"Release ramp analysis","Only 3 capture states reached the predefined 10%/25-step release gate")

    def fig11(fig):
        ax=fig.add_subplot(111); labels=["thumb first","index first"]; values=[summary["release"]["by_finger"][x]["rate"] for x in ("thumb","index")]; ns=[summary["release"]["by_finger"][x]["N"] for x in ("thumb","index")]
        bars=ax.bar(labels,values,color=[COLORS[0],COLORS[4]]); ax.bar_label(bars,labels=[f"{round(v*n)}/{n}" for v,n in zip(values,ns)]); ax.set(ylabel="valid recovery fraction",ylim=(0,1))
        _style(ax,"Thumb-first vs index-first","Fixture-free, floor-free retention plus contact-free usable motion through 1000 steps")

    def fig12(fig):
        ax=fig.add_subplot(111); checkpoints=list(summary["release"]["post_release_survival"]); vals=[summary["release"]["post_release_survival"][x]["successes"]/summary["release"]["post_release_survival"][x]["N"] for x in checkpoints]
        ax.step([int(x) for x in checkpoints],vals,where="post",color=COLORS[2],marker="o"); ax.set(xlabel="post-release steps",ylabel="retained and released finger contact-free",ylim=(0,1))
        _style(ax,"Post-release survival","All 24 executed release attempts; successful recoveries remain valid through step 1000")

    def fig13(fig):
        ax=fig.add_subplot(111); metrics=("A_retained","storage_finger_contact","palm_contact","coordinated_capture"); x=np.arange(len(metrics)); width=.36
        ax.bar(x-width/2,[sum(row[m] for row in serial)/len(serial) for m in metrics],width,label="serial",color=COLORS[0]); ax.bar(x+width/2,[sum(row[m] for row in simultaneous)/len(simultaneous) for m in metrics],width,label="simultaneous",color=COLORS[1]); ax.set(xticks=x,xticklabels=[m.replace("_","\n") for m in metrics],ylabel="fraction",ylim=(0,.15)); ax.legend()
        _style(ax,"Serial vs simultaneous capture","350 matched state-subset pairs per condition")

    def fig14(fig):
        ax=fig.add_subplot(111); labels=["serial","fixed sim","W1 -5,-5","W1 -5,+5","W1 +5,-5","W1 +5,+5"]; vals=[summary["serial"]["rate"],summary["simultaneous"]["rate"]]+[summary["wrist_by_command"][x]["coordinated_capture"]["rate"] for x in summary["wrist_by_command"]]
        ax.bar(np.arange(len(vals)),vals,color=[COLORS[0],COLORS[1],*([COLORS[2]]*4)]); ax.set(xticks=np.arange(len(vals)),xticklabels=labels,ylabel="capture fraction",ylim=(0,.2)); ax.tick_params(axis="x",rotation=22)
        _style(ax,"Serial vs wrist-coordinated capture","Direction-specific diagnostic effect; no wrist range is scientifically frozen")

    def fig15(fig):
        ax=fig.add_subplot(111); rows=data["corridor"]["trials"]; x=np.arange(len(rows)); width=.36
        ax.bar(x-width/2,[r["old_sphere_minimum_actual_path_clearance_m"]*1000 for r in rows],width,label="sphere metric on actual path",color=COLORS[0]); ax.bar(x+width/2,[r["exact_geom_minimum_actual_path_clearance_m"]*1000 for r in rows],width,label="exact geom distance",color=COLORS[2]); ax.axhline(0,color="black",lw=1); ax.set(xlabel="Phase 3C-0 trial",ylabel="minimum actual-path clearance (mm)"); ax.legend()
        _style(ax,"Corridor metric audit","No unused-finger contact was observed; prior negative candidate-path values were conservative")

    def fig16(fig):
        row, arrays=representative_release(); ax=fig.add_subplot(111); forces=arrays["normal_forces_n"]
        ax.stackplot(arrays["step"],*[forces[:,i] for i in range(6)],labels=SUPPORT_SURFACES,alpha=.8); ax.set(xlabel="thumb-release ramp step",ylabel="normal force (N)"); ax.legend(ncol=3,fontsize=8)
        _style(ax,"Representative CC-A success sequence",f"{row['state_id']}: thumb first, 100-step ramp, retained/contact-free through 1000 post steps")

    for name, draw in zip(FIGURE_NAMES,(fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13,fig14,fig15,fig16)):
        _save(figure_dir,name,draw)
    return [str(figure_dir/name) for name in FIGURE_NAMES]


def main() -> None:
    data=analyze(); figures=create_figures(data)
    print(json.dumps({"summary": str(ROOT/"outputs/phase3C05/analysis_summary.json"), "figures": figures},indent=2))


if __name__ == "__main__":
    main()
