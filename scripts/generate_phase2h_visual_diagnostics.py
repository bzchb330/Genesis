#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import shutil

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2_5_trajectory import _rotation_change
from seqgrasp.experiments.phase2h_render import (
    CameraSpec, DEFAULT_CAMERAS, FINGER_COLORS, MuJoCoDiagnosticRenderer,
    annotate_frame,
)
from seqgrasp.experiments.phase2h_visuals import (
    FINGER_NAMES, assert_replay_matches, diagnostic_series, replay_trial,
    scene_for_trial,
)
from seqgrasp.experiments.resource_components import (
    PALM_REFERENCE_TO_COMPILED, _finger_prefixes, _point_inside_geom,
    reconstruct_grasp,
)
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_config import load_phase2_config


FIGURE_DIR = ROOT / "docs" / "figures" / "phase2H"
OUTPUT_DIR = ROOT / "outputs" / "phase2H"
METRIC_METHOD_ID = "phase2h_existing_strict_gate_prefix_v3"
OBLIQUE = DEFAULT_CAMERAS[2]
CLOSE_UP = CameraSpec("close-up", 128.0, -24.0, 0.17)
FAILURE_MODES = (
    "NO_B_CONTACT_BEFORE_RELEASE",
    "B_ROTATED_OUT",
    "B_SLIPPED_TO_TABLE",
    "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE",
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _phase2w_source() -> tuple[Path, list[dict]]:
    evidence = json.loads((ROOT / "outputs" / "phase2W" / "analysis" / "evidence.json").read_text(encoding="utf-8"))
    expected = evidence["B_only_failure_mechanisms"]
    for summary_path in (ROOT / "outputs" / "phase2W" / "b_only_dynamic").rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        failures = Counter()
        for pose in summary.get("poses", []):
            failures.update(pose.get("failure_mechanisms", {}))
        if summary.get("total_B_only_candidates") == 8192 and dict(failures) == expected:
            source = summary_path.with_name("candidate_results.jsonl")
            return summary_path, _jsonl(source)
    raise FileNotFoundError("final Phase 2W B-only dataset not found")


def _metric_source() -> tuple[Path, list[dict]]:
    summaries = list((OUTPUT_DIR / "trial_metrics").rglob("summary.json"))
    complete = []
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "complete"
            and payload.get("trial_count") == 8192
            and payload.get("metric_method_id") == METRIC_METHOD_ID
        ):
            complete.append(path)
    if len(complete) != 1:
        raise RuntimeError(f"expected one complete Phase 2H metric set, found {len(complete)}")
    return complete[0], _jsonl(complete[0].with_name("metrics.jsonl"))


def _event_relative(metric: dict) -> int:
    mechanism = metric["failure_mechanism"]
    if mechanism == "B_ROTATED_OUT" and metric["first_rotation_violation_relative_step"] is not None:
        return int(metric["first_rotation_violation_relative_step"])
    if mechanism == "B_SLIPPED_TO_TABLE" and metric["first_table_contact_relative_step"] is not None:
        return int(metric["first_table_contact_relative_step"])
    if mechanism == "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE" and metric["first_dual_contact_loss_relative_step"] is not None:
        return int(metric["first_dual_contact_loss_relative_step"])
    if metric["first_strict_failure_relative_step"] is not None:
        return int(metric["first_strict_failure_relative_step"])
    return int(metric["unsupported_any_hand_contact_steps"])


def _ranking(metric: dict, primary: str) -> tuple:
    secondary = "dual_contact_survival_steps" if primary == "strict_survival_steps" else "strict_survival_steps"
    return (
        -int(metric[primary]),
        -int(metric[secondary]),
        -int(metric["unsupported_any_hand_contact_steps"]),
        metric["trial_id"],
    )


def _representatives(metrics: list[dict], failure: str) -> list[dict]:
    rows = sorted(
        (row for row in metrics if row["failure_mechanism"] == failure),
        key=lambda row: (_event_relative(row), row["trial_id"]),
    )
    if not rows:
        return []
    median = rows[(len(rows) - 1) // 2]
    late = sorted(rows, key=lambda row: (-_event_relative(row), row["trial_id"]))[0]
    return [median] if median["trial_id"] == late["trial_id"] else [median, late]


def _failure_step(metric: dict) -> int:
    return int(metric["fixture_release_timestep"] + _event_relative(metric))


def _safe_steps(values, total: int) -> list[int]:
    return sorted({int(np.clip(value, 0, total - 1)) for value in values})


@dataclass
class CaptureRequest:
    frame_steps: set[int] = field(default_factory=set)
    multiview_steps: set[int] = field(default_factory=set)
    contact_steps: set[int] = field(default_factory=set)
    thumbnail_step: int | None = None
    video_path: Path | None = None
    video_start: int | None = None
    video_end: int | None = None
    video_stride: int = 5
    capture_first_dual: bool = False


class TrialCapture:
    def __init__(self, trial: dict, cfg, request: CaptureRequest):
        self.trial = trial
        self.cfg = cfg
        self.request = request
        self.release = int(trial["fixture_release_timestep"])
        self.lookat = np.asarray(trial["placement"]["position_m"], dtype=float)
        self.renderer = MuJoCoDiagnosticRenderer(cfg)
        self.frames: dict[tuple[int, str], np.ndarray] = {}
        self.thumbnail = None
        self.video_writer = None
        self.reference_position = None
        self.reference_quaternion = None
        self.first_dual_step = None

    def _metrics(self, step: int, row: dict) -> list[str]:
        if self.reference_position is None:
            translation = rotation = 0.0
        else:
            translation = float(np.linalg.norm(np.asarray(row["B_position_m"]) - self.reference_position))
            rotation = _rotation_change(np.asarray(row["B_quaternion"]), self.reference_quaternion)
        flags = np.asarray(row["B_per_finger_contact_flag"], dtype=bool)
        return [
            f"trial {self.trial['trial_id']}",
            f"step {step} (release {step - self.release:+d})",
            f"index {'yes' if flags[0] else 'no'} | thumb {'yes' if flags[3] else 'no'} | table {'yes' if row['B_table_contact'] else 'no'}",
            f"B translation {translation:.4f} m | rotation {rotation:.3f} rad",
            f"index force {row['B_per_finger_normal_force_N'][0]:.3f} N | thumb force {row['B_per_finger_normal_force_N'][3]:.3f} N",
        ]

    def __call__(self, step, model, data, row):
        if step == self.release - 1:
            self.reference_position = np.asarray(row["B_position_m"], dtype=float).copy()
            self.reference_quaternion = np.asarray(row["B_quaternion"], dtype=float).copy()
        flags_now = np.asarray(row["B_per_finger_contact_flag"], dtype=bool)
        first_dual_now = bool(
            self.request.capture_first_dual
            and self.first_dual_step is None
            and flags_now[0] and flags_now[3]
            and not flags_now[1] and not flags_now[2]
        )
        if first_dual_now:
            self.first_dual_step = step
        if step in self.request.frame_steps or step == self.request.thumbnail_step or first_dual_now:
            cameras = DEFAULT_CAMERAS if step in self.request.multiview_steps else (OBLIQUE,)
            if step in self.request.contact_steps:
                cameras = (*cameras, CLOSE_UP)
            for camera in cameras:
                frame = self.renderer.render(
                    model, data, camera,
                    lookat=self.lookat,
                    candidate_region=self.trial["candidate_region"],
                    row=row,
                    contact_overlay=step in self.request.contact_steps,
                    show_vectors=step in self.request.multiview_steps,
                )
                self.frames[(step, camera.name)] = frame
                if step == self.request.thumbnail_step and self.thumbnail is None:
                    self.thumbnail = np.asarray(Image.fromarray(frame).resize((320, 240), Image.Resampling.LANCZOS))
        if (
            self.request.video_path is not None
            and self.request.video_start <= step <= self.request.video_end
            and (step - self.request.video_start) % self.request.video_stride == 0
        ):
            frame = self.renderer.render(
                model, data, OBLIQUE, lookat=self.lookat,
                candidate_region=self.trial["candidate_region"], row=row,
            )
            frame = annotate_frame(frame, self._metrics(step, row), released=step >= self.release)
            if self.video_writer is None:
                self.request.video_path.parent.mkdir(parents=True, exist_ok=True)
                self.video_writer = imageio.get_writer(self.request.video_path, fps=25, macro_block_size=8)
            self.video_writer.append_data(frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
        self.renderer.close()


def _panel_text(arrays, series, step: int, release: int) -> str:
    flags = series["contact_flags"][step]
    return (
        f"release {step - release:+d} | index {'Y' if flags[0] else 'N'} | thumb {'Y' if flags[3] else 'N'} | "
        f"table {'Y' if arrays['B_table_contact'][step] else 'N'}\n"
        f"translation {series['translation_m'][step]:.4f} m | rotation {series['rotation_rad'][step]:.3f} rad"
    )


def _image_grid(path: Path, title: str, rows: list[tuple[str, list[tuple[np.ndarray, str]]]], columns: int) -> None:
    maximum = max(len(panels) for _, panels in rows)
    columns = min(columns, maximum)
    row_blocks = sum(math.ceil(len(panels) / columns) for _, panels in rows)
    fig, axes = plt.subplots(row_blocks, columns, figsize=(4.5 * columns, 3.35 * row_blocks), squeeze=False)
    cursor = 0
    for label, panels in rows:
        for block in range(math.ceil(len(panels) / columns)):
            for column in range(columns):
                axis = axes[cursor, column]
                index = block * columns + column
                if index >= len(panels):
                    axis.axis("off")
                    continue
                image, caption = panels[index]
                axis.imshow(image)
                axis.axis("off")
                is_release = "FIXTURE RELEASE" in caption
                axis.set_title(
                    caption,
                    fontsize=8,
                    color="crimson" if is_release else "black",
                    weight="bold" if is_release else "normal",
                )
                if column == 0:
                    axis.text(-0.04, 0.5, label, rotation=90, va="center", ha="right", transform=axis.transAxes, weight="bold")
            cursor += 1
    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_three_views(path: Path, frames: dict, step: int, title: str, subtitle: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for axis, camera in zip(axes, DEFAULT_CAMERAS):
        axis.imshow(frames[(step, camera.name)])
        axis.axis("off")
        axis.set_title(camera.name)
    handles = [Patch(color=FINGER_COLORS[name], label=name) for name in FINGER_NAMES]
    handles += [
        Patch(color=(0.2, 0.3, 0.8, 1.0), label="B"),
        Line2D([0], [0], color="crimson", lw=3, label="palm normal"),
        Line2D([0], [0], color="black", lw=3, label="gravity"),
        Patch(color=(0.1, 0.7, 0.7, 0.25), label="candidate B region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False)
    fig.suptitle(title, fontsize=17, weight="bold")
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _force_traces(path: Path, selected: list[dict], traces: dict[str, dict], cfg25) -> None:
    fig, axes = plt.subplots(len(selected), 2, figsize=(13.5, 2.25 * len(selected)), sharex=False)
    for row_index, metric in enumerate(selected):
        arrays = traces[metric["trial_id"]]["arrays"]
        release = int(metric["fixture_release_timestep"])
        time = np.arange(len(arrays["timestep"])) - release
        forces = arrays["B_per_finger_normal_force_N"]
        ratios = arrays["B_per_finger_tangential_normal_ratio"]
        failure = metric["first_strict_failure_relative_step"]
        left, right = axes[row_index]
        left.plot(time, forces[:, 0], label="index", color=FINGER_COLORS["index"][:3])
        left.plot(time, forces[:, 3], label="thumb", color=FINGER_COLORS["thumb"][:3])
        right.plot(time, ratios[:, 0], label="index", color=FINGER_COLORS["index"][:3])
        right.plot(time, ratios[:, 3], label="thumb", color=FINGER_COLORS["thumb"][:3])
        for axis in (left, right):
            axis.axvline(0, color="black", linestyle="--", linewidth=1, label="fixture release")
            if failure is not None:
                axis.axvline(failure, color="crimson", linestyle=":", linewidth=1.4, label="first strict failure")
            axis.grid(alpha=0.25)
            axis.set_xlim(-150, cfg25.timing.unsupported_hold_steps)
        left.set_ylabel(f"trial {row_index + 1}\nnormal force [N]")
        right.set_ylabel("tangential / normal")
        left.text(0.01, 0.90, metric["trial_id"].split(":", 1)[1][:12], transform=left.transAxes, fontsize=8)
    axes[0, 0].set_title("Index/thumb normal force")
    axes[0, 1].set_title("Tangential-to-normal ratio")
    axes[-1, 0].set_xlabel("steps relative to fixture release")
    axes[-1, 1].set_xlabel("steps relative to fixture release")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Representative force traces - five longest strict-survival trials", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _motion_traces(path: Path, selected: list[dict], traces: dict[str, dict], cfg25) -> None:
    fig, axes = plt.subplots(len(selected), 3, figsize=(15.5, 2.2 * len(selected)), sharex=False)
    for row_index, metric in enumerate(selected):
        arrays = traces[metric["trial_id"]]["arrays"]
        series = traces[metric["trial_id"]]["series"]
        release = int(metric["fixture_release_timestep"])
        time = np.arange(len(arrays["timestep"])) - release
        values = (
            (series["translation_m"], "translation [m]"),
            (series["rotation_rad"], "rotation [rad]"),
            (series["vertical_displacement_m"], "vertical displacement [m]"),
        )
        event_lines = (
            (metric["first_dual_contact_loss_relative_step"], "contact loss", "tab:orange"),
            (metric["first_table_contact_relative_step"], "table contact", "tab:red"),
            (metric["first_rotation_violation_relative_step"], "rotation threshold", "tab:purple"),
        )
        for column, ((value, ylabel), axis) in enumerate(zip(values, axes[row_index])):
            axis.plot(time, value, color="tab:blue", linewidth=1.2)
            axis.axvline(0, color="black", linestyle="--", linewidth=1)
            for event, _, color in event_lines:
                if event is not None:
                    axis.axvline(event, color=color, linestyle=":", linewidth=1)
            axis.grid(alpha=0.25)
            axis.set_xlim(-150, cfg25.timing.unsupported_hold_steps)
            axis.set_ylabel(ylabel if column else f"trial {row_index + 1}\n{ylabel}")
        axes[row_index, 0].text(0.01, 0.90, metric["trial_id"].split(":", 1)[1][:12], transform=axes[row_index, 0].transAxes, fontsize=8)
    for axis, title in zip(axes[0], ("B translation from release", "B rotation from release", "B vertical motion")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("steps relative to fixture release")
    legend = [
        Line2D([0], [0], color="black", linestyle="--", label="fixture release"),
        Line2D([0], [0], color="tab:orange", linestyle=":", label="contact loss"),
        Line2D([0], [0], color="tab:red", linestyle=":", label="table contact"),
        Line2D([0], [0], color="tab:purple", linestyle=":", label="rotation threshold"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Representative object-motion traces", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _contact_timeline(path: Path, selected: list[dict], traces: dict[str, dict]) -> None:
    labels = ("index-B", "thumb-B", "middle-B", "ring-B", "B-table", "fixture")
    colors = (
        FINGER_COLORS["index"], FINGER_COLORS["thumb"], FINGER_COLORS["middle"],
        FINGER_COLORS["ring"], (0.75, 0.15, 0.15, 1.0), (0.35, 0.35, 0.35, 1.0),
    )
    fig, axes = plt.subplots(len(selected), 1, figsize=(15.5, 1.55 * len(selected)), sharex=True)
    for trial_index, (axis, metric) in enumerate(zip(np.atleast_1d(axes), selected)):
        arrays = traces[metric["trial_id"]]["arrays"]
        release = int(metric["fixture_release_timestep"])
        time = np.arange(len(arrays["timestep"])) - release
        flags = arrays["B_per_finger_contact_flag"].astype(bool)
        states = (
            flags[:, 0], flags[:, 3], flags[:, 1], flags[:, 2],
            arrays["B_table_contact"].astype(bool), arrays["fixture_active"].astype(bool),
        )
        for row_index, (state, color) in enumerate(zip(states, colors)):
            active = np.flatnonzero(state)
            axis.scatter(time[active], np.full(len(active), row_index), marker="s", s=7, color=color)
        axis.axvline(0, color="black", linewidth=1.2)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set_ylim(len(labels) - 0.5, -0.5)
        axis.set_ylabel(f"{trial_index + 1}\n{metric['trial_id'].split(':', 1)[1][:8]}", rotation=0, labelpad=22, fontsize=8)
        axis.grid(axis="x", alpha=0.18)
    axes[-1].set_xlabel("steps relative to fixture release")
    fig.suptitle("Contact-state timeline - ten longest dual-contact-survival trials", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _survival_distribution(path: Path, metrics: list[dict]) -> None:
    references = (25, 50, 100, 200, 300, 400, 500)
    strict = np.asarray([row["strict_survival_steps"] for row in metrics])
    dual = np.asarray([row["dual_contact_survival_steps"] for row in metrics])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for column, (values, name, color) in enumerate(((dual, "dual-contact", "tab:blue"), (strict, "strict", "tab:orange"))):
        axes[0, column].hist(values, bins=np.arange(-0.5, 501.5, 10), color=color, alpha=0.8)
        sorted_values = np.sort(values)
        axes[1, column].step(sorted_values, np.arange(1, len(values) + 1) / len(values), where="post", color=color)
        for axis in (axes[0, column], axes[1, column]):
            for reference in references:
                axis.axvline(reference, color="0.45", linestyle=":", linewidth=0.8)
            axis.set_xlim(-2, 505)
            axis.grid(alpha=0.2)
            axis.set_xlabel("survival duration [steps]")
        axes[0, column].set_title(f"{name} survival histogram")
        axes[1, column].set_title(f"{name} survival empirical CDF")
    axes[0, 0].set_ylabel("trial count")
    axes[1, 0].set_ylabel("empirical cumulative fraction")
    fig.suptitle("Phase 2W contact-survival duration distributions (all 8,192 trials)", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _funnel(path: Path, metrics: list[dict]) -> dict[str, int]:
    index_established = [row for row in metrics if row["index_contact_established_pre_release"]]
    both_established = [row for row in index_established if row["thumb_contact_established_pre_release"]]
    dual_pre = [row for row in both_established if row["dual_contact_pre_release"]]
    dual_release = [row for row in dual_pre if row["dual_contact_at_release"]]
    counts = {
        "GEOMETRICALLY\nACCESSIBLE": len(metrics),
        "INDEX CONTACT\nESTABLISHED": len(index_established),
        "THUMB CONTACT\nESTABLISHED": len(both_established),
        "DUAL CONTACT\nPRE-RELEASE": len(dual_pre),
        "DUAL CONTACT\nAT RELEASE": len(dual_release),
    }
    for threshold in (10, 25, 50, 100, 200, 500):
        counts[f"survives\n{threshold} steps"] = sum(row["strict_survival_steps"] >= threshold for row in metrics)
    labels, values = list(counts), list(counts.values())
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(values)))
    fig, axis = plt.subplots(figsize=(16, 7.5))
    bars = axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(values)), labels, fontsize=8)
    axis.set_ylabel("measured trial count")
    axis.set_title("From geometric accessibility to unsupported retention", weight="bold", fontsize=17)
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012, f"{value:,}", ha="center", fontsize=9)
    axis.text(
        0.99, 0.98,
        "Intermediate stages are diagnostic counts, not success criteria.",
        transform=axis.transAxes, va="top", ha="right",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return counts


def _wrist_diagnostics(path: Path, metrics: list[dict], trials: dict[str, dict]) -> list[dict]:
    grouped = defaultdict(list)
    for metric in metrics:
        grouped[metric["wrist_pose_id"]].append(metric)
    rows = []
    for pose_id, subset in grouped.items():
        trial = trials[subset[0]["trial_id"]]
        failures = Counter(row["failure_mechanism"] for row in subset)
        dominant = sorted(failures.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append({
            "pose_id": pose_id,
            "rpy": subset[0]["wrist_rpy_deg"],
            "geometry_access": float(trial["geometry_rank_evidence"]["minimum_group_access"]),
            "dual_pre_rate": float(np.mean([row["dual_contact_pre_release"] for row in subset])),
            "median_survival": float(np.median([row["dual_contact_survival_steps"] for row in subset])),
            "maximum_strict": int(max(row["strict_survival_steps"] for row in subset)),
            "dominant_failure": dominant,
            "trial_count": len(subset),
        })
    rows.sort(key=lambda row: row["pose_id"])
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(18, 7.2), gridspec_kw={"width_ratios": [2.6, 1.1, 1.1, 2.2]})
    labels = [f"{row['pose_id'][:22]}...\nRPY {row['rpy']}" for row in rows]
    axes[0].barh(y, [row["geometry_access"] for row in rows], color="tab:cyan")
    axes[0].set_yticks(y, labels, fontsize=7)
    axes[0].set_xlabel("minimum group geometry access")
    axes[1].barh(y, [row["dual_pre_rate"] for row in rows], color="tab:blue")
    axes[1].set_xlabel("dual-contact pre-release rate")
    axes[1].set_yticks([])
    axes[2].barh(y - 0.18, [row["median_survival"] for row in rows], height=0.35, label="median dual", color="tab:green")
    axes[2].barh(y + 0.18, [row["maximum_strict"] for row in rows], height=0.35, label="max strict", color="tab:orange")
    axes[2].set_xlabel("post-release steps")
    axes[2].set_yticks([])
    axes[2].legend(fontsize=8)
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(len(rows) - 0.5, -0.5)
    axes[3].set_xticks([])
    axes[3].set_yticks(y, [row["dominant_failure"] for row in rows], fontsize=7)
    axes[3].set_title("dominant failure mode", fontsize=10)
    for axis in axes[:3]:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
    fig.suptitle("Retrospective static-wrist dynamic diagnostics - no re-selection", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return rows


def _endpoint_records(pose_id: str) -> dict[str, dict]:
    candidates = []
    for summary_path in (ROOT / "outputs" / "phase2W" / "endpoint_screen" / "refined").rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("orientation_count") == 114:
            candidates.append(summary_path.with_name("endpoint_trials.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one final refined endpoint screen, found {len(candidates)}")
    rows = [
        row for row in _jsonl(candidates[0])
        if row.get("accepted") and row.get("pose_id") == pose_id
    ]
    result = {}
    for group in ("FINGERTIP", "PALMAR_SECURED"):
        subset = sorted((row for row in rows if row["group"] == group), key=lambda row: row["source_state_id"])
        if not subset:
            raise RuntimeError(f"no accepted {group} endpoint at diagnostic wrist")
        medians = {
            key: np.median([float(row[key]) for row in subset])
            for key in ("ferrari_canny_epsilon", "total_A_normal_force_N", "A_translation_drift_m", "A_rotation_drift_rad", "minimum_joint_margin_rad")
        }
        scales = {
            key: max(np.std([float(row[key]) for row in subset]), 1e-12)
            for key in medians
        }
        result[group] = min(
            subset,
            key=lambda row: (
                sum(abs(float(row[key]) - medians[key]) / scales[key] for key in medians),
                row["source_state_id"],
            ),
        )
    return result


def _park_B(model, data) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qadr:qadr + 3] = [-0.35, 0.35, 0.10]
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def _endpoint_comparison(path: Path, endpoints: dict[str, dict], candidate_region: dict, base_cfg) -> None:
    frames = {}
    metadata = {}
    bounds = candidate_region["center_bounds_m"]
    region_center = np.asarray([(bounds[axis][0] + bounds[axis][1]) / 2 for axis in "xyz"])
    for group, record in endpoints.items():
        cfg, model, data, _ = reconstruct_grasp(record, base_cfg)
        _park_B(model, data)
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
        com = np.asarray(data.xpos[object_id], dtype=float)
        lookat = (com + region_center) / 2.0
        renderer = MuJoCoDiagnosticRenderer(cfg)
        for camera in DEFAULT_CAMERAS:
            frames[(group, camera.name)] = renderer.render(
                model, data, camera, lookat=lookat, candidate_region=candidate_region,
                show_vectors=True, markers=[(com, (0.85, 0.05, 0.05, 1.0))],
            )
        renderer.close()
        metadata[group] = record
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    for row_index, group in enumerate(("FINGERTIP", "PALMAR_SECURED")):
        for column, camera in enumerate(DEFAULT_CAMERAS):
            axis = axes[row_index, column]
            axis.imshow(frames[(group, camera.name)])
            axis.axis("off")
            axis.set_title(f"{group} - {camera.name}")
        record = metadata[group]
        axes[row_index, 0].text(
            0.01, 0.98,
            f"A COM marked red | middle+ring occupied | index+thumb free\nsource {record['source_state_id']}",
            transform=axes[row_index, 0].transAxes, fontsize=8, va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    fig.suptitle("Validated endpoints at the same diagnostic wrist pose - visualization only", weight="bold", fontsize=16)
    fig.text(0.5, 0.945, "Same camera, world frame, scale, wrist orientation, and B candidate-region overlay; no A+B trial was run.", ha="center", fontsize=10)
    handles = [Patch(color=FINGER_COLORS[name], label=name) for name in FINGER_NAMES]
    handles += [
        Line2D([0], [0], marker="o", color="darkred", linestyle="none", label="A COM"),
        Patch(color="0.55", label="actual palm surface"),
        Patch(color=(0.1, 0.7, 0.7, 0.25), label="B candidate region"),
        Line2D([0], [0], color="black", label="gravity"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False)
    fig.subplots_adjust(
        left=0.015, right=0.985, top=0.88, bottom=0.13,
        wspace=0.035, hspace=0.18,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _palm_space_payload(record: dict, resources, base_cfg) -> dict:
    cfg, model, data, _ = reconstruct_grasp(record, base_cfg)
    low = np.asarray(resources.free_palm_box_lower_m, dtype=float)
    high = np.asarray(resources.free_palm_box_upper_m, dtype=float)
    step = float(resources.free_palm_voxel_size_m)
    axes = [np.arange(low[index] + step / 2.0, high[index], step) for index in range(3)]
    shape = tuple(len(axis) for axis in axes)
    points_ref = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points_compiled = points_ref @ PALM_REFERENCE_TO_COMPILED.T
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    points_world = data.xpos[palm_id] + points_compiled @ palm_rotation.T
    prefixes = set(_finger_prefixes(cfg).values())
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    relevant = [object_geom]
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if any(body_name.startswith(prefix + "_") for prefix in prefixes):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                relevant.append(geom_id)
    occupied = np.zeros(len(points_ref), dtype=bool)
    for geom_id in relevant:
        occupied |= _point_inside_geom(model, data, geom_id, points_world)
    free_grid = (~occupied).reshape(shape)
    labels, count = ndimage.label(free_grid, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1) if count else 0
    largest = labels.ravel() == largest_label
    padded = np.pad(free_grid, 1, constant_values=False)
    distance = ndimage.distance_transform_edt(padded, sampling=step)[1:-1, 1:-1, 1:-1]
    sphere_index = np.unravel_index(np.argmax(distance), distance.shape)
    sphere_center = np.asarray([axes[index][sphere_index[index]] for index in range(3)])
    object_rotation = data.geom_xmat[object_geom].reshape(3, 3)
    object_size = model.geom_size[object_geom, :3]
    corners_local = np.asarray([
        [x, y, z] for x in (-object_size[0], object_size[0])
        for y in (-object_size[1], object_size[1]) for z in (-object_size[2], object_size[2])
    ])
    corners_world = data.geom_xpos[object_geom] + corners_local @ object_rotation.T
    corners_compiled = (corners_world - data.xpos[palm_id]) @ palm_rotation
    corners_ref = corners_compiled @ PALM_REFERENCE_TO_COMPILED
    palm_surfaces = []
    for palm_geom in (index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == palm_id):
        if int(model.geom_type[palm_geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        size = model.geom_size[palm_geom, :3]
        local = np.asarray([
            [x, y, z] for x in (-size[0], size[0])
            for y in (-size[1], size[1]) for z in (-size[2], size[2])
        ])
        world = data.geom_xpos[palm_geom] + local @ data.geom_xmat[palm_geom].reshape(3, 3).T
        compiled = (world - data.xpos[palm_id]) @ palm_rotation
        palm_surfaces.append(compiled @ PALM_REFERENCE_TO_COMPILED)
    com_compiled = (data.geom_xpos[object_geom] - data.xpos[palm_id]) @ palm_rotation
    return {
        "points": points_ref, "occupied": occupied, "largest": largest,
        "low": low, "high": high, "corners": corners_ref,
        "com": com_compiled @ PALM_REFERENCE_TO_COMPILED,
        "palm_surfaces": palm_surfaces,
        "sphere_center": sphere_center, "sphere_radius": float(np.max(distance)),
        "free_volume": float(np.sum(~occupied) * step ** 3),
        "occupied_fraction": float(np.mean(occupied)),
    }


def _draw_box_edges(axis, corners: np.ndarray, color, linewidth=1.4):
    for left in range(8):
        for right in range(left + 1, 8):
            if np.sum(~np.isclose(corners[left], corners[right])) == 1:
                axis.plot(*zip(corners[left], corners[right]), color=color, linewidth=linewidth)


def _palm_space_plot(path: Path, endpoints: dict[str, dict], resources, base_cfg) -> None:
    payloads = {group: _palm_space_payload(record, resources, base_cfg) for group, record in endpoints.items()}
    fig = plt.figure(figsize=(15.5, 7.5))
    for index, group in enumerate(("FINGERTIP", "PALMAR_SECURED"), start=1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        payload = payloads[group]
        points = payload["points"]
        free_index = np.flatnonzero(payload["largest"])[::24]
        occupied_index = np.flatnonzero(payload["occupied"])[::2]
        axis.scatter(*points[free_index].T, s=1.2, alpha=0.08, color="tab:cyan", label="largest connected free region")
        axis.scatter(*points[occupied_index].T, s=9, alpha=0.65, color="tab:red", label="occupied voxels")
        low, high = payload["low"], payload["high"]
        box = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        _draw_box_edges(axis, box, "0.25", 1.0)
        _draw_box_edges(axis, payload["corners"], "darkred", 2.0)
        for surface in payload["palm_surfaces"]:
            _draw_box_edges(axis, surface, "black", 2.2)
        axis.scatter(*payload["com"], marker="x", s=90, color="black", label="A COM")
        sphere = payload["sphere_center"]
        axis.scatter(*sphere, marker="o", s=140, facecolors="none", edgecolors="goldenrod", linewidths=2, label="inscribed-sphere center")
        axis.set_title(
            f"{group}\nfree {payload['free_volume']:.7f} m³ | occupied {payload['occupied_fraction']:.5f}\n"
            f"largest sphere radius {payload['sphere_radius']:.4f} m"
        )
        axis.set_xlabel("palm reference x [m]")
        axis.set_ylabel("palm reference y [m]")
        axis.set_zlabel("palm reference z [m]")
        axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_zlim(low[2], high[2])
        axis.view_init(elev=24, azim=-56)
    handles, labels = axis.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="black", lw=2.2, label="actual palm collision surface")]
    labels += ["actual palm collision surface"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Why scalar free-palm volume is insensitive to spatial arrangement", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _contact_sheet(entries: list[tuple[str, dict, np.ndarray]]) -> tuple[Path, Path]:
    review = OUTPUT_DIR / "visual_review"
    thumbnails = review / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    cards = []
    html_cards = []
    font = ImageFont.load_default(size=15)
    for index, (category, metric, thumbnail) in enumerate(entries):
        filename = f"{index:03d}_{metric['trial_id'].split(':', 1)[1][:12]}.jpg"
        Image.fromarray(thumbnail).save(thumbnails / filename, quality=88)
        text = [
            category,
            f"trial {metric['trial_id'].split(':', 1)[1][:16]}",
            f"wrist {metric['wrist_pose_id'][:28]}",
            f"proposal {metric['trajectory_proposal_center']}",
            f"strict {metric['strict_survival_steps']} | dual {metric['dual_contact_survival_steps']}",
            f"failure {metric['failure_mechanism']}",
        ]
        card = Image.new("RGB", (340, 365), "white")
        card.paste(Image.fromarray(thumbnail).resize((320, 240)), (10, 10))
        draw = ImageDraw.Draw(card)
        for line_index, line in enumerate(text):
            draw.text((12, 255 + 17 * line_index), line, fill=(15, 30, 50), font=font)
        cards.append(card)
        html_cards.append(
            f"<article><img src='thumbnails/{filename}'><b>{category}</b>"
            f"<code>{metric['trial_id']}</code><span>wrist: {metric['wrist_pose_id']}</span>"
            f"<span>proposal: {metric['trajectory_proposal_center']}</span>"
            f"<span>strict: {metric['strict_survival_steps']} | dual: {metric['dual_contact_survival_steps']}</span>"
            f"<span>failure: {metric['failure_mechanism']}</span></article>"
        )
    columns = 5
    rows = math.ceil(len(cards) / columns)
    sheet = Image.new("RGB", (columns * 340, rows * 365 + 70), (235, 240, 246))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), "Phase 2H deterministic visual review contact sheet", fill=(10, 25, 45), font=ImageFont.load_default(size=24))
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % columns) * 340, 70 + (index // columns) * 365))
    png_path = review / "contact_sheet.png"
    sheet.save(png_path)
    html_path = review / "contact_sheet.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Phase 2H visual review</title>"
        "<style>body{font:14px system-ui;background:#eef2f6;color:#102030;margin:20px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}"
        "article{background:white;padding:10px;border-radius:8px;box-shadow:0 1px 4px #abc}"
        "img{width:100%;display:block;margin-bottom:8px}code,span{display:block;overflow-wrap:anywhere;margin-top:3px}</style>"
        "<h1>Phase 2H deterministic visual review</h1><main>" + "".join(html_cards) + "</main>",
        encoding="utf-8",
    )
    return html_path, png_path


def _visual_index(path: Path, records: list[dict], special: dict, videos: dict, contact_paths: tuple[Path, Path]) -> None:
    lines = [
        "# Phase 2H visual index",
        "",
        "All simulation images are deterministic replays of existing Phase 2W trial IDs. No search candidate, controller, release time, wrist pose, B pose, physics parameter, or outcome was changed.",
        "",
        "For visualization ordering only, `dual_contact_survival_steps` is the consecutive post-release prefix with index+thumb contact and no middle/ring assist. `strict_survival_steps` is the consecutive prefix satisfying the existing Phase 2W pre-release dual-contact, no-assist, minimum-contact, normal-force, cumulative penetration, table, translation, rotation, and numerical gates. These are diagnostic durations, not new success criteria.",
        "",
        "| File | What it shows | Underlying trial IDs | Wrist pose IDs | Type | Factual observation |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        trial_ids = "<br>".join(f"`{value}`" for value in record.get("trial_ids", [])) or "N/A"
        wrist_ids = "<br>".join(f"`{value}`" for value in record.get("wrist_ids", [])) or "N/A"
        lines.append(
            f"| `{record['path']}` | {record['shows']} | {trial_ids} | {wrist_ids} | {record['type']} | {record['observation']} |"
        )
    lines += ["", "## Selected trial IDs", ""]
    for label, metric in special.items():
        lines.append(f"- {label}: `{metric['trial_id']}` ({metric['wrist_pose_id']})")
    lines += ["", "## Ignored diagnostic artifacts", ""]
    for label, video in videos.items():
        lines.append(f"- {label}: `{video.relative_to(ROOT)}`")
    lines.append(f"- Contact sheet HTML: `{contact_paths[0].relative_to(ROOT)}`")
    lines.append(f"- Contact sheet PNG: `{contact_paths[1].relative_to(ROOT)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate required Phase 2H visual diagnostics from exact Phase 2W replays")
    parser.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path, trial_rows = _phase2w_source()
    metric_path, metrics = _metric_source()
    if len(trial_rows) != 8192 or len(metrics) != 8192:
        raise RuntimeError("Phase 2H visualization requires the complete 8192-trial dataset")
    trials = {row["trial_id"]: row for row in trial_rows}
    if set(trials) != {row["trial_id"] for row in metrics}:
        raise RuntimeError("Phase 2H metric/trial ID sets differ")
    metric_by_id = {row["trial_id"]: row for row in metrics}
    phase2w_evidence = json.loads((ROOT / "outputs" / "phase2W" / "analysis" / "evidence.json").read_text(encoding="utf-8"))
    diagnostic_pose_id = phase2w_evidence["highest_ranked_failed_candidate"]["pose"]["pose_id"]
    top_strict = sorted(metrics, key=lambda row: _ranking(row, "strict_survival_steps"))[:20]
    top_dual = sorted(metrics, key=lambda row: _ranking(row, "dual_contact_survival_steps"))[:20]
    best_strict, best_dual = top_strict[0], top_dual[0]
    diagnostic_pose_metric = sorted(
        (row for row in metrics if row["wrist_pose_id"] == diagnostic_pose_id),
        key=lambda row: _ranking(row, "dual_contact_survival_steps"),
    )[0]
    failure_examples = {mode: _representatives(metrics, mode) for mode in FAILURE_MODES}
    longest_rotation = sorted(
        (row for row in metrics if row["failure_mechanism"] == "B_ROTATED_OUT"),
        key=lambda row: (-_event_relative(row), row["trial_id"]),
    )[0]
    longest_slip = sorted(
        (row for row in metrics if row["failure_mechanism"] == "B_SLIPPED_TO_TABLE"),
        key=lambda row: (-_event_relative(row), row["trial_id"]),
    )[0]
    special = {
        "longest strict survival": best_strict,
        "longest dual-contact survival": best_dual,
        "longest rotation failure": longest_rotation,
        "longest slip/table failure": longest_slip,
    }
    cfg25, _ = load_phase2_5_config()
    phase2, _ = load_phase2_config()
    base_cfg = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    total_steps = lambda metric: int(metric["fixture_release_timestep"] + cfg25.timing.unsupported_hold_steps)
    requests: dict[str, CaptureRequest] = defaultdict(CaptureRequest)

    # V1 uses the highest-ranked Phase 2W diagnostic wrist; V2 uses the
    # globally longest measured dual-contact trial.
    diagnostic_release = int(diagnostic_pose_metric["fixture_release_timestep"])
    requests[diagnostic_pose_metric["trial_id"]].frame_steps.add(diagnostic_release - 50)
    requests[diagnostic_pose_metric["trial_id"]].multiview_steps.add(diagnostic_release - 50)
    release = int(best_dual["fixture_release_timestep"])
    dual_steps = [release + offset for offset in (-100, -50, -10, 0, 10, 25, 50, 100)]
    requests[best_dual["trial_id"]].frame_steps.update(_safe_steps(dual_steps, total_steps(best_dual)))
    requests[best_dual["trial_id"]].multiview_steps.add(release - 50)
    contact_steps = [release + offset for offset in (-10, 0, 10, 25, 50)]
    requests[best_dual["trial_id"]].frame_steps.update(_safe_steps(contact_steps, total_steps(best_dual)))
    requests[best_dual["trial_id"]].contact_steps.update(_safe_steps(contact_steps, total_steps(best_dual)))

    # V3/V4 detailed sequences include approach, release, first failure and after-failure state.
    sequence_steps = {}
    for label, metric in (("strict", best_strict), ("dual", best_dual)):
        release = int(metric["fixture_release_timestep"])
        strict_relative = metric["first_strict_failure_relative_step"]
        failure = release + (int(strict_relative) if strict_relative is not None else cfg25.timing.unsupported_hold_steps)
        end = min(total_steps(metric) - 1, max(failure + 25, release + 100))
        steps = _safe_steps(
            [*np.linspace(0, end, 10), release - 1, release, failure - 1, failure, failure + 25],
            total_steps(metric),
        )
        sequence_steps[label] = steps
        requests[metric["trial_id"]].frame_steps.update(steps)
        requests[metric["trial_id"]].capture_first_dual = True

    # V5 deterministic median and late representatives.
    failure_steps = {}
    for mode, examples in failure_examples.items():
        failure_steps[mode] = {}
        for metric in examples:
            release = int(metric["fixture_release_timestep"])
            event = max(_event_relative(metric), 1)
            if mode == "NO_B_CONTACT_BEFORE_RELEASE":
                values = [release - 100, release, release + 25, release + 100]
            else:
                values = [release - 50, release, release + max(1, event // 2), release + event]
            steps = _safe_steps(values, total_steps(metric))
            failure_steps[mode][metric["trial_id"]] = steps
            requests[metric["trial_id"]].frame_steps.update(steps)

    # V15 videos: one exact existing trial per requested category.
    video_dir = OUTPUT_DIR / "videos"
    video_metrics = {
        "best_strict_survival": best_strict,
        "best_dual_contact_survival": best_dual,
        "representative_rotation_failure": failure_examples["B_ROTATED_OUT"][0],
        "representative_slip_failure": failure_examples["B_SLIPPED_TO_TABLE"][0],
        "representative_immediate_contact_loss": failure_examples["CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE"][0],
        "representative_no_contact": failure_examples["NO_B_CONTACT_BEFORE_RELEASE"][0],
    }
    videos = {}
    video_aliases = []
    for label, metric in video_metrics.items():
        request = requests[metric["trial_id"]]
        release = int(metric["fixture_release_timestep"])
        event = _failure_step(metric)
        video_path = video_dir / f"{label}.mp4"
        videos[label] = video_path
        if request.video_path is None:
            request.video_path = video_path
            request.video_start = max(0, release - 150)
            request.video_end = min(total_steps(metric) - 1, max(release + 150, event + 75))
        else:
            video_aliases.append((request.video_path, video_path))

    # V16 top lists and ten deterministic failure representatives.
    failure_sheet = []
    for mode in FAILURE_MODES:
        candidates = sorted((row for row in metrics if row["failure_mechanism"] == mode), key=lambda row: (_event_relative(row), row["trial_id"]))
        for quantile in (0.5, 0.85, 1.0):
            index = min(len(candidates) - 1, int(round((len(candidates) - 1) * quantile)))
            if candidates[index]["trial_id"] not in {row["trial_id"] for row in failure_sheet}:
                failure_sheet.append(candidates[index])
            if len(failure_sheet) >= 10:
                break
        if len(failure_sheet) >= 10:
            break
    contact_entries = [*(('top strict', row) for row in top_strict), *(('top dual', row) for row in top_dual), *(('failure representative', row) for row in failure_sheet[:10])]
    for _, metric in contact_entries:
        request = requests[metric["trial_id"]]
        if request.thumbnail_step is None:
            request.thumbnail_step = min(total_steps(metric) - 1, max(int(metric["fixture_release_timestep"]), _failure_step(metric)))

    trace_ids = {
        *(row["trial_id"] for row in top_strict[:5]),
        *(row["trial_id"] for row in top_dual[:10]),
        *(row["trial_id"] for rows in failure_examples.values() for row in rows),
        best_strict["trial_id"], best_dual["trial_id"],
    }
    captures, traces = {}, {}
    needed = sorted(set(requests) | trace_ids)
    for index, trial_id in enumerate(needed, start=1):
        trial = trials[trial_id]
        request = requests[trial_id]
        capture = TrialCapture(trial, scene_for_trial(base_cfg, trial), request)
        try:
            replay_summary, arrays = replay_trial(trial, cfg25, base_cfg, diagnostic_callback=capture)
            assert_replay_matches(trial, replay_summary)
        finally:
            capture.close()
        captures[trial_id] = {
            "frames": capture.frames,
            "thumbnail": capture.thumbnail,
            "first_dual_step": capture.first_dual_step,
        }
        if trial_id in trace_ids:
            traces[trial_id] = {"arrays": arrays, "series": diagnostic_series(replay_summary, arrays, cfg25)}
        print(f"Phase 2H selected replay/render: {index}/{len(needed)}")
    for source, target in video_aliases:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    records = []
    def record(filename, shows, trial_metrics, kind, observation):
        aggregate = not trial_metrics
        records.append({
            "path": str((FIGURE_DIR / filename).relative_to(ROOT)), "shows": shows,
            "trial_ids": (
                [row["trial_id"] for row in trial_metrics]
                if not aggregate else [f"all 8192 IDs in {metric_path.with_name('metrics.jsonl').relative_to(ROOT)}"]
            ),
            "wrist_ids": (
                sorted({row["wrist_pose_id"] for row in trial_metrics})
                if not aggregate else sorted({row["wrist_pose_id"] for row in metrics})
            ),
            "type": kind, "observation": observation,
        })

    # V1
    diagnostic_trial = trials[diagnostic_pose_metric["trial_id"]]
    actual_step = int(diagnostic_pose_metric["fixture_release_timestep"]) - 50
    actual_path = FIGURE_DIR / "actual_wrist_B_geometry.pdf"
    _save_three_views(
        actual_path, captures[diagnostic_pose_metric["trial_id"]]["frames"], actual_step,
        "Actual Phase 2W Allegro-hand / B geometry",
        f"B shown at exact trial pose | wrist RPY {diagnostic_trial['wrist_pose']['relative_rpy_deg']} deg | arrows: palm normal (red), gravity (black)",
    )
    record(actual_path.name, "Actual MuJoCo hand, table, B, candidate region, wrist and world vectors in three views.", [diagnostic_pose_metric], "raw simulation visualization", "The candidate B region lies adjacent to the colored index/thumb geometry at the highest-ranked diagnostic wrist pose.")

    # V2
    data = traces[best_dual["trial_id"]]
    panels = []
    for step in _safe_steps(dual_steps, total_steps(best_dual)):
        caption = _panel_text(data["arrays"], data["series"], step, int(best_dual["fixture_release_timestep"]))
        if step == int(best_dual["fixture_release_timestep"]):
            caption = "FIXTURE RELEASE\n" + caption
        panels.append((captures[best_dual["trial_id"]]["frames"][(step, OBLIQUE.name)], caption))
    dual_path = FIGURE_DIR / "representative_dual_contact_sequence.pdf"
    _image_grid(dual_path, "Representative pre-release dual-contact sequence", [("exact replay", panels)], 4)
    record(dual_path.name, "Eight fixed-camera frames around fixture release with contact and motion annotations.", [best_dual], "raw simulation visualization", f"This trial had {best_dual['dual_contact_survival_steps']} consecutive post-release dual-contact steps.")

    # V3/V4
    for label, metric, filename, title in (
        ("strict", best_strict, "best_strict_survival_sequence.pdf", "Longest strict-state survival sequence"),
        ("dual", best_dual, "best_dual_contact_survival_sequence.pdf", "Longest dual-contact survival sequence"),
    ):
        data = traces[metric["trial_id"]]
        steps = list(sequence_steps[label])
        if captures[metric["trial_id"]]["first_dual_step"] is not None:
            steps.append(captures[metric["trial_id"]]["first_dual_step"])
        steps = sorted(set(steps))
        panels = [(captures[metric["trial_id"]]["frames"][(step, OBLIQUE.name)], _panel_text(data["arrays"], data["series"], step, int(metric["fixture_release_timestep"]))) for step in steps]
        path = FIGURE_DIR / filename
        _image_grid(path, f"{title} - first failure: {metric['first_strict_failure_mechanism']}", [("exact replay", panels)], 4)
        record(filename, "Approach-through-failure fixed-camera sequence including release-adjacent frames.", [metric], "raw simulation visualization", f"Measured strict/dual survival was {metric['strict_survival_steps']}/{metric['dual_contact_survival_steps']} steps.")

    # V5
    failure_files = {
        "NO_B_CONTACT_BEFORE_RELEASE": "failure_example_no_contact.pdf",
        "B_ROTATED_OUT": "failure_example_rotation.pdf",
        "B_SLIPPED_TO_TABLE": "failure_example_slip.pdf",
        "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE": "failure_example_contact_loss.pdf",
    }
    for mode, filename in failure_files.items():
        rows = []
        for row_index, metric in enumerate(failure_examples[mode]):
            data = traces[metric["trial_id"]]
            panels = [(captures[metric["trial_id"]]["frames"][(step, OBLIQUE.name)], _panel_text(data["arrays"], data["series"], step, int(metric["fixture_release_timestep"]))) for step in failure_steps[mode][metric["trial_id"]]]
            rows.append(("median" if row_index == 0 else "late", panels))
        path = FIGURE_DIR / filename
        _image_grid(path, f"Deterministic {mode} examples", rows, 4)
        record(filename, "Median-time and late deterministic examples for one Phase 2W failure category.", failure_examples[mode], "raw simulation visualization", f"Panels show the measured {mode} event without changing the trial.")

    # V6
    metric = best_dual; data = traces[metric["trial_id"]]
    contact_panels = []
    for step in _safe_steps(contact_steps, total_steps(metric)):
        force = data["arrays"]["B_per_finger_normal_force_N"][step]
        caption = _panel_text(data["arrays"], data["series"], step, int(metric["fixture_release_timestep"])) + f"\nindex {force[0]:.3f} N | thumb {force[3]:.3f} N"
        contact_panels.append((captures[metric["trial_id"]]["frames"][(step, CLOSE_UP.name)], caption))
    contact_path = FIGURE_DIR / "contact_geometry_around_release.pdf"
    _image_grid(
        contact_path,
        "Actual contacts and force-scaled inward normals (yellow = index, magenta = thumb)",
        [("close-up", contact_panels)],
        5,
    )
    record(contact_path.name, "Close-up actual MuJoCo contacts; colored points/arrows identify fingers and force magnitude.", [metric], "raw simulation visualization", "Arrows are drawn only where MuJoCo reports contact.")

    # V7-V12
    _force_traces(FIGURE_DIR / "representative_force_traces.pdf", top_strict[:5], traces, cfg25)
    record("representative_force_traces.pdf", "Index/thumb force and tangential-normal traces for five longest strict-state trials.", top_strict[:5], "statistical visualization", "All five traces use time relative to their exact fixture-release step.")
    _motion_traces(FIGURE_DIR / "representative_motion_traces.pdf", top_strict[:5], traces, cfg25)
    record("representative_motion_traces.pdf", "Translation, rotation and vertical displacement with measured event markers.", top_strict[:5], "statistical visualization", "Motion is referenced to each trial's B pose immediately before release.")
    _contact_timeline(FIGURE_DIR / "contact_state_timeline.pdf", top_dual[:10], traces)
    record("contact_state_timeline.pdf", "Six-state contact raster for ten longest dual-contact trials.", top_dual[:10], "statistical visualization", "The timeline exposes which digit or table contact changes first.")
    _survival_distribution(FIGURE_DIR / "survival_duration_distribution.pdf", metrics)
    record("survival_duration_distribution.pdf", "Histograms and empirical CDFs over all 8,192 trials.", [], "statistical visualization", "Reference lines at 25-500 steps are descriptive and no preferred threshold is selected.")
    funnel_counts = _funnel(FIGURE_DIR / "acquisition_retention_funnel.pdf", metrics)
    record("acquisition_retention_funnel.pdf", "Exact measured counts from geometric sampling through unsupported strict-state retention.", [], "statistical visualization", f"The waterfall starts with {len(metrics):,} existing candidates and labels intermediate stages diagnostically.")
    wrist_rows = _wrist_diagnostics(FIGURE_DIR / "wrist_dynamic_diagnostics.pdf", metrics, trials)
    record("wrist_dynamic_diagnostics.pdf", "Retrospective physical performance for all ten dynamically tested wrist poses.", [], "statistical visualization", "The figure reports geometry, contact and failure measurements without selecting a new wrist.")

    # V13/V14
    endpoints = _endpoint_records(diagnostic_pose_id)
    endpoint_path = FIGURE_DIR / "fingertip_vs_palmar_same_wrist.pdf"
    _endpoint_comparison(endpoint_path, endpoints, diagnostic_trial["candidate_region"], base_cfg)
    endpoint_metrics = [diagnostic_pose_metric]
    record(endpoint_path.name, "Validated FINGERTIP and PALMAR endpoints at identical view, scale and diagnostic wrist.", endpoint_metrics, "raw simulation visualization", "Both rendered states retain middle+ring occupancy with index+thumb free; no A+B outcome is implied.")
    palm_path = FIGURE_DIR / "palm_space_metric_diagnostic.pdf"
    _palm_space_plot(palm_path, endpoints, phase2.resources, base_cfg)
    record(palm_path.name, "Configured volume, occupied voxels, largest connected free region, A geometry/COM and inscribed center.", endpoint_metrics, "statistical spatial visualization", "Nearly equal scalar free volume coexists with visibly different A placement.")

    # V16/V18
    entries = []
    for category, metric in contact_entries:
        thumbnail = captures[metric["trial_id"]]["thumbnail"]
        if thumbnail is None:
            raise RuntimeError(f"missing contact-sheet thumbnail for {metric['trial_id']}")
        entries.append((category, metric, thumbnail))
    contact_paths = _contact_sheet(entries)
    _visual_index(ROOT / "docs" / "PHASE2H_VISUAL_INDEX.md", records, special, videos, contact_paths)

    result = {
        "status": "PHASE2H_VISUAL_DIAGNOSTICS_COMPLETE",
        "source_phase2W_summary": str(summary_path.relative_to(ROOT)),
        "source_phase2H_metrics": str(metric_path.relative_to(ROOT)),
        "pdf_count": len(records),
        "videos": {key: str(value.relative_to(ROOT)) for key, value in videos.items()},
        "contact_sheet_html": str(contact_paths[0].relative_to(ROOT)),
        "contact_sheet_png": str(contact_paths[1].relative_to(ROOT)),
        "special_trials": {key: value["trial_id"] for key, value in special.items()},
        "funnel_counts": funnel_counts,
        "wrist_diagnostics": wrist_rows,
    }
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "visual_diagnostics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
