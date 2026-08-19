#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import git_commit_sha
from seqgrasp.experiments.resource_components import free_palm_volume
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2tr_config import assert_index_thumb_free_topology, load_phase2tr_config


STOP_REASON = "No eligible common B region: the validated native index+thumb region systematically overlaps the endpoint hand, while 3,568 positive-epsilon mapped-region candidates produced zero strict B-only successes."


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _largest(root: Path, name: str) -> Path:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in root.rglob(name)]
    if not candidates:
        raise FileNotFoundError(f"missing {name} under {root}")
    return max(candidates)[2]


def _mean(rows, key):
    return float(statistics.fmean(float(row[key]) for row in rows))


def _metrics(rows):
    return {
        "count": len(rows),
        "ferrari_canny_epsilon_mean": _mean(rows, "ferrari_canny_epsilon"),
        "total_A_normal_force_N_mean": _mean(rows, "total_A_normal_force_N"),
        "translation_drift_m_mean": _mean(rows, "A_translation_drift_m"),
        "rotation_drift_rad_mean": _mean(rows, "A_rotation_drift_rad"),
        "minimum_joint_margin_rad_mean": _mean(rows, "minimum_joint_margin_rad"),
        "palm_contact_fraction_mean": _mean(rows, "palm_A_contact_fraction"),
        "COM_to_palm_surface_distance_m_mean": _mean(rows, "COM_to_palm_surface_distance_m"),
    }


def _find_summary(root: Path, predicate=lambda _: True):
    rows = []
    for path in root.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if predicate(payload):
            rows.append((path.stat().st_mtime_ns, path, payload))
    if not rows:
        raise FileNotFoundError(f"missing summary under {root}")
    return max(rows)[1:]


def _font():
    return "Helvetica"


def _header(c, title, subtitle, font):
    width, height = landscape(letter)
    c.setFillColor(colors.HexColor("#0B1F33")); c.rect(0, height - 62, width, 62, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont(font, 21); c.drawString(32, height - 37, title)
    c.setFillColor(colors.HexColor("#B9D7EA")); c.setFont(font, 9); c.drawRightString(width - 32, height - 36, subtitle)
    return width, height


def _footer(c, font):
    c.setFillColor(colors.HexColor("#5E6B75")); c.setFont(font, 8)
    c.drawString(32, 18, "Phase 2T-R - measured evidence only - no A+B formal outcomes")


def _panel(c, x, y, w, h, title, font, fill="#F4F7FA"):
    c.setFillColor(colors.HexColor(fill)); c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#16324F")); c.setFont(font, 13); c.drawString(x + 14, y + h - 23, title)


def _bar(c, x, y, w, value, maximum, color, label, font):
    c.setFillColor(colors.HexColor("#DDE5EB")); c.roundRect(x, y, w, 12, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(color)); c.roundRect(x, y, w * (value / maximum if maximum else 0), 12, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#263746")); c.setFont(font, 9); c.drawString(x, y + 17, label)


def _endpoints_pdf(path, evidence, font):
    c = canvas.Canvas(str(path), pagesize=landscape(letter), pageCompression=1)
    width, height = _header(c, "Equal-digit endpoint populations", "middle+ring occupied | index+thumb free", font)
    groups = [("FINGERTIP", evidence["endpoint_metrics"]["FINGERTIP"], 32, "#EAF3FA"), ("PALMAR", evidence["endpoint_metrics"]["PALMAR_SECURED"], 406, "#F4EEF9")]
    for name, metric, x, fill in groups:
        _panel(c, x, 82, 354, 430, name, font, fill)
        cx, cy = x + 175, 330
        c.setFillColor(colors.HexColor("#D8B384")); c.rect(cx - 25, cy - 25, 50, 50, fill=1, stroke=0)
        if name == "PALMAR":
            c.setFillColor(colors.HexColor("#7A8B99")); c.roundRect(cx - 85, cy - 45, 32, 90, 8, fill=1, stroke=0)
            c.setStrokeColor(colors.HexColor("#7A8B99")); c.setLineWidth(5); c.line(cx - 53, cy, cx - 27, cy)
        fingers = [("middle", -52, 42), ("ring", 52, 42), ("index free", -75, -65), ("thumb free", 75, -65)]
        for label, dx, dy in fingers:
            active = "free" not in label
            c.setFillColor(colors.HexColor("#2878B5" if active else "#B7C2CC"))
            c.circle(cx + dx, cy + dy, 12, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#263746")); c.setFont(font, 9); c.drawCentredString(cx + dx, cy + dy - 25, label)
        c.setFillColor(colors.HexColor("#263746")); c.setFont(font, 11)
        lines = [
            f"Validated states: {metric['count']}",
            f"Mean epsilon: {metric['ferrari_canny_epsilon_mean']:.4f}",
            f"Mean A force: {metric['total_A_normal_force_N_mean']:.3f} N",
            f"Mean drift: {1000*metric['translation_drift_m_mean']:.3f} mm / {metric['rotation_drift_rad_mean']:.3f} rad",
            f"Palm persistence: {metric['palm_contact_fraction_mean']:.3f}",
        ]
        for i, line in enumerate(lines): c.drawString(x + 24, 220 - i * 24, line)
    _footer(c, font); c.save()


def _b_only_pdf(path, evidence, font):
    c = canvas.Canvas(str(path), pagesize=landscape(letter), pageCompression=1)
    width, height = _header(c, "Native index+thumb B-only positive control", "No middle/ring assistance", font)
    _panel(c, 32, 286, 728, 225, "Strict gate", font)
    c.setFillColor(colors.HexColor("#1D6F42")); c.setFont(font, 44); c.drawString(58, 390, "5 / 12")
    c.setFillColor(colors.HexColor("#263746")); c.setFont(font, 13); c.drawString(58, 365, "strict successes - target reached")
    c.setFont(font, 11); c.drawString(350, 425, "Every success:")
    for i, line in enumerate(("index+thumb contact before release", "fixture fully released", "500 unsupported steps", "no table contact", "no middle/ring contact")):
        c.drawString(365, 399 - i * 25, "+  " + line)
    _panel(c, 32, 82, 728, 180, "Small-pose/yaw robustness", font)
    _bar(c, 58, 180, 650, 25, 100, "#2F80C1", "25 strict successes / 100 perturbations", font)
    _bar(c, 58, 130, 650, 73, 100, "#D46A4C", "73 slipped to table", font)
    _bar(c, 58, 90, 650, 2, 100, "#D7A23A", "2 rotated out", font)
    _footer(c, font); c.save()


def _main_pdf(path, evidence, font):
    c = canvas.Canvas(str(path), pagesize=landscape(letter), pageCompression=1)
    width, height = _header(c, "Phase 2T-R result: stopped before A+B calibration", "No formal comparison was run", font)
    stages = [
        ("Endpoint populations", "100 FINGERTIP\n102 PALMAR", "#1D6F42"),
        ("B-only gate", "5/12 strict\n25/100 robust", "#1D6F42"),
        ("Common region", "0 eligible\nregion", "#B5473C"),
        ("Calibration/formal", "Not run", "#7A8B99"),
    ]
    for i, (title, body, color) in enumerate(stages):
        x = 34 + i * 190
        c.setFillColor(colors.HexColor(color)); c.roundRect(x, 290, 150, 130, 10, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont(font, 12); c.drawCentredString(x + 75, 390, title)
        for j, line in enumerate(body.splitlines()): c.setFont(font, 16); c.drawCentredString(x + 75, 350 - j * 25, line)
        if i < 3:
            c.setStrokeColor(colors.HexColor("#7A8B99")); c.setLineWidth(2); c.line(x + 155, 355, x + 185, 355)
    _panel(c, 34, 78, 720, 175, "Why the common-region gate failed", font, "#FFF4EE")
    c.setFillColor(colors.HexColor("#263746")); c.setFont(font, 11)
    lines = [
        "Validated Phase 2S index+thumb region: 100% endpoint-hand initial overlap in both groups.",
        "Positive-epsilon mapped region 4358: 0/2,048 strict native index+thumb successes.",
        "Positive-epsilon mapped region 4454: 0/1,520 strict native index+thumb successes.",
        "The protocol forbids freezing a systematically overlapping region or using A+B outcomes to move it.",
    ]
    for i, line in enumerate(lines): c.drawString(55, 215 - i * 31, line)
    _footer(c, font); c.save()


def _failure_pdf(path, evidence, font):
    c = canvas.Canvas(str(path), pagesize=landscape(letter), pageCompression=1)
    width, height = _header(c, "Failure mechanisms before the common-region stop", "Counts are engineering-search diagnostics", font)
    panels = [
        ("FINGERTIP endpoint rejections", evidence["fingertip_search"]["failure_mechanisms"], 32),
        ("PALMAR endpoint rejections", evidence["palmar_search"]["failure_mechanisms"], 282),
        ("Mapped B-only failures", evidence["mapped_B_only_failures"], 532),
    ]
    for title, values, x in panels:
        _panel(c, x, 80, 228, 430, title, font)
        top = max(values.values()) if values else 1
        for i, (label, value) in enumerate(sorted(values.items(), key=lambda item: item[1], reverse=True)[:7]):
            y = 440 - i * 48
            _bar(c, x + 16, y, 196, value, top, "#D46A4C", f"{label}: {value}", font)
    _footer(c, font); c.save()


def main() -> int:
    phase2tr, _ = load_phase2tr_config()
    phase2, _ = load_phase2_config()
    f_path = _largest(ROOT / phase2tr.output_dir / "fingertip_states", "accepted_states.jsonl")
    p_path = _largest(ROOT / phase2tr.output_dir / "palmar_states", "accepted_states.jsonl")
    f_rows, p_rows = _jsonl(f_path)[:100], _jsonl(p_path)[:100]
    for row in f_rows + p_rows: assert_index_thumb_free_topology(row)
    f_summary_path, f_summary = _find_summary(ROOT / phase2tr.output_dir / "fingertip_states")
    p_summary_path, p_summary = _find_summary(ROOT / phase2tr.output_dir / "palmar_states")
    b_summary_path, b_summary = _find_summary(ROOT / phase2tr.output_dir / "b_only_index_thumb")
    geometry_path, geometry = _find_summary(ROOT / phase2tr.output_dir / "geometry")
    mapped = []
    for path in (ROOT / phase2tr.output_dir / "b_only_common_regions").rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapped.append(payload)
    mapped_by_pose = {int(row["mapper_pose_candidate_index"]): row for row in mapped}
    mapped_failures = Counter()
    for pose in (4168, 4358, 4454):
        mapped_failures.update(mapped_by_pose[pose]["failure_mechanisms"])
    cfg = load_configs(scene_filename=phase2tr.scene_filename)
    for rows in (f_rows, p_rows):
        for row in rows:
            enriched = dict(row); enriched["grasp_id"] = row["grasp_state_id"]
            row["free_palm_volume_m3"] = free_palm_volume(enriched, phase2.resources, cfg)
    evidence = {
        "status": "STOPPED_BEFORE_COMMON_B_REGION_FREEZE",
        "stop_reason": STOP_REASON,
        "git_sha": git_commit_sha(ROOT),
        "fingertip_search": f_summary,
        "palmar_search": p_summary,
        "endpoint_metrics": {"FINGERTIP": _metrics(f_rows), "PALMAR_SECURED": _metrics(p_rows)},
        "identity_assertion": "occupied=middle+ring and free=index+thumb in every included state",
        "B_only": b_summary,
        "invalid_original_region_geometry": geometry,
        "mapped_region_searches": {str(key): value for key, value in sorted(mapped_by_pose.items())},
        "mapped_B_only_candidate_total": sum(mapped_by_pose[pose]["candidate_count"] for pose in (4168, 4358, 4454)),
        "mapped_B_only_strict_success_total": sum(mapped_by_pose[pose]["strict_success_count"] for pose in (4168, 4358, 4454)),
        "mapped_B_only_failures": dict(mapped_failures),
        "free_palm_volume_m3_mean": {"FINGERTIP": _mean(f_rows, "free_palm_volume_m3"), "PALMAR_SECURED": _mean(p_rows, "free_palm_volume_m3")},
        "workspace": {key: value for key, value in geometry["groups"].items()},
        "calibration": "not run - no eligible common B region",
        "controller_freeze": "not created - dynamic calibration did not start",
        "matching": "not run",
        "formal": "not run",
        "interpretation": "No TR1-TR6 label is supported: populations and B-only control exist, but the protocol has no category for failure to identify a collision-compatible common B region before sequential calibration.",
    }
    output = ROOT / phase2tr.output_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    results = f"""# Phase 2T-R index+thumb control results

## Outcome

Phase 2T-R stopped before the common B distribution could be frozen. {STOP_REASON}

This is not a formal FINGERTIP-versus-PALMAR result. No A+B calibration, matching, formal trials, McNemar test, or bootstrap was run, and transfer was not simulated.

## Targeted endpoints

- FINGERTIP_INDEX_THUMB_FREE: {f_summary['valid_states']} valid / {f_summary['attempts']} attempts ({f_summary['acceptance_rate']:.4f}); {f_summary['local_seed_derived_states']} local and {f_summary['global_search_derived_states']} broader targeted states.
- PALMAR_INDEX_THUMB_FREE: {p_summary['valid_states']} valid / {p_summary['attempts']} attempts ({p_summary['acceptance_rate']:.4f}); {p_summary['local_seed_derived_states']} local and {p_summary['global_search_derived_states']} broader targeted states.
- Every included state passed the exact identity assertion: occupied={{middle, ring}}, free={{index, thumb}}.

## Index+thumb B-only evidence

- Native Phase 2S proposal revalidation/search: {b_summary['strict_success_count']}/{b_summary['candidate_count']} strict successes.
- Robustness: {b_summary['robustness_success_count']}/{b_summary['robustness_trial_count']} ({b_summary['robustness_success_fraction']:.3f}).
- Middle/ring assistance was rejected by construction and checked from the per-finger contact time series.

## Common-region gate

The native positive region had 5,000/5,000 initial hand-overlap placements in each group. Two alternative mapper regions with positive opposition/force-closure evidence produced 0/2,048 and 0/1,520 strict index+thumb B-only successes. Including the 512-candidate exploratory mapped region, the mapped search used {evidence['mapped_B_only_candidate_total']} candidates with zero strict successes. Together with the 12 native-region candidates, the Phase 2T-R B-only search used 4,092 candidates, within the 4,096 cap.

Because no region simultaneously met dynamic index+thumb graspability, nonzero access in both groups, and absence of systematic A/hand overlap, the B distribution and controller were not frozen. Later stages are not applicable.
"""
    preliminary = f"""# Phase 2T-R preliminary evidence

The targeted endpoint populations are feasible and not rare: FINGERTIP accepted {f_summary['valid_states']}/{f_summary['attempts']} and PALMAR accepted {p_summary['valid_states']}/{p_summary['attempts']}. Both fix free-digit count and identity.

Native index+thumb B-only acquisition is reproducible (5/12 strict; 25/100 under perturbation), but its validated location is geometrically incompatible with the newly targeted endpoint hand postures because every checked placement overlaps the hand. Positive-epsilon mapped regions did not yield a strict dynamic positive control within the remaining authorized budget.

The evidence therefore supports a topology-aware resource representation, but it does not support a PALMAR-versus-FINGERTIP sequential success comparison in this stage. Transfer was not simulated.
"""
    interpretation = f"""# Phase 2T-R interpretation

None of TR1-TR6 can be assigned from measured evidence.

- TR1 is false: both targeted endpoint populations exceed the minimum.
- TR2 is false as written: native index+thumb B-only control achieved strict successes.
- TR3 is not established: sequential A+B calibration was not run.
- TR4, TR5, and TR6 require formal outcomes, which do not exist.

The measured stop occurred between the B-only and calibration stages: no B region simultaneously retained strict index+thumb B-only evidence, nonzero shared endpoint access, and absence of systematic initial hand overlap. Assigning a comparative class would overstate the experiment.
"""
    resource = """# Phase 2T-R resource-topology evidence

Phase 2T established that free_finger_count=2 is insufficient: index+middle produced 0/4,096 strict B-only successes. Phase 2T-R fixed both count and identity and showed that middle+ring-supported endpoints leaving index+thumb free can be generated in both FINGERTIP and PALMAR groups.

The next resource representation must distinguish at least three noninterchangeable descriptions:

1. digit count;
2. digit identity and acquisition topology;
3. reachable workspace and collision-compatible placement.

Phase 2T-R adds direct evidence for the third distinction: index+thumb can acquire B in isolation, yet the validated positive region can be unusable from a particular retained-A endpoint because of initial hand overlap. No scalar aggregate or weights are defined here.
"""
    for path, text in (
        (ROOT / "docs" / "PHASE2TR_INDEX_THUMB_CONTROL_RESULTS.md", results),
        (ROOT / "docs" / "PHASE2TR_PRELIMINARY_EVIDENCE.md", preliminary),
        (ROOT / "docs" / "PHASE2TR_INTERPRETATION.md", interpretation),
        (ROOT / "docs" / "PHASE2TR_RESOURCE_TOPOLOGY_EVIDENCE.md", resource),
    ):
        path.write_text(text, encoding="utf-8")
    figures = ROOT / "docs" / "figures" / "phase2TR"; figures.mkdir(parents=True, exist_ok=True)
    font = _font()
    _endpoints_pdf(figures / "index_thumb_free_endpoints.pdf", evidence, font)
    _b_only_pdf(figures / "index_thumb_b_only_control.pdf", evidence, font)
    _main_pdf(figures / "phase2TR_main_result.pdf", evidence, font)
    _failure_pdf(figures / "failure_modes.pdf", evidence, font)
    print(json.dumps({"status": evidence["status"], "analysis": str(output / "evidence.json"), "figures": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
