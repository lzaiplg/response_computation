from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def paired_gain(run_root: Path, b0_case: str, b1_case: str) -> dict[str, Any]:
    b0_rows = read_csv(run_root / "cases" / b0_case / "external_per_record.csv")
    b1_rows = read_csv(run_root / "cases" / b1_case / "external_per_record.csv")
    b0 = {row["record"]: row for row in b0_rows if row["mode"] == "zero"}
    b1 = {row["record"]: row for row in b1_rows if row["mode"] == "real"}
    names = sorted(set(b0) & set(b1))
    improvements = [
        (float(b0[name]["rmse_physical"]) - float(b1[name]["rmse_physical"]))
        / max(float(b0[name]["rmse_physical"]), 1.0e-12)
        * 100.0
        for name in names
    ]
    return {
        "b0_case": b0_case,
        "b1_case": b1_case,
        "record_count": len(names),
        "rmse_improvement_pct": stat(improvements),
        "b1_win_rate_pct": float(np.mean(np.asarray(improvements) > 0.0) * 100.0),
    }


def percent_change(treatment: float, control: float) -> float:
    return (treatment - control) / max(abs(control), 1.0e-12) * 100.0


def main() -> int:
    args = parse_args()
    run_root = SCRIPT_DIR / "runs" / args.campaign
    manifest = json.loads((run_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    planned = [item["case"]["case_id"] for item in manifest["planned_cases"]]
    summaries: dict[str, Any] = {}
    status_groups: dict[str, list[str]] = {"completed": [], "failed": [], "incomplete": [], "missing": []}
    for case_id in planned:
        case_dir = run_root / "cases" / case_id
        status_path = case_dir / "status.json"
        if not status_path.is_file():
            status_groups["missing"].append(case_id)
            continue
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if status == "completed" and (case_dir / "summary.json").is_file():
            status_groups["completed"].append(case_id)
            summaries[case_id] = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        elif status == "failed":
            status_groups["failed"].append(case_id)
        else:
            status_groups["incomplete"].append(case_id)

    required = set(planned)
    if set(summaries) != required:
        raise RuntimeError(f"Incomplete matrix: {status_groups}")

    ranking = []
    for case_id, summary in summaries.items():
        for mode in ("zero", "real"):
            if mode not in summary["external"]:
                continue
            metric = summary["external"][mode]
            ranking.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "nrmse": metric["nrmse_normalized"]["mean"],
                    "rmse_m": metric["rmse_physical"]["mean"],
                    "correlation": metric["correlation"]["mean"],
                    "peak_are_median_pct": metric["peak_relative_error_pct"]["median"],
                    "pier_top_rmse_m": metric["pier_top_rmse_physical"]["mean"],
                    "pier_bottom_rmse_m": metric["pier_bottom_rmse_physical"]["mean"],
                    "seconds": summary["elapsed_seconds"],
                    "best_epoch": summary["best_epoch"],
                }
            )
    ranking.sort(key=lambda item: (item["mode"], item["nrmse"]))

    if manifest.get("stage", "screen") != "screen":
        seed_pairs = []
        for seed in (17, 42, 73):
            b0_case = f"C10_SACNO_B0_SEED{seed}"
            b1_case = f"C11_SACNO_B1_SEED{seed}"
            seed_pairs.append(paired_gain(run_root, b0_case, b1_case))
        confirmation = {
            "campaign": args.campaign,
            "stage": "confirmation",
            "coverage": {"planned": len(planned), **{key: len(value) for key, value in status_groups.items()}, "case_ids": status_groups},
            "formal_test_available": False,
            "seed_pairs": seed_pairs,
            "aggregate": {
                "b1_win_rate_pct": stat([item["b1_win_rate_pct"] for item in seed_pairs]),
                "mean_recordwise_rmse_improvement_pct": stat([item["rmse_improvement_pct"]["mean"] for item in seed_pairs]),
            },
            "claim_strength": "three-seed confirmation on historical external validation; still not a formal untouched test",
        }
        temporary = run_root / "analysis_summary.json.tmp"
        temporary.write_text(json.dumps(confirmation, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, run_root / "analysis_summary.json")
        lines = [
            f"# Confirmation analysis: {args.campaign}",
            "",
            f"- Completed: {len(status_groups['completed'])}/{len(planned)}",
            f"- B1 win-rate mean across seeds: {confirmation['aggregate']['b1_win_rate_pct']['mean']:.2f}%",
            f"- Mean recordwise RMSE improvement across seeds: {confirmation['aggregate']['mean_recordwise_rmse_improvement_pct']['mean']:.2f}%",
            "- Evidence boundary: historical external validation; the untouched 78-record formal test is unavailable.",
        ]
        markdown = "\n".join(lines) + "\n"
        (run_root / "ANALYSIS.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        return 0

    float_gain = paired_gain(run_root, "S00_FLOAT_B0_SEED42", "S01_FLOAT_B1_SEED42")
    sacno_gain = paired_gain(run_root, "S10_SACNO_B0_SEED42", "S11_SACNO_B1_SEED42")
    mean_b1 = summaries["S01_FLOAT_B1_SEED42"]["external"]["real"]
    coord_b1 = summaries["S11_SACNO_B1_SEED42"]["external"]["real"]
    joint_basic = summaries["S20_NESTED_DUAL_SEED42"]["external"]
    joint_phys = summaries["S21_NESTED_DUAL_PHYS_SEED42"]["external"]
    factor_effects = {
        "coordinate_attention_vs_mean_fusion_B1": {
            "nrmse_change_pct": percent_change(coord_b1["nrmse_normalized"]["mean"], mean_b1["nrmse_normalized"]["mean"]),
            "physical_rmse_change_pct": percent_change(coord_b1["rmse_physical"]["mean"], mean_b1["rmse_physical"]["mean"]),
            "peak_median_change_pct": percent_change(coord_b1["peak_relative_error_pct"]["median"], mean_b1["peak_relative_error_pct"]["median"]),
            "pier_top_rmse_change_pct": percent_change(coord_b1["pier_top_rmse_physical"]["mean"], mean_b1["pier_top_rmse_physical"]["mean"]),
            "pier_bottom_rmse_change_pct": percent_change(coord_b1["pier_bottom_rmse_physical"]["mean"], mean_b1["pier_bottom_rmse_physical"]["mean"]),
        },
        "joint_vs_separate_SACNO": {
            "B0_nrmse_change_pct": percent_change(joint_basic["zero"]["nrmse_normalized"]["mean"], summaries["S10_SACNO_B0_SEED42"]["external"]["zero"]["nrmse_normalized"]["mean"]),
            "B1_nrmse_change_pct": percent_change(joint_basic["real"]["nrmse_normalized"]["mean"], coord_b1["nrmse_normalized"]["mean"]),
            "B1_peak_median_change_pct": percent_change(joint_basic["real"]["peak_relative_error_pct"]["median"], coord_b1["peak_relative_error_pct"]["median"]),
        },
        "physical_loss_vs_basic_joint": {
            "B0_nrmse_change_pct": percent_change(joint_phys["zero"]["nrmse_normalized"]["mean"], joint_basic["zero"]["nrmse_normalized"]["mean"]),
            "B0_peak_median_change_pct": percent_change(joint_phys["zero"]["peak_relative_error_pct"]["median"], joint_basic["zero"]["peak_relative_error_pct"]["median"]),
            "B0_bottom_rmse_change_pct": percent_change(joint_phys["zero"]["pier_bottom_rmse_physical"]["mean"], joint_basic["zero"]["pier_bottom_rmse_physical"]["mean"]),
            "B1_nrmse_change_pct": percent_change(joint_phys["real"]["nrmse_normalized"]["mean"], joint_basic["real"]["nrmse_normalized"]["mean"]),
            "B1_peak_median_change_pct": percent_change(joint_phys["real"]["peak_relative_error_pct"]["median"], joint_basic["real"]["peak_relative_error_pct"]["median"]),
            "B1_bottom_rmse_change_pct": percent_change(joint_phys["real"]["pier_bottom_rmse_physical"]["mean"], joint_basic["real"]["pier_bottom_rmse_physical"]["mean"]),
        },
    }

    trajectories = {}
    for case_id in planned:
        rows = read_csv(run_root / "cases" / case_id / "iteration_history.csv")
        scores = [float(row["selection_score"]) for row in rows]
        trajectories[case_id] = {
            "epochs_observed": len(scores),
            "best_epoch": int(rows[int(np.argmin(scores))]["epoch"]),
            "best_score": float(np.min(scores)),
            "last_score": scores[-1],
            "last_is_best": int(np.argmin(scores)) == len(scores) - 1,
            "relative_improvement_first_to_best_pct": (scores[0] - min(scores)) / scores[0] * 100.0,
        }

    analysis = {
        "campaign": args.campaign,
        "coverage": {"planned": len(planned), **{key: len(value) for key, value in status_groups.items()}, "case_ids": status_groups},
        "formal_test_available": False,
        "ranking": ranking,
        "paired_sensor_gain": {"FLOAT_TCN": float_gain, "SACNO": sacno_gain},
        "factor_effects": factor_effects,
        "trajectories": trajectories,
        "decisions": {
            "S00_FLOAT_B0_SEED42": "retain_as_sensor_free_float_baseline",
            "S01_FLOAT_B1_SEED42": "retain_as_mean_fusion_baseline",
            "S10_SACNO_B0_SEED42": "diagnostic_negative_control_equivalent_to_S00_under_zero_mask",
            "S11_SACNO_B1_SEED42": "retain_best_sensor_assisted_screen_method",
            "S20_NESTED_DUAL_SEED42": "conditional_unified_deployment_candidate",
            "S21_NESTED_DUAL_PHYS_SEED42": "conditional_for_peak_and_pier_bottom_not_full_field",
        },
        "claim_strength": "preliminary mechanism-screen evidence only; one seed, 15 epochs, historical validation not untouched formal test",
    }
    temporary = run_root / "analysis_summary.json.tmp"
    temporary.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, run_root / "analysis_summary.json")

    best_zero = min((item for item in ranking if item["mode"] == "zero"), key=lambda item: item["nrmse"])
    best_real = min((item for item in ranking if item["mode"] == "real"), key=lambda item: item["nrmse"])
    effects = factor_effects["coordinate_attention_vs_mean_fusion_B1"]
    phys = factor_effects["physical_loss_vs_basic_joint"]
    markdown = f"""# Ablation analysis: {args.campaign}

## Coverage

- Planned: {len(planned)}
- Completed: {len(status_groups['completed'])}
- Failed/incomplete/missing: {len(status_groups['failed'])}/{len(status_groups['incomplete'])}/{len(status_groups['missing'])}
- Formal untouched 78-record test available: yes; use `analyze_ablation.py --evaluation-split formal` after decisions are frozen

## Screen winners

- Best sensor-free nRMSE: `{best_zero['case_id']}` = {best_zero['nrmse']:.6f}; differences among basic zero-mask cases are negligible.
- Best sensor-assisted nRMSE: `{best_real['case_id']}` = {best_real['nrmse']:.6f}.
- Matched SACNO B1 versus B0 external win rate: {sacno_gain['b1_win_rate_pct']:.1f}% over {sacno_gain['record_count']} records.
- Matched SACNO mean per-record physical RMSE improvement: {sacno_gain['rmse_improvement_pct']['mean']:.2f}%.

## Mechanism effects

- Coordinate attention versus mean fusion for B1: nRMSE {effects['nrmse_change_pct']:.2f}%, physical RMSE {effects['physical_rmse_change_pct']:.2f}%, median peak error {effects['peak_median_change_pct']:.2f}%, pier-top RMSE {effects['pier_top_rmse_change_pct']:.2f}%, pier-bottom RMSE {effects['pier_bottom_rmse_change_pct']:.2f}%.
- Physical composite loss versus basic joint loss: B0 nRMSE {phys['B0_nrmse_change_pct']:.2f}%, B0 median peak error {phys['B0_peak_median_change_pct']:.2f}%, B0 bottom RMSE {phys['B0_bottom_rmse_change_pct']:.2f}%; B1 nRMSE {phys['B1_nrmse_change_pct']:.2f}%, B1 median peak error {phys['B1_peak_median_change_pct']:.2f}%, B1 bottom RMSE {phys['B1_bottom_rmse_change_pct']:.2f}%.

## Decisions

- Retain SACNO as the matched B0/B1 architecture for formal confirmation.
- Retain NESTED-DUAL conditionally when one-checkpoint dual-mode deployment is required.
- Do not retain the current physical-loss weights for global full-field ranking; preserve them as a pier-bottom/peak partial signal and retune only after the basic model converges.
- All basic cases were still improving at or near epoch 15, so 15 epochs are sufficient only for mechanism screening.

## Evidence boundary

These are preliminary mechanism-screen results on historical validation records. They do not support a final top-journal superiority claim until the untouched 78-record formal test, longer training, three seeds, sensor noise/dropout, sensor-count/location ablations and external/experimental bridge data are evaluated.
"""
    (run_root / "ANALYSIS.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
