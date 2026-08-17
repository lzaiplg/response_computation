from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mode_for(case: dict[str, Any]) -> str:
    return "zero" if case["sensor_training"] == "separate_zero" else "real"


def metric(summary: dict[str, Any], mode: str, name: str) -> float:
    return float(summary["external"][mode][name]["mean"])


def paired_gain(run_root: Path, b0_case: str, b1_case: str) -> dict[str, Any]:
    b0_rows = {row["record"]: row for row in read_csv(run_root / "cases" / b0_case / "external_per_record.csv")}
    b1_rows = {row["record"]: row for row in read_csv(run_root / "cases" / b1_case / "external_per_record.csv")}
    names = sorted(set(b0_rows) & set(b1_rows))
    improvements = np.asarray(
        [
            (float(b0_rows[name]["rmse_physical"]) - float(b1_rows[name]["rmse_physical"]))
            / max(float(b0_rows[name]["rmse_physical"]), 1.0e-12)
            * 100.0
            for name in names
        ],
        dtype=np.float64,
    )
    return {
        "b0_case": b0_case,
        "b1_case": b1_case,
        "record_count": len(names),
        "mean_improvement_pct": float(np.mean(improvements)),
        "median_improvement_pct": float(np.median(improvements)),
        "p05_improvement_pct": float(np.percentile(improvements, 5)),
        "p95_improvement_pct": float(np.percentile(improvements, 95)),
        "win_rate_pct": float(np.mean(improvements > 0.0) * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    run_root = SCRIPT_DIR / "runs" / args.campaign
    manifest = json.loads((run_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    planned = [item["case"] for item in manifest["planned_cases"]]
    summaries: dict[str, dict[str, Any]] = {}
    status_groups = {key: [] for key in ("completed", "failed", "incomplete", "missing")}
    for case in planned:
        case_id = case["case_id"]
        case_dir = run_root / "cases" / case_id
        status_path = case_dir / "status.json"
        summary_path = case_dir / "summary.json"
        if not status_path.exists():
            status_groups["missing"].append(case_id)
        elif json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed" and summary_path.exists():
            status_groups["completed"].append(case_id)
            summaries[case_id] = json.loads(summary_path.read_text(encoding="utf-8"))
        elif json.loads(status_path.read_text(encoding="utf-8")).get("status") == "failed":
            status_groups["failed"].append(case_id)
        else:
            status_groups["incomplete"].append(case_id)
    if len(summaries) != len(planned):
        raise RuntimeError(f"Incomplete matrix: {status_groups}")

    rows = []
    for case in planned:
        case_id = case["case_id"]
        summary = summaries[case_id]
        mode = mode_for(case)
        rows.append(
            {
                "case_id": case_id,
                "strategy": case["strategy"],
                "architecture": case.get("architecture", "sacno"),
                "mode": mode,
                "seed": case["seed"],
                "loss_profile": case["loss_profile"],
                "sensor_noise_std": case.get("sensor_noise_std", 0.0),
                "sensor_dropout_prob": case.get("sensor_dropout_prob", 0.0),
                "direction_score_weight": case.get("direction_score_weight", 0.0),
                "nrmse": metric(summary, mode, "nrmse_normalized"),
                "rmse_physical": metric(summary, mode, "rmse_physical"),
                "peak_abs_error_physical": metric(summary, mode, "peak_absolute_error_physical"),
                "peak_are_median_pct": float(summary["external"][mode]["peak_relative_error_pct"]["median"]),
                "correlation": metric(summary, mode, "correlation"),
                "pier_top_rmse_physical": metric(summary, mode, "pier_top_rmse_physical"),
                "pier_bottom_rmse_physical": metric(summary, mode, "pier_bottom_rmse_physical"),
                "max_direction_nrmse": metric(summary, mode, "max_direction_nrmse_normalized"),
                "max_direction_rmse_physical": metric(summary, mode, "max_direction_rmse_physical"),
                "elapsed_minutes": summary["elapsed_seconds"] / 60.0,
                "best_epoch": summary["best_epoch"],
            }
        )

    strategy_summary: dict[str, Any] = {}
    for strategy in sorted({case["strategy"] for case in planned}):
        strategy_cases = [case for case in planned if case["strategy"] == strategy]
        item: dict[str, Any] = {"cases": [case["case_id"] for case in strategy_cases]}
        for mode in ("zero", "real"):
            selected = [row for row in rows if row["strategy"] == strategy and row["mode"] == mode]
            if not selected:
                continue
            item[mode] = {
                key: stat([float(row[key]) for row in selected])
                for key in (
                    "nrmse",
                    "rmse_physical",
                    "peak_abs_error_physical",
                    "peak_are_median_pct",
                    "correlation",
                    "pier_top_rmse_physical",
                    "pier_bottom_rmse_physical",
                    "max_direction_nrmse",
                    "max_direction_rmse_physical",
                    "elapsed_minutes",
                )
            }
        b0 = next((case["case_id"] for case in strategy_cases if mode_for(case) == "zero"), None)
        b1 = next((case["case_id"] for case in strategy_cases if mode_for(case) == "real"), None)
        if b0 and b1:
            item["paired_sensor_gain"] = paired_gain(run_root, b0, b1)
        robust = []
        for case in strategy_cases:
            robust_summary = summaries[case["case_id"]].get("external_robust")
            if robust_summary:
                mode = mode_for(case)
                robust.append(
                    {
                        "case_id": case["case_id"],
                        "mode": mode,
                        "nrmse": float(robust_summary[mode]["nrmse_normalized"]["mean"]),
                        "rmse_physical": float(robust_summary[mode]["rmse_physical"]["mean"]),
                        "peak_abs_error_physical": float(robust_summary[mode]["peak_absolute_error_physical"]["mean"]),
                        "peak_are_median_pct": float(robust_summary[mode]["peak_relative_error_pct"]["median"]),
                    }
                )
        if robust:
            item["robust_external"] = robust
        strategy_summary[strategy] = item

    coverage = {"planned": len(planned), **{key: len(value) for key, value in status_groups.items()}, "case_ids": status_groups}
    analysis = {
        "campaign": args.campaign,
        "target": manifest.get("target"),
        "coverage": coverage,
        "formal_test_available": manifest.get("formal_test_available", False),
        "ranking": sorted(rows, key=lambda row: (row["mode"], row["nrmse"])),
        "strategies": strategy_summary,
        "claim_strength": "mechanism screen; external split is historical validation, not the untouched formal test",
    }
    (run_root / "analysis_summary.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (run_root / "final_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Strategy screen analysis: {args.campaign}",
        "",
        f"- Planned/completed/failed/incomplete/missing: {coverage['planned']}/{coverage['completed']}/{coverage['failed']}/{coverage['incomplete']}/{coverage['missing']}",
        "- External evaluation: historical 157-record split; the untouched formal test is unavailable.",
        "",
        "## Strategy ranking",
        "",
        "| Strategy | Mode | nRMSE | Physical RMSE | Worst-dir nRMSE | Peak ARE median | Pier-top RMSE | Time (min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda row: (row["mode"], row["nrmse"])):
        lines.append(
            f"| {row['strategy']} | {row['mode']} | {row['nrmse']:.6f} | {row['rmse_physical']:.6f} | {row['max_direction_nrmse']:.6f} | {row['peak_are_median_pct']:.3f}% | {row['pier_top_rmse_physical']:.6f} | {row['elapsed_minutes']:.1f} |"
        )
    lines.extend(["", "## Strategy decisions", ""])
    for strategy, item in strategy_summary.items():
        gain = item.get("paired_sensor_gain")
        gain_text = f"; B1 paired win {gain['win_rate_pct']:.1f}% and mean RMSE improvement {gain['mean_improvement_pct']:.2f}%" if gain else ""
        lines.append(f"- `{strategy}`{gain_text}.")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This is a mechanism screen. A candidate must pass a three-seed confirmation, matched multi-seed baseline, sensor noise/dropout tests and the untouched formal test before a strong superiority claim.",
        ]
    )
    (run_root / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
