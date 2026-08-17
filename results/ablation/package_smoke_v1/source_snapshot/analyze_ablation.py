from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic immutable-campaign ablation analysis.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--evaluation-split", choices=("formal", "external"), default="formal")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "results" / "ablation")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(section: dict[str, Any], mode: str, name: str, statistic: str = "mean") -> float | None:
    value = section.get(mode, {}).get(name)
    if not isinstance(value, dict):
        return None
    item = value.get(statistic)
    return float(item) if item is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_per_record(path: Path, mode: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            row["record"]: row
            for row in csv.DictReader(handle)
            if row.get("mode") == mode and row.get("record")
        }


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def group_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        summary.get("target"),
        summary.get("architecture", "sacno"),
        summary.get("fusion"),
        summary.get("loss_profile"),
        summary.get("seed"),
        summary.get("sensor_noise_std", 0.0),
        summary.get("sensor_dropout_prob", 0.0),
        summary.get("eval_noise_std", 0.0),
        summary.get("eval_dropout_prob", 0.0),
        summary.get("peak_score_weight", 0.0),
        summary.get("direction_score_weight", 0.0),
    )


def main() -> int:
    args = parse_args()
    run_root = (args.output_root / args.campaign).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    case_items: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_dir in sorted((run_root / "cases").iterdir() if (run_root / "cases").is_dir() else []):
        summary_path = case_dir / "summary.json"
        if not summary_path.is_file():
            missing.append(case_dir.name)
            continue
        summary = read_json(summary_path)
        if summary.get("status") != "completed":
            missing.append(case_dir.name)
            continue
        section = summary.get(args.evaluation_split)
        if not isinstance(section, dict):
            missing.append(f"{case_dir.name}:{args.evaluation_split}")
            continue
        case_items.append({"case_dir": case_dir, "summary": summary, "section": section})

    ranking: list[dict[str, Any]] = []
    for item in case_items:
        summary = item["summary"]
        section = item["section"]
        for mode in ("zero", "real"):
            if mode not in section:
                continue
            ranking.append(
                {
                    "case_id": summary["case_id"],
                    "mode": mode,
                    "target": summary["target"],
                    "architecture": summary.get("architecture", "sacno"),
                    "fusion": summary.get("fusion"),
                    "loss_profile": summary.get("loss_profile"),
                    "seed": summary.get("seed"),
                    "record_count": section.get("sensor_information_gain", {}).get("record_count"),
                    "rmse_physical_mean": metric(section, mode, "rmse_physical"),
                    "rmse_physical_p95": metric(section, mode, "rmse_physical", "p95"),
                    "correlation_mean": metric(section, mode, "correlation"),
                    "correlation_min": metric(section, mode, "correlation", "min"),
                    "peak_relative_error_pct_median": metric(section, mode, "peak_relative_error_pct", "median"),
                    "pier_top_rmse_physical_mean": metric(section, mode, "pier_top_rmse_physical"),
                    "X_rmse_physical_mean": metric(section, mode, "X_rmse_physical"),
                    "Y_rmse_physical_mean": metric(section, mode, "Y_rmse_physical"),
                    "Z_rmse_physical_mean": metric(section, mode, "Z_rmse_physical"),
                }
            )
    ranking.sort(key=lambda row: (float(row["rmse_physical_mean"]), -float(row["correlation_mean"])))

    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for item in case_items:
        training = item["summary"].get("sensor_training")
        if training in ("separate_zero", "separate_real"):
            groups.setdefault(group_key(item["summary"]), {})[training] = item
    paired: list[dict[str, Any]] = []
    for group, pair in groups.items():
        if "separate_zero" not in pair or "separate_real" not in pair:
            continue
        b0 = pair["separate_zero"]
        b1 = pair["separate_real"]
        b0_rows = load_per_record(b0["case_dir"] / f"{args.evaluation_split}_per_record.csv", "zero")
        b1_rows = load_per_record(b1["case_dir"] / f"{args.evaluation_split}_per_record.csv", "real")
        common = sorted(set(b0_rows) & set(b1_rows))
        if not common:
            continue
        improvements = [
            (as_float(b0_rows[name], "rmse_physical") - as_float(b1_rows[name], "rmse_physical"))
            / max(as_float(b0_rows[name], "rmse_physical"), 1.0e-12) * 100.0
            for name in common
        ]
        paired.append(
            {
                "b0_case_id": b0["summary"]["case_id"],
                "b1_case_id": b1["summary"]["case_id"],
                "architecture": group[1],
                "fusion": group[2],
                "loss_profile": group[3],
                "seed": group[4],
                "record_count": len(common),
                "b0_rmse_mean": sum(as_float(b0_rows[name], "rmse_physical") for name in common) / len(common),
                "b1_rmse_mean": sum(as_float(b1_rows[name], "rmse_physical") for name in common) / len(common),
                "b0_correlation_mean": sum(as_float(b0_rows[name], "correlation") for name in common) / len(common),
                "b1_correlation_mean": sum(as_float(b1_rows[name], "correlation") for name in common) / len(common),
                "b1_rmse_improvement_pct_mean": sum(improvements) / len(improvements),
                "b1_win_rate_pct": sum(value > 0.0 for value in improvements) / len(improvements) * 100.0,
                "formal_scope": args.evaluation_split == "formal",
            }
        )
    paired.sort(key=lambda row: (-float(row["b1_rmse_improvement_pct_mean"]), float(row["b1_rmse_mean"])))

    write_csv(run_root / f"ranking_{args.evaluation_split}.csv", ranking)
    write_csv(run_root / f"paired_{args.evaluation_split}.csv", paired)
    summary = {
        "campaign": args.campaign,
        "evaluation_split": args.evaluation_split,
        "formal_test_is_untouched": args.evaluation_split == "formal",
        "completed_case_count": len(case_items),
        "missing_or_incomplete": missing,
        "ranked_case_count": len(ranking),
        "paired_comparison_count": len(paired),
        "best_case": ranking[0] if ranking else None,
        "best_paired_comparison": paired[0] if paired else None,
    }
    (run_root / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Ablation analysis: {args.campaign}",
        "",
        f"Evaluation split: `{args.evaluation_split}`.",
        f"Completed cases: {len(case_items)}; ranked conditions: {len(ranking)}; matched B0/B1 pairs: {len(paired)}.",
        "",
        "The formal split is used only for final reporting after training and strategy selection. It is not used for checkpoint selection.",
        "" if not missing else f"Incomplete items: {', '.join(missing)}",
        "",
        f"See `ranking_{args.evaluation_split}.csv` for condition ranking and `paired_{args.evaluation_split}.csv` for matched B0/B1 gains.",
    ]
    (run_root / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
