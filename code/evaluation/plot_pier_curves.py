# -*- coding: utf-8 -*-
"""Compare B0 (no sensors) and B1 (with sensors) on four pier-top nodes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NODE_IDS = (143, 144, 145, 146)
DIRECTIONS = ("X", "Y", "Z")


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-dir", type=Path, default=package / "results" / "B0" / "test" / "predicted_npz")
    parser.add_argument("--b1-dir", type=Path, default=package / "results" / "B1" / "test" / "predicted_npz")
    parser.add_argument("--output-dir", type=Path, default=package / "results" / "curves")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-records", action="store_true")
    return parser.parse_args()


def scalar(value: object) -> str:
    value = np.asarray(value).reshape(()).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def record_index(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            result[scalar(data["record_name"])] = path
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    error = prediction - truth
    if truth.size == 0 or np.std(truth) < 1.0e-12 or np.std(prediction) < 1.0e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    peak = float(np.max(np.abs(truth)))
    return {
        "rmse_m_s2": float(np.sqrt(np.mean(error * error))),
        "correlation": correlation,
        "peak_relative_error_pct": (float(np.max(np.abs(prediction))) - peak) / max(peak, 1.0e-12) * 100.0,
    }


def safe_name(record: str) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in record)[:70]
    return f"{clean}_{hashlib.sha1(record.encode('utf-8')).hexdigest()[:8]}"


def load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def plot_record(record: str, b0: dict[str, np.ndarray], b1: dict[str, np.ndarray], path: Path) -> None:
    time = np.asarray(b0["time"], dtype=np.float64)
    node_ids = np.asarray(b0["node_ids"], dtype=int)
    row_map = {int(node): row for row, node in enumerate(node_ids)}
    truth = np.asarray(b0["true_accel_abs"], dtype=np.float64)
    pred0 = np.asarray(b0["prediction_accel_abs"], dtype=np.float64)
    pred1 = np.asarray(b1["prediction_accel_abs"], dtype=np.float64)
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=True)
    for row, node in enumerate(NODE_IDS):
        for col, direction in enumerate(DIRECTIONS):
            axis = axes[row, col]
            if node not in row_map:
                axis.text(0.5, 0.5, f"node {node} missing", ha="center", va="center")
                axis.set_axis_off()
                continue
            node_row = row_map[node]
            axis.plot(time, truth[node_row, :, col], color="black", lw=1.0, label="True")
            axis.plot(time, pred0[node_row, :, col], color="#d95f02", lw=0.8, label="B0 no sensor")
            axis.plot(time, pred1[node_row, :, col], color="#1b9e77", lw=0.8, label="B1 sensor")
            axis.set_title(f"Pier-top node {node} — {direction}")
            axis.grid(alpha=0.25)
            if row == 3:
                axis.set_xlabel("Time (s)")
            if col == 0:
                axis.set_ylabel("Acceleration (m/s²)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(record, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    b0_index = record_index(args.b0_dir.resolve())
    b1_index = record_index(args.b1_dir.resolve())
    common = sorted(set(b0_index) & set(b1_index))
    if not common:
        raise FileNotFoundError("No common B0/B1 predicted_npz records found.")
    if args.limit is not None:
        common = common[: max(0, args.limit)]
    elif not args.all_records:
        common = common[:1]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for record in common:
        b0 = load_result(b0_index[record])
        b1 = load_result(b1_index[record])
        plot_record(record, b0, b1, output / "plots" / f"{safe_name(record)}.png")
        node_ids = np.asarray(b0["node_ids"], dtype=int)
        row_map = {int(node): row for row, node in enumerate(node_ids)}
        truth = np.asarray(b0["true_accel_abs"], dtype=np.float64)
        for node in NODE_IDS:
            if node not in row_map:
                continue
            for col, direction in enumerate(DIRECTIONS):
                for label, prediction in (("B0_no_sensor", b0["prediction_accel_abs"]), ("B1_sensor", b1["prediction_accel_abs"])):
                    item = metrics(truth[row_map[node], :, col], np.asarray(prediction)[row_map[node], :, col])
                    rows.append({"record": record, "node_id": node, "direction": direction, "model": label, **item})

    csv_path = output / "pier_curve_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {"records_plotted": common, "nodes": list(NODE_IDS), "directions": list(DIRECTIONS), "metrics_csv": str(csv_path)}
    for model in ("B0_no_sensor", "B1_sensor"):
        selected = [row for row in rows if row["model"] == model]
        summary[model] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("rmse_m_s2", "correlation", "peak_relative_error_pct")}
    (output / "pier_curve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PIER CURVE COMPARISON PASS\nRecords plotted: {len(common)}\nOutput: {output}")


if __name__ == "__main__":
    main()
