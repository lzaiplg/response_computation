from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ablation_data import BridgeResponseDataset, fixed_geometry, load_mapping, target_scales
from ablation_model import OperatorConfig, SensorAvailabilityConditionedOperator


PIER_TOP_NODE_IDS = (143, 144, 145, 146)
DIRECTIONS = ("X", "Y", "Z")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def choose_record(case_dir: Path, requested: str | None) -> str:
    rows = read_rows(case_dir / "external_per_record.csv")
    names = {row["record"] for row in rows}
    if requested:
        if requested not in names:
            raise ValueError(f"Record not found in {case_dir}: {requested}")
        return requested
    return min(rows, key=lambda row: float(row["nrmse_normalized"]))["record"]


def load_case(
    case_dir: Path,
    dataset: BridgeResponseDataset,
    index: int,
    mode: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config_data = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    mapping = load_mapping("accel_abs")
    sensor_rows = [int(value) for value in mapping["sensor_node_rows"]]
    _, coords_np = fixed_geometry()
    config = OperatorConfig(
        width=int(config_data["width"]),
        blocks=int(config_data["blocks"]),
        fusion=str(config_data["fusion"]),
        architecture=str(config_data["architecture"]),
    )
    model = SensorAvailabilityConditionedOperator(torch.from_numpy(coords_np).to(device), sensor_rows, config).to(device)
    checkpoint = torch.load(case_dir / "checkpoint" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    item = dataset[index]
    ground = item["ground"].unsqueeze(0).to(device)
    sensor = item["sensor"].unsqueeze(0).to(device)
    target_norm = item["target"].numpy()
    with torch.no_grad():
        if mode == "zero":
            mask = torch.zeros(1, sensor.shape[1], device=device)
            prediction_norm = model(ground, torch.zeros_like(sensor), mask)["prior"][0].cpu().numpy()
        else:
            mask = torch.ones(1, sensor.shape[1], device=device)
            prediction_norm = model(ground, sensor, mask)["posterior"][0].cpu().numpy()
    scales = target_scales("accel_abs").reshape(3, 1, 1)
    return target_norm * scales, prediction_norm * scales, config_data


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = truth.reshape(-1).astype(np.float64)
    prediction = prediction.reshape(-1).astype(np.float64)
    truth = truth - truth.mean()
    prediction = prediction - prediction.mean()
    denominator = np.linalg.norm(truth) * np.linalg.norm(prediction)
    return float(np.dot(truth, prediction) / denominator) if denominator > 1.0e-15 else 0.0


def metrics_for(truth: np.ndarray, prediction: np.ndarray, row: int, direction: int) -> dict[str, float]:
    true_curve = truth[direction, row]
    pred_curve = prediction[direction, row]
    return {
        "rmse_physical": float(np.sqrt(np.mean((pred_curve - true_curve) ** 2))),
        "correlation": correlation(true_curve, pred_curve),
        "true_peak_physical": float(np.max(np.abs(true_curve))),
        "pred_peak_physical": float(np.max(np.abs(pred_curve))),
        "peak_absolute_error_physical": float(abs(np.max(np.abs(pred_curve)) - np.max(np.abs(true_curve)))),
        "peak_relative_error_pct": float(
            abs(np.max(np.abs(pred_curve)) - np.max(np.abs(true_curve)))
            / max(float(np.max(np.abs(true_curve))), 1.0e-12)
            * 100.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--b1-case", default="K11_AMS_B1_SEED42")
    parser.add_argument("--b0-case", default="K00_SACNO_B0_SEED73")
    parser.add_argument("--record", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = BridgeResponseDataset("external", "accel_abs")
    b1_dir = run_root / "cases" / args.b1_case
    b0_dir = run_root / "cases" / args.b0_case
    record = choose_record(b1_dir, args.record)
    if record not in dataset.names:
        raise ValueError(f"Record is not in the current external split: {record}")
    index = dataset.names.index(record)
    device = torch.device(args.device)
    truth_b1, pred_b1, cfg_b1 = load_case(b1_dir, dataset, index, "real", device)
    truth_b0, pred_b0, cfg_b0 = load_case(b0_dir, dataset, index, "zero", device)
    if not np.allclose(truth_b1, truth_b0):
        raise RuntimeError("B0 and B1 use different truth curves")

    node_ids, _ = fixed_geometry()
    node_rows: dict[str, int] = {}
    for node_id in PIER_TOP_NODE_IDS:
        matches = np.flatnonzero(node_ids == node_id)
        if len(matches) != 1:
            raise ValueError(f"Node ID {node_id} is not unique")
        node_rows[str(node_id)] = int(matches[0])

    time = np.arange(truth_b1.shape[-1])
    colors = {"truth": "#1f2937", "prediction": "#d94841"}
    metrics: dict[str, object] = {
        "record": record,
        "b1_case": args.b1_case,
        "b0_case": args.b0_case,
        "node_ids": list(PIER_TOP_NODE_IDS),
        "node_rows": node_rows,
        "target": "accel_abs",
        "b1": {},
        "b0": {},
        "note": "Record selected by minimum B1 external nRMSE unless --record was provided.",
    }

    for direction_index, direction in enumerate(DIRECTIONS):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8.2), sharex=True)
        fig.suptitle(
            f"Pier-top {direction}-direction acceleration | record={record}",
            fontsize=14,
        )
        for column, node_id in enumerate(PIER_TOP_NODE_IDS):
            row = node_rows[str(node_id)]
            for plot_row, (label, truth, prediction, case_id, cfg, mode_key) in enumerate(
                (
                    ("B1 with sensors", truth_b1, pred_b1, args.b1_case, cfg_b1, "b1"),
                    ("B0 without sensors", truth_b0, pred_b0, args.b0_case, cfg_b0, "b0"),
                )
            ):
                axis = axes[plot_row, column]
                axis.plot(time, truth[direction_index, row], color=colors["truth"], linewidth=1.1, label="True")
                axis.plot(
                    time,
                    prediction[direction_index, row],
                    color=colors["prediction"],
                    linewidth=1.0,
                    linestyle="--",
                    label="Prediction",
                )
                axis.set_title(f"{label}\nnode {node_id}")
                axis.grid(True, alpha=0.25)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if column == 0:
                    axis.set_ylabel("Acceleration (m/s²)")
                    axis.text(
                        0.01,
                        0.97,
                        f"{case_id}\n{cfg['architecture']} + {cfg['loss_profile']}",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )
                if plot_row == 1:
                    axis.set_xlabel("Time step")
                if column == 0 and plot_row == 0:
                    axis.legend(loc="upper right", frameon=False, fontsize=8)
                metrics[mode_key].setdefault(str(node_id), {})[direction] = metrics_for(
                    truth, prediction, row, direction_index
                )
        fig.text(
            0.5,
            0.01,
            "Solid dark line: true response; dashed red line: reconstructed response. Same record for B0/B1.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.94))
        fig.savefig(output_dir / f"all_pier_top_curves_{direction}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for mode_key in ("b0", "b1"):
        all_values = [
            metrics[mode_key][str(node_id)][direction]["rmse_physical"]
            for node_id in PIER_TOP_NODE_IDS
            for direction in DIRECTIONS
        ]
        metrics[mode_key + "_mean_rmse_all_piers_directions"] = float(np.mean(all_values))
    (output_dir / "all_pier_top_curves.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
