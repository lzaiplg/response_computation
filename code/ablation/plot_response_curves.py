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


def read_external_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_record(case_dir: Path, requested: str | None) -> str:
    rows = read_external_rows(case_dir / "external_per_record.csv")
    if requested:
        if not any(row["record"] == requested for row in rows):
            raise ValueError(f"Record not found in {case_dir.name}: {requested}")
        return requested
    # Show the best external example for the selected B1 model. The report
    # states this explicitly so the figure is not mistaken for a blind-test
    # average curve.
    return min(rows, key=lambda row: float(row["nrmse_normalized"]))["record"]


def case_config(case_dir: Path) -> dict[str, object]:
    return json.loads((case_dir / "config.json").read_text(encoding="utf-8"))


def load_prediction(
    case_dir: Path,
    dataset: BridgeResponseDataset,
    index: int,
    node_id: int,
    mode: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object]]:
    config_data = case_config(case_dir)
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
    target = item["target"].numpy()
    with torch.no_grad():
        if mode == "zero":
            mask = torch.zeros(1, sensor.shape[1], device=device)
            output = model(ground, torch.zeros_like(sensor), mask)["prior"]
        else:
            mask = torch.ones(1, sensor.shape[1], device=device)
            output = model(ground, sensor, mask)["posterior"]
    prediction = output[0].cpu().numpy()
    scales = target_scales("accel_abs").reshape(3, 1, 1)
    truth_physical = target * scales
    prediction_physical = prediction * scales

    node_ids, _ = fixed_geometry()
    matches = np.flatnonzero(node_ids == int(node_id))
    if len(matches) != 1:
        raise ValueError(f"Node ID {node_id} is not unique in geometry")
    row = int(matches[0])
    return truth_physical[:, row, :], prediction_physical[:, row, :], row, config_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot true/predicted acceleration curves from a completed ablation case.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--b1-case", default="K11_AMS_B1_SEED42")
    parser.add_argument("--b0-case", default="K00_SACNO_B0_SEED73")
    parser.add_argument("--record", default=None)
    parser.add_argument("--node-id", type=int, default=143)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    dataset = BridgeResponseDataset("external", "accel_abs")
    b1_dir = run_root / "cases" / args.b1_case
    b0_dir = run_root / "cases" / args.b0_case
    record = find_record(b1_dir, args.record)
    if record not in dataset.names:
        raise ValueError(f"Record is not in the current external split: {record}")
    index = dataset.names.index(record)
    device = torch.device(args.device)

    truth_b1, pred_b1, node_row, cfg_b1 = load_prediction(b1_dir, dataset, index, args.node_id, "real", device)
    truth_b0, pred_b0, _, cfg_b0 = load_prediction(b0_dir, dataset, index, args.node_id, "zero", device)
    if not np.allclose(truth_b1, truth_b0):
        raise RuntimeError("B0 and B1 truth curves are not identical")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(truth_b1.shape[-1])
    directions = ("X", "Y", "Z")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
    fig.suptitle(
        "Bridge acceleration response reconstruction | "
        f"record={record} | node={args.node_id} (row {node_row})",
        fontsize=14,
    )
    colors = {"truth": "#1f2937", "prediction": "#d94841"}
    panels = ((axes[0], truth_b1, pred_b1, "B1 with sensors", args.b1_case, cfg_b1),
              (axes[1], truth_b0, pred_b0, "B0 without sensors", args.b0_case, cfg_b0))
    for row_axes, truth, prediction, label, case_id, cfg in panels:
        for direction, axis in enumerate(row_axes):
            axis.plot(time, truth[direction], color=colors["truth"], linewidth=1.15, label="True")
            axis.plot(time, prediction[direction], color=colors["prediction"], linewidth=1.0, linestyle="--", label="Prediction")
            axis.set_title(f"{label} | {directions[direction]} direction")
            axis.set_ylabel("Acceleration (m/s²)")
            axis.grid(True, alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if direction == 0:
                axis.text(
                    0.01,
                    0.97,
                    f"{case_id}\narch={cfg['architecture']}, loss={cfg['loss_profile']}",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            if row_axes is axes[1]:
                axis.set_xlabel("Time step")
            if direction == 0:
                axis.legend(loc="upper right", frameon=False)
    fig.text(
        0.5,
        0.01,
        "Solid dark line: true response; dashed red line: reconstructed response. "
        "The same external record and pier-top node are used for B0/B1.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "record": record,
        "node_id": args.node_id,
        "node_row": node_row,
        "b1_case": args.b1_case,
        "b0_case": args.b0_case,
        "b1_case_config": cfg_b1,
        "b0_case_config": cfg_b0,
        "target": "accel_abs",
        "note": "Record selected by minimum B1 external nRMSE unless --record was provided.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
