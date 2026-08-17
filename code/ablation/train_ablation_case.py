from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ablation_data import BridgeResponseDataset, fixed_geometry, load_mapping, target_scales
from ablation_losses import component_losses, per_sample_rmse, profile_weights
from ablation_model import OperatorConfig, SensorAvailabilityConditionedOperator


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sensor-training", choices=("separate_zero", "separate_real", "joint"), required=True)
    parser.add_argument("--fusion", choices=("mean", "coord_attention"), required=True)
    parser.add_argument("--architecture", choices=("sacno", "ms_sacno", "ams_sacno", "spectral_sacno", "qg_sacno"), default="sacno")
    parser.add_argument("--loss-profile", choices=("basic", "physical", "peak_balanced", "phase_balanced", "physics_proxy", "direction_balanced"), required=True)
    parser.add_argument("--target", choices=("disp", "accel_abs"), default="disp")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--sensor-noise-std", type=float, default=0.0)
    parser.add_argument("--sensor-dropout-prob", type=float, default=0.0)
    parser.add_argument("--eval-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-dropout-prob", type=float, default=0.0)
    parser.add_argument("--peak-score-weight", type=float, default=0.0)
    parser.add_argument("--direction-score-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset: BridgeResponseDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0 if args.smoke else args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(not args.smoke and args.num_workers > 0),
    )


def masks(batch: int, sensors: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros(batch, sensors, device=device)
    real = torch.ones(batch, sensors, device=device)
    return zero, real


def augment_sensor(
    sensor: torch.Tensor,
    sensor_mask: torch.Tensor,
    noise_std: float,
    dropout_prob: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply only deployment-available sensor corruption; truth is never used."""
    if not training and noise_std <= 0.0 and dropout_prob <= 0.0:
        return sensor, sensor_mask
    mask = sensor_mask.clone()
    if dropout_prob > 0.0:
        keep = (torch.rand_like(mask) >= dropout_prob).to(mask.dtype) * mask
        # Keep at least one available sensor for a B1 training sample so the
        # robust branch does not silently become B0 for every sensor.
        dropped_all = (keep.sum(dim=1) == 0) & (mask.sum(dim=1) > 0)
        if dropped_all.any():
            keep[dropped_all, 0] = 1.0
        mask = keep
    corrupted = sensor * mask[:, :, None, None]
    if noise_std > 0.0:
        corrupted = corrupted + torch.randn_like(corrupted) * float(noise_std) * mask[:, :, None, None]
    return corrupted, mask


def forward_modes(
    model: SensorAvailabilityConditionedOperator,
    batch: dict[str, Any],
    training_mode: str,
    device: torch.device,
    sensor_noise_std: float = 0.0,
    sensor_dropout_prob: float = 0.0,
    training: bool = False,
) -> dict[str, torch.Tensor]:
    ground = batch["ground"].to(device, non_blocking=True)
    sensor = batch["sensor"].to(device, non_blocking=True)
    zero_mask, real_mask = masks(ground.shape[0], sensor.shape[1], device)
    if training_mode == "separate_zero":
        output = model(ground, torch.zeros_like(sensor), zero_mask)
        return {"zero": output["prior"]}
    sensor, real_mask = augment_sensor(
        sensor, real_mask, sensor_noise_std, sensor_dropout_prob, training
    )
    output = model(ground, sensor, real_mask)
    if training_mode == "separate_real":
        return {"real": output["posterior"]}
    return {"zero": output["prior"], "real": output["posterior"], "correction": output["correction"]}


def train_epoch(
    model: SensorAvailabilityConditionedOperator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    row_weights: torch.Tensor,
    unobserved: torch.Tensor,
    sensor_rows: torch.Tensor,
    sensor_to_output_ratio: torch.Tensor,
    spatial_edges: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    weights = profile_weights(args.loss_profile)
    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = forward_modes(
                model,
                batch,
                args.sensor_training,
                device,
                sensor_noise_std=args.sensor_noise_std,
                sensor_dropout_prob=args.sensor_dropout_prob,
                training=True,
            )
            mode_losses: dict[str, dict[str, torch.Tensor]] = {}
            sensor_target = batch["sensor"].to(device, non_blocking=True) * sensor_to_output_ratio.view(1, 1, 3, 1)
            for mode in ("zero", "real"):
                if mode in outputs:
                    mode_losses[mode] = component_losses(
                        outputs[mode],
                        target,
                        row_weights,
                        args.loss_profile,
                        sensor_target=sensor_target if mode == "real" else None,
                        sensor_rows=sensor_rows if mode == "real" else None,
                        spatial_edges=spatial_edges,
                    )
            loss = torch.stack([item["total"] for item in mode_losses.values()]).mean()
            ranking = target.new_zeros(())
            correction = target.new_zeros(())
            if args.sensor_training == "joint":
                zero_error = per_sample_rmse(outputs["zero"], target, unobserved)
                real_error = per_sample_rmse(outputs["real"], target, unobserved)
                ranking = torch.relu(real_error - zero_error).mean()
                correction = outputs["correction"].square().mean()
                loss = loss + weights.ranking * ranking + weights.correction * correction
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        values = {"total": float(loss.detach()), "ranking": float(ranking.detach()), "correction": float(correction.detach())}
        for mode, losses in mode_losses.items():
            for key, value in losses.items():
                values[f"{mode}_{key}"] = float(value.detach())
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def summary_stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 1.0e-15 else 0.0


@torch.no_grad()
def evaluate(
    model: SensorAvailabilityConditionedOperator,
    loader: DataLoader,
    modes_to_evaluate: list[str],
    output_scales: np.ndarray,
    unobserved_rows: np.ndarray,
    pier_top_rows: np.ndarray,
    pier_bottom_rows: np.ndarray,
    device: torch.device,
    sensor_noise_std: float = 0.0,
    sensor_dropout_prob: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        ground = batch["ground"].to(device, non_blocking=True)
        sensor = batch["sensor"].to(device, non_blocking=True)
        zero_mask, real_mask = masks(ground.shape[0], sensor.shape[1], device)
        zero_output = model(ground, torch.zeros_like(sensor), zero_mask)["prior"]
        if "real" in modes_to_evaluate:
            eval_sensor, eval_mask = augment_sensor(
                sensor,
                real_mask,
                sensor_noise_std,
                sensor_dropout_prob,
                training=False,
            )
            real_output = model(ground, eval_sensor, eval_mask)["posterior"]
        else:
            real_output = None
        target = batch["target"].cpu().numpy()
        predictions = {"zero": zero_output.cpu().numpy()}
        if real_output is not None:
            predictions["real"] = real_output.cpu().numpy()
        for item_index, name in enumerate(batch["name"]):
            truth_norm = target[item_index]
            truth = truth_norm * output_scales.reshape(3, 1, 1)
            for mode in modes_to_evaluate:
                pred_norm = predictions[mode][item_index]
                pred = pred_norm * output_scales.reshape(3, 1, 1)
                error_norm = pred_norm[:, unobserved_rows] - truth_norm[:, unobserved_rows]
                error = pred[:, unobserved_rows] - truth[:, unobserved_rows]
                true_unobserved = truth[:, unobserved_rows]
                pred_unobserved = pred[:, unobserved_rows]
                true_peak = float(np.max(np.abs(true_unobserved)))
                pred_peak = float(np.max(np.abs(pred_unobserved)))
                row: dict[str, Any] = {
                    "record": name,
                    "mode": mode,
                    "nrmse_normalized": float(np.sqrt(np.mean(error_norm * error_norm))),
                    "rmse_physical": float(np.sqrt(np.mean(error * error))),
                    "mae_physical": float(np.mean(np.abs(error))),
                    "correlation": correlation(pred_unobserved, true_unobserved),
                    "true_peak_physical": true_peak,
                    "pred_peak_physical": pred_peak,
                    "peak_absolute_error_physical": abs(pred_peak - true_peak),
                    "peak_relative_error_pct": abs(pred_peak - true_peak) / max(true_peak, 1.0e-12) * 100.0,
                    "pier_top_rmse_physical": float(np.sqrt(np.mean((pred[:, pier_top_rows] - truth[:, pier_top_rows]) ** 2))),
                    "pier_bottom_rmse_physical": float(np.sqrt(np.mean((pred[:, pier_bottom_rows] - truth[:, pier_bottom_rows]) ** 2))),
                }
                for direction, label in enumerate(("X", "Y", "Z")):
                    directional = error[direction]
                    directional_norm = error_norm[direction]
                    row[f"{label}_rmse_physical"] = float(np.sqrt(np.mean(directional * directional)))
                    row[f"{label}_nrmse_normalized"] = float(np.sqrt(np.mean(directional_norm * directional_norm)))
                    row[f"{label}_correlation"] = correlation(pred_unobserved[direction], true_unobserved[direction])
                row["max_direction_nrmse_normalized"] = max(
                    row["X_nrmse_normalized"], row["Y_nrmse_normalized"], row["Z_nrmse_normalized"]
                )
                row["max_direction_rmse_physical"] = max(
                    row["X_rmse_physical"], row["Y_rmse_physical"], row["Z_rmse_physical"]
                )
                rows.append(row)

    summary: dict[str, Any] = {}
    for mode in modes_to_evaluate:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            key: summary_stat([float(row[key]) for row in selected])
            for key in (
                "nrmse_normalized",
                "rmse_physical",
                "mae_physical",
                "correlation",
                "peak_absolute_error_physical",
                "peak_relative_error_pct",
                "pier_top_rmse_physical",
                "pier_bottom_rmse_physical",
                "X_rmse_physical",
                "Y_rmse_physical",
                "Z_rmse_physical",
                "X_nrmse_normalized",
                "Y_nrmse_normalized",
                "Z_nrmse_normalized",
                "max_direction_nrmse_normalized",
                "max_direction_rmse_physical",
            )
        }
    if set(modes_to_evaluate) == {"zero", "real"}:
        zero = {row["record"]: row for row in rows if row["mode"] == "zero"}
        real = {row["record"]: row for row in rows if row["mode"] == "real"}
        common = sorted(set(zero) & set(real))
        improvements = [
            (zero[name]["rmse_physical"] - real[name]["rmse_physical"]) / max(zero[name]["rmse_physical"], 1.0e-12) * 100.0
            for name in common
        ]
        summary["sensor_information_gain"] = {
            "record_count": len(common),
            "rmse_improvement_pct": summary_stat(improvements),
            "b1_win_rate_pct": float(np.mean(np.asarray(improvements) > 0.0) * 100.0),
        }
    return summary, rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def selection_score(
    summary: dict[str, Any],
    modes: list[str],
    peak_weight: float,
    direction_weight: float,
) -> float:
    """Full-field error plus peak and worst-direction selection terms."""
    scores = []
    for mode in modes:
        nrmse = summary[mode]["nrmse_normalized"]["mean"]
        peak_fraction = summary[mode]["peak_relative_error_pct"]["median"] / 100.0
        worst_direction = summary[mode]["max_direction_nrmse_normalized"]["mean"]
        scores.append(float(nrmse + peak_weight * peak_fraction + direction_weight * worst_direction))
    return float(np.mean(scores))


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            item: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "epoch":
                    item[key] = int(float(value))
                else:
                    item[key] = float(value)
            rows.append(item)
    return rows


def main() -> int:
    args = parse_args()
    case_dir = args.run_root.resolve() / "cases" / args.case_id
    status_path = case_dir / "status.json"
    resume_existing = False
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            print(f"SKIP COMPLETE {args.case_id}")
            return 0
        if status.get("status") == "started":
            resume_existing = True
        else:
            raise RuntimeError(f"Failed case requires launcher --retry-failed: {case_dir}")
    if not resume_existing:
        case_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    atomic_json(
        status_path,
        {"status": "started", "case_id": args.case_id, "started_unix": started, "resumed": resume_existing},
    )
    requested_config = vars(args) | {"run_root": str(args.run_root.resolve())}
    config_path = case_dir / "config.json"
    if resume_existing:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "case_id", "sensor_training", "fusion", "architecture", "loss_profile", "target", "seed",
            "batch_size", "learning_rate", "weight_decay", "width", "blocks", "gradient_clip", "smoke",
            "sensor_noise_std", "sensor_dropout_prob", "eval_noise_std", "eval_dropout_prob", "peak_score_weight",
            "direction_score_weight",
        )
        mismatch = {
            key: [existing_config.get(key), requested_config.get(key)]
            for key in compatibility_keys
            if existing_config.get(key) != requested_config.get(key)
        }
        if mismatch:
            raise RuntimeError(f"Resume configuration mismatch: {mismatch}")
    else:
        atomic_json(config_path, requested_config)
    try:
        set_seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        limit_train = 8 if args.smoke else None
        limit_eval = 4 if args.smoke else None
        epochs = 1 if args.smoke else args.epochs
        train_dataset = BridgeResponseDataset("train", args.target, limit_train)
        tune_dataset = BridgeResponseDataset("tune", args.target, limit_eval)
        external_dataset = BridgeResponseDataset("external", args.target, limit_eval)
        formal_dataset = BridgeResponseDataset("formal", args.target, limit_eval)
        train_loader = make_loader(train_dataset, args, True)
        tune_loader = make_loader(tune_dataset, args, False)
        external_loader = make_loader(external_dataset, args, False)
        formal_loader = make_loader(formal_dataset, args, False)

        mapping = load_mapping(args.target)
        sensor_rows = [int(value) for value in mapping["sensor_node_rows"]]
        scales = target_scales(args.target)
        sensor_scales = np.asarray(mapping["sensor_acceleration_scales"], dtype=np.float32)
        sensor_to_output_ratio = torch.as_tensor(sensor_scales / scales, device=device)
        node_ids, coords_np = fixed_geometry()
        coords = torch.from_numpy(coords_np).to(device)
        config = OperatorConfig(
            width=args.width,
            blocks=args.blocks,
            fusion=args.fusion,
            architecture=args.architecture,
        )
        model = SensorAvailabilityConditionedOperator(coords, sensor_rows, config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        spatial_edges = torch.nonzero(model.adjacency > 0.0, as_tuple=False).T
        spatial_edges = spatial_edges[:, spatial_edges[0] != spatial_edges[1]]
        sensor_rows_tensor = torch.as_tensor(sensor_rows, dtype=torch.long, device=device)
        row_weights = torch.ones(153, device=device)
        row_weights[sensor_rows] = 0.2
        unobserved_np = np.asarray([row for row in range(153) if row not in set(sensor_rows)], dtype=np.int64)
        unobserved = torch.as_tensor(unobserved_np, device=device)
        row_by_id = {int(node): row for row, node in enumerate(node_ids)}
        pier_top = np.asarray([row_by_id[node] for node in (143, 144, 145, 146)], dtype=np.int64)
        pier_bottom = np.asarray([row_by_id[node] for node in (107, 108, 109, 110)], dtype=np.int64)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        modes_to_evaluate = ["zero", "real"] if args.sensor_training == "joint" else ["zero" if args.sensor_training.endswith("zero") else "real"]
        history: list[dict[str, Any]] = read_history(case_dir / "iteration_history.csv")
        best_score = float("inf")
        best_epoch = 0
        no_improvement = 0

        start_epoch = 1
        latest_path = case_dir / "checkpoint" / "latest.pt"
        if resume_existing and latest_path.is_file():
            latest = torch.load(latest_path, map_location=device, weights_only=False)
            if latest["strategy_signature"] != model.strategy_signature():
                raise RuntimeError("Resume checkpoint strategy signature mismatch")
            model.load_state_dict(latest["model"], strict=True)
            optimizer.load_state_dict(latest["optimizer"])
            start_epoch = int(latest["epoch"]) + 1
            if history:
                best_item = min(history, key=lambda item: float(item["selection_score"]))
                best_score = float(best_item["selection_score"])
                best_epoch = int(best_item["epoch"])
                no_improvement = max(0, int(history[-1]["epoch"]) - best_epoch)
            print(f"RESUME {args.case_id} FROM EPOCH {start_epoch}", flush=True)

        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.time()
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                args,
                row_weights,
                unobserved,
                sensor_rows_tensor,
                sensor_to_output_ratio,
                spatial_edges,
                device,
            )
            tune_summary, _ = evaluate(
                model, tune_loader, modes_to_evaluate, scales, unobserved_np, pier_top, pier_bottom, device
            )
            score = selection_score(
                tune_summary,
                modes_to_evaluate,
                args.peak_score_weight,
                args.direction_score_weight,
            )
            item = {
                "epoch": epoch,
                "seconds": time.time() - epoch_started,
                "selection_score": score,
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
            for mode in modes_to_evaluate:
                item[f"tune_{mode}_nrmse"] = tune_summary[mode]["nrmse_normalized"]["mean"]
                item[f"tune_{mode}_rmse_physical"] = tune_summary[mode]["rmse_physical"]["mean"]
                item[f"tune_{mode}_peak_abs_physical"] = tune_summary[mode]["peak_absolute_error_physical"]["mean"]
                item[f"tune_{mode}_peak_are_pct"] = tune_summary[mode]["peak_relative_error_pct"]["median"]
            history.append(item)
            write_rows(case_dir / "iteration_history.csv", history)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            checkpoint = {
                "case_id": args.case_id,
                "epoch": epoch,
                "strategy_signature": model.strategy_signature(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "selection_score": score,
                "args": vars(args) | {"run_root": str(args.run_root.resolve())},
                "parameter_count": parameter_count,
            }
            atomic_checkpoint(case_dir / "checkpoint" / "latest.pt", checkpoint)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                no_improvement = 0
                atomic_checkpoint(case_dir / "checkpoint" / "best.pt", checkpoint)
            else:
                no_improvement += 1
            if not args.smoke and no_improvement >= args.patience:
                break

        best_path = case_dir / "checkpoint" / "best.pt"
        best = torch.load(best_path, map_location=device, weights_only=False)
        if best["strategy_signature"] != model.strategy_signature():
            raise RuntimeError("Checkpoint strategy signature mismatch")
        model.load_state_dict(best["model"], strict=True)
        tune_summary, tune_rows = evaluate(
            model, tune_loader, modes_to_evaluate, scales, unobserved_np, pier_top, pier_bottom, device
        )
        external_summary, external_rows = evaluate(
            model, external_loader, modes_to_evaluate, scales, unobserved_np, pier_top, pier_bottom, device
        )
        formal_summary, formal_rows = evaluate(
            model, formal_loader, modes_to_evaluate, scales, unobserved_np, pier_top, pier_bottom, device
        )
        robust_summary = None
        robust_rows = []
        if args.eval_noise_std > 0.0 or args.eval_dropout_prob > 0.0:
            set_seed(args.seed + 1000003)
            robust_summary, robust_rows = evaluate(
                model,
                external_loader,
                modes_to_evaluate,
                scales,
                unobserved_np,
                pier_top,
                pier_bottom,
                device,
                sensor_noise_std=args.eval_noise_std,
                sensor_dropout_prob=args.eval_dropout_prob,
            )
        write_rows(case_dir / "tune_per_record.csv", tune_rows)
        write_rows(case_dir / "external_per_record.csv", external_rows)
        write_rows(case_dir / "formal_per_record.csv", formal_rows)
        if robust_rows:
            write_rows(case_dir / "external_robust_per_record.csv", robust_rows)
        summary = {
            "status": "completed",
            "case_id": args.case_id,
            "target": args.target,
            "sensor_training": args.sensor_training,
            "fusion": args.fusion,
            "architecture": args.architecture,
            "loss_profile": args.loss_profile,
            "seed": args.seed,
            "smoke": args.smoke,
            "sensor_noise_std": args.sensor_noise_std,
            "sensor_dropout_prob": args.sensor_dropout_prob,
            "eval_noise_std": args.eval_noise_std,
            "eval_dropout_prob": args.eval_dropout_prob,
            "peak_score_weight": args.peak_score_weight,
            "direction_score_weight": args.direction_score_weight,
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "elapsed_seconds": time.time() - started,
            "tune": tune_summary,
            "external": external_summary,
            "formal": formal_summary,
            "external_robust": robust_summary,
            "external_scope_warning": "The 157-record external split is historical validation; formal metrics are from the untouched 78-record test split.",
            "formal_scope": "Untouched 78-record formal test; never used for training, checkpoint selection or strategy selection.",
            "checkpoint": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
        }
        atomic_json(case_dir / "summary.json", summary)
        atomic_json(status_path, {"status": "completed", "case_id": args.case_id, "finished_unix": time.time()})
        print(f"CASE COMPLETE {args.case_id}")
        return 0
    except Exception as error:
        trace = traceback.format_exc()
        (case_dir / "error.log").write_text(trace, encoding="utf-8")
        atomic_json(
            status_path,
            {"status": "failed", "case_id": args.case_id, "error_type": type(error).__name__, "error": str(error)},
        )
        print(trace, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
