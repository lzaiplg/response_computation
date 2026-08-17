from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ablation_data import BridgeResponseDataset, TEST_LIST, dataset_manifest, deterministic_development_split, fixed_geometry, load_mapping, read_names, target_scales, VAL_LIST
from ablation_losses import component_losses
from ablation_model import OperatorConfig, SensorAvailabilityConditionedOperator


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, tune = deterministic_development_split()
    external = read_names(VAL_LIST)
    formal = read_names(TEST_LIST)
    partitions = (train, tune, external, formal)
    if any(set(left) & set(right) for index, left in enumerate(partitions) for right in partitions[index + 1:]):
        raise RuntimeError("Split overlap detected")
    dataset = BridgeResponseDataset("train", "accel_abs", limit=1)
    sample = dataset[0]
    ground = sample["ground"].unsqueeze(0).to(device)
    sensor = sample["sensor"].unsqueeze(0).to(device)
    zero_mask = torch.zeros(1, sensor.shape[1], device=device)
    real_mask = torch.ones_like(zero_mask)
    _, coords_np = fixed_geometry()
    coords = torch.from_numpy(coords_np).to(device)
    reports = []
    parameter_counts = {}
    for architecture, fusion in (
        ("sacno", "mean"),
        ("sacno", "coord_attention"),
        ("ms_sacno", "coord_attention"),
        ("ams_sacno", "coord_attention"),
        ("spectral_sacno", "coord_attention"),
        ("qg_sacno", "coord_attention"),
    ):
        config = OperatorConfig(width=16, blocks=1, fusion=fusion, architecture=architecture)
        model = SensorAvailabilityConditionedOperator(coords, dataset.sensor_rows.tolist(), config).to(device)
        model.eval()
        with torch.no_grad():
            zero = model(ground, torch.zeros_like(sensor), zero_mask)
            real = model(ground, sensor, real_mask)
        expected = (1, 3, 153, 1000)
        for key in ("prior", "correction", "posterior"):
            if tuple(zero[key].shape) != expected or tuple(real[key].shape) != expected:
                raise RuntimeError(f"{fusion}/{key}: invalid shape")
        zero_correction = float(zero["correction"].abs().max().cpu())
        zero_identity = float((zero["posterior"] - zero["prior"]).abs().max().cpu())
        if zero_correction != 0.0 or zero_identity != 0.0:
            raise RuntimeError("Zero-sensor limit is not exact")
        weights = torch.ones(153, device=device)
        sensor_rows = torch.as_tensor(dataset.sensor_rows, dtype=torch.long, device=device)
        spatial_edges = torch.nonzero(model.adjacency > 0.0, as_tuple=False).T
        spatial_edges = spatial_edges[:, spatial_edges[0] != spatial_edges[1]]
        mapping = load_mapping("accel_abs")
        sensor_ratio = torch.as_tensor(
            torch.as_tensor(mapping["sensor_acceleration_scales"], device=device)
            / torch.as_tensor(target_scales("accel_abs"), device=device)
        )
        sensor_target = sample["sensor"].unsqueeze(0).to(device) * sensor_ratio.view(1, 1, 3, 1)
        losses = {}
        for profile in ("physical", "peak_balanced", "phase_balanced", "physics_proxy", "direction_balanced"):
            loss = component_losses(
                real["posterior"],
                sample["target"].unsqueeze(0).to(device),
                weights,
                profile,
                sensor_target=sensor_target,
                sensor_rows=sensor_rows,
                spatial_edges=spatial_edges,
            )
            if not all(torch.isfinite(value) for value in loss.values()):
                raise RuntimeError(f"Non-finite preflight loss: {profile}")
            losses[profile] = {key: float(value.cpu()) for key, value in loss.items()}
        count = sum(parameter.numel() for parameter in model.parameters())
        parameter_counts[f"{architecture}/{fusion}"] = count
        reports.append(
            {
                "architecture": architecture,
                "fusion": fusion,
                "parameter_count": count,
                "output_shape": list(expected),
                "zero_correction_max_abs": zero_correction,
                "zero_prior_posterior_max_abs_difference": zero_identity,
                "losses": losses,
            }
        )
    report = {
        "status": "PASS",
        "python": sys.executable,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": dataset_manifest(),
        "sample_shapes": {key: list(value.shape) for key, value in sample.items() if torch.is_tensor(value)},
        "models": reports,
        "fairness_check": "B0 and B1 are two mask states of the same instantiated model and therefore have identical parameters.",
        "formal_test_available": True,
        "formal_test_count": len(formal),
    }
    output = SCRIPT_DIR / "preflight_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PREFLIGHT PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
