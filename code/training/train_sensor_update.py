# -*- coding: utf-8 -*-
"""Train 7-channel pix2pixHD sensor-update models on the fixed 7:2:1 split.

Experiments:
    B1-S5-U: --sensor-mode real
    B0-7Z  : --sensor-mode zero  (strict 7-channel control)

Input = ground RGB(3) + sparse sensor RGB(3) + mask(1).
Target is the 153-row absolute-acceleration RGB image.
Best checkpoint is selected by validation L1 on the 148 unobserved nodes.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from pix2pixHD_sensor_update_model import (
    Config as ModelConfig,
    GANLoss,
    corrected_architecture_report,
    feature_matching_loss,
    make_models,
    save_checkpoint,
    set_requires_grad,
    tensor_to_image,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = PACKAGE_ROOT / "data" / "processed" / "sensor_update_acceleration"
DATASET_ROOT = FORMAL_ROOT / "dataset_rgb"
MAPPING_PATH = FORMAL_ROOT / "sensor_update_acc_mapping.json"
DEFAULT_A1 = (
    PACKAGE_ROOT / "models" / "A1_acceleration_best.pt"
)


@dataclass
class TrainProfile:
    learning_rate_g: float = 1.0e-4
    learning_rate_d: float = 2.5e-5
    lambda_l1: float = 50.0
    lambda_g1_l1: float = 10.0
    lambda_fm_stage1: float = 1.0
    lambda_fm_stage2: float = 10.0
    warmup_epochs: int = 5
    discriminator_update_interval: int = 2
    real_label: float = 0.9
    gradient_clip: float = 5.0
    sensor_row_weight: float = 0.2


class SensorUpdateDataset(Dataset):
    def __init__(self, split: str, sensor_mode: str, limit: int | None = None) -> None:
        self.split = split
        self.sensor_mode = sensor_mode
        root = DATASET_ROOT / split
        folders = {name: root / name for name in ("ground", "sensor", "mask", "target")}
        file_sets = {name: {p.name: p for p in folder.glob("*.png")} for name, folder in folders.items()}
        names = sorted(set.intersection(*(set(v) for v in file_sets.values())))
        if limit is not None:
            names = names[: max(0, limit)]
        if not names:
            raise FileNotFoundError(f"No S5-U files found under {root}")
        self.samples = [({k: file_sets[k][name] for k in file_sets}, name) for name in names]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def rgb_to_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
        if array.shape != (153, 1000, 3):
            raise ValueError(f"{path.name}: RGB shape={array.shape}")
        array = array / 127.5 - 1.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    @staticmethod
    def mask_to_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.shape != (153, 1000):
            raise ValueError(f"{path.name}: mask shape={array.shape}")
        return torch.from_numpy((array / 255.0)[None, ...].copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        paths, name = self.samples[index]
        ground = self.rgb_to_tensor(paths["ground"])
        sensor = self.rgb_to_tensor(paths["sensor"])
        mask = self.mask_to_tensor(paths["mask"])
        target = self.rgb_to_tensor(paths["target"])

        # Remove neutral-gray PNG quantization from unsensed rows exactly.
        sensor = sensor * mask
        if self.sensor_mode == "zero":
            sensor = torch.zeros_like(sensor)
            mask = torch.zeros_like(mask)
        condition = torch.cat([ground, sensor, mask], dim=0)
        return {
            "input": condition,
            "ground": ground,
            "sensor": sensor,
            "mask": mask,
            "target": target,
            "name": name,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("summary", "smoke", "train"), default="smoke")
    p.add_argument("--sensor-mode", choices=("real", "zero"), default="real")
    p.add_argument("--run-name", default="B1_MAIN10_V2_ACC_S5_U")
    p.add_argument("--output-root", type=Path, default=None, help="Directory for the run; defaults to package/results/<run-name>.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage2-epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--min-improvement", type=float, default=1.0e-5)
    p.add_argument("--init-a1", type=Path, default=DEFAULT_A1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--overwrite-run", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--limit-train-samples", type=int, default=None)
    p.add_argument("--limit-val-samples", type=int, default=None)
    p.add_argument("--smoke-train-samples", type=int, default=4)
    p.add_argument("--smoke-val-samples", type=int, default=2)
    p.add_argument("--sensor-row-weight", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_logging(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(path, mode="a" if append else "w", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(); sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh); logger.addHandler(sh)


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split: str, sensor_mode: str, batch_size: int, shuffle: bool, workers: int, limit: int | None) -> DataLoader:
    ds = SensorUpdateDataset(split, sensor_mode, limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def load_masks(device: torch.device, profile: TrainProfile) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    rows = [int(x) for x in mapping["sensor_node_rows"]]
    ids = [int(x) for x in mapping["sensor_node_ids"]]
    unobserved = torch.ones(153, dtype=torch.float32, device=device)
    unobserved[rows] = 0.0
    weights = torch.ones(153, dtype=torch.float32, device=device)
    weights[rows] = float(profile.sensor_row_weight)
    return weights, unobserved, rows, ids


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, row_weights: torch.Tensor) -> torch.Tensor:
    w = row_weights.view(1, 1, -1, 1)
    numerator = (torch.abs(pred - target) * w).sum()
    denominator = w.sum() * pred.shape[0] * pred.shape[1] * pred.shape[3]
    return numerator / denominator.clamp_min(1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    return weighted_l1(pred, target, row_mask)


def pairwise_diversity(tensors: list[torch.Tensor]) -> float:
    if len(tensors) < 2:
        return 0.0
    return float(np.mean([F.l1_loss(a, b).item() for a, b in combinations(tensors, 2)]))


@torch.no_grad()
def validate(generator, loader: DataLoader, device: torch.device, row_weights: torch.Tensor, unobserved: torch.Tensor, max_batches: int | None) -> dict[str, float]:
    generator.eval()
    sums = {"all_l1": 0.0, "weighted_l1": 0.0, "unobserved_l1": 0.0, "unobserved_mse": 0.0}
    channel = np.zeros(3, dtype=np.float64)
    count = 0
    preds: list[torch.Tensor] = []; truths: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        generated, _ = generator(inputs)
        sums["all_l1"] += F.l1_loss(generated, targets).item()
        sums["weighted_l1"] += weighted_l1(generated, targets, row_weights).item()
        sums["unobserved_l1"] += masked_l1(generated, targets, unobserved).item()
        mask = unobserved.view(1, 1, 153, 1)
        sq = ((generated - targets) ** 2 * mask).sum() / (mask.sum() * generated.shape[0] * generated.shape[1] * generated.shape[3])
        sums["unobserved_mse"] += sq.item()
        for d in range(3):
            channel[d] += masked_l1(generated[:, d:d+1], targets[:, d:d+1], unobserved).item()
        count += 1
        if len(preds) < 4:
            for i in range(generated.shape[0]):
                if len(preds) >= 4: break
                preds.append(generated[i].detach().cpu()); truths.append(targets[i].detach().cpu())
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if count == 0: raise RuntimeError("Empty validation loader")
    result = {k: v / count for k, v in sums.items()}
    result.update({"unobserved_l1_r": float(channel[0]/count), "unobserved_l1_g": float(channel[1]/count), "unobserved_l1_b": float(channel[2]/count)})
    pd = pairwise_diversity(preds); td = pairwise_diversity(truths)
    result.update({"prediction_diversity": pd, "target_diversity": td, "diversity_ratio": pd/max(td,1e-12)})
    return result


def mask_to_image(mask: torch.Tensor) -> Image.Image:
    array = np.rint(mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


@torch.no_grad()
def save_previews(generator, loader: DataLoader, device: torch.device, directory: Path, prefix: str, max_samples: int = 3) -> None:
    generator.eval(); directory.mkdir(parents=True, exist_ok=True); saved = 0
    for batch in loader:
        inputs = batch["input"].to(device); targets = batch["target"].to(device)
        generated, _ = generator(inputs)
        for i in range(inputs.shape[0]):
            stem = Path(batch["name"][i]).stem
            tensor_to_image(inputs[i, 0:3]).save(directory / f"{prefix}_{stem}_ground.png")
            tensor_to_image(inputs[i, 3:6]).save(directory / f"{prefix}_{stem}_sensor.png")
            mask_to_image(inputs[i, 6:7]).save(directory / f"{prefix}_{stem}_mask.png")
            tensor_to_image(generated[i]).save(directory / f"{prefix}_{stem}_prediction.png")
            tensor_to_image(targets[i]).save(directory / f"{prefix}_{stem}_target.png")
            saved += 1
            if saved >= max_samples: return


def build_models(device: torch.device, profile: TrainProfile):
    config = ModelConfig(input_nc=7, output_nc=3)
    g, d = make_models(config, device)
    og = optim.Adam(g.parameters(), lr=profile.learning_rate_g, betas=(0.5, 0.999))
    od = optim.Adam(d.parameters(), lr=profile.learning_rate_d, betas=(0.5, 0.999))
    gan = GANLoss(real_label=profile.real_label).to(device)
    return config, g, d, og, od, gan


def transfer_a1_checkpoint(path: Path, generator, discriminator, device: torch.device) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    old_g = checkpoint["generator"]; old_d = checkpoint["discriminator"]
    new_g = generator.state_dict(); new_d = discriminator.state_dict()
    copied_g = 0; expanded_g = 0; copied_d = 0; expanded_d = 0

    for key, value in old_g.items():
        if key not in new_g: continue
        if new_g[key].shape == value.shape:
            new_g[key] = value; copied_g += 1
        elif key in ("g1.input_layer.1.weight", "g2.local_conv.1.weight") and value.ndim == 4 and value.shape[1] == 3 and new_g[key].shape[1] == 7:
            expanded = torch.zeros_like(new_g[key]); expanded[:, :3] = value
            new_g[key] = expanded; expanded_g += 1
    generator.load_state_dict(new_g, strict=True)

    for key, value in old_d.items():
        if key not in new_d: continue
        if new_d[key].shape == value.shape:
            new_d[key] = value; copied_d += 1
        elif key.endswith("layers.0.0.weight") and value.ndim == 4 and value.shape[1] == 6 and new_d[key].shape[1] == 10:
            expanded = torch.zeros_like(new_d[key])
            expanded[:, :3] = value[:, :3]       # old ground condition
            expanded[:, 7:10] = value[:, 3:6]    # old response channels
            new_d[key] = expanded; expanded_d += 1
    discriminator.load_state_dict(new_d, strict=True)
    return {"checkpoint_epoch": int(checkpoint.get("epoch", -1)), "generator_copied": copied_g, "generator_expanded": expanded_g, "discriminator_copied": copied_d, "discriminator_expanded": expanded_d}


def train_step(*, generator, discriminator, optimizer_g, optimizer_d, gan_loss, inputs, targets, profile, lambda_fm, adversarial_enabled, update_d, row_weights):
    d_value = 0.0
    if adversarial_enabled and update_d:
        set_requires_grad(discriminator, True); optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad(): fake_d, _ = generator(inputs)
        pred_real = discriminator(inputs, targets); pred_fake = discriminator(inputs, fake_d)
        loss_d = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))
        loss_d.backward(); torch.nn.utils.clip_grad_norm_(discriminator.parameters(), profile.gradient_clip); optimizer_d.step()
        d_value = float(loss_d.item())

    set_requires_grad(discriminator, False); optimizer_g.zero_grad(set_to_none=True)
    generated, g1_low = generator(inputs)
    low_target = F.interpolate(targets, size=g1_low.shape[-2:], mode="bilinear", align_corners=False)
    low_weights = F.interpolate(row_weights.view(1,1,153,1), size=(g1_low.shape[-2], 1), mode="nearest").view(-1)
    loss_l1 = weighted_l1(generated, targets, row_weights)
    loss_g1 = weighted_l1(g1_low, low_target, low_weights)
    if adversarial_enabled:
        fake_features = discriminator(inputs, generated)
        with torch.no_grad(): real_features = discriminator(inputs, targets)
        loss_gan = gan_loss(fake_features, True); loss_fm = feature_matching_loss(fake_features, real_features)
    else:
        loss_gan = torch.zeros((), device=inputs.device); loss_fm = torch.zeros((), device=inputs.device)
    loss_g = profile.lambda_l1*loss_l1 + profile.lambda_g1_l1*loss_g1 + loss_gan + lambda_fm*loss_fm
    loss_g.backward(); torch.nn.utils.clip_grad_norm_(generator.parameters(), profile.gradient_clip); optimizer_g.step(); set_requires_grad(discriminator, True)
    out = {"g_total": float(loss_g.item()), "d_total": d_value, "g_gan": float(loss_gan.item()), "weighted_l1": float(loss_l1.item()), "g1_l1": float(loss_g1.item()), "feature_matching": float(loss_fm.item())}
    if not all(math.isfinite(v) for v in out.values()): raise FloatingPointError(out)
    return out


def resolve_resume(value: str | None, result_root: Path) -> Path | None:
    if value is None: return None
    if value.lower() in ("latest", "best"): return result_root / "checkpoints" / f"{value.lower()}.pt"
    return Path(value).resolve()


def architecture_summary(device: torch.device, profile: TrainProfile) -> None:
    _, g, d, *_ = build_models(device, profile)
    for line in corrected_architecture_report(g, d): logging.info(line)
    dummy = torch.zeros(1, 7, 153, 1000, device=device)
    with torch.no_grad(): output, low = g(dummy); ds = d(dummy, output)
    logging.info("Input shape: %s", tuple(dummy.shape)); logging.info("Output shape: %s", tuple(output.shape)); logging.info("G1 low: %s", tuple(low.shape)); logging.info("D outputs: %s", [tuple(x[-1].shape) for x in ds])


def run_training(args: argparse.Namespace, result_root: Path, device: torch.device, profile: TrainProfile, smoke: bool) -> None:
    train_limit = args.smoke_train_samples if smoke else args.limit_train_samples
    val_limit = args.smoke_val_samples if smoke else args.limit_val_samples
    train_loader = make_loader("train", args.sensor_mode, args.batch_size, True, args.num_workers, train_limit)
    val_loader = make_loader("val", args.sensor_mode, args.batch_size, False, args.num_workers, val_limit)
    config, g, d, og, od, gan = build_models(device, profile)
    row_weights, unobserved, sensor_rows, sensor_ids = load_masks(device, profile)

    checkpoint_dir = result_root / "checkpoints"; preview_dir = result_root / "validation_previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = result_root / "training_history.json"; state_path = result_root / "training_state.json"
    history: list[dict[str, Any]] = []; start_epoch = 1; best = float("inf"); best_epoch = 0; no_improve = 0
    transfer_report = None

    resume_path = resolve_resume(args.resume, result_root)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        g.load_state_dict(ckpt["generator"]); d.load_state_dict(ckpt["discriminator"])
        if "optimizer_g" in ckpt: og.load_state_dict(ckpt["optimizer_g"])
        if "optimizer_d" in ckpt: od.load_state_dict(ckpt["optimizer_d"])
        start_epoch = int(ckpt["epoch"]) + 1
        if history_path.is_file(): history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best_item = min(history, key=lambda x: float(x["val_unobserved_l1"]))
            best = float(best_item["val_unobserved_l1"]); best_epoch = int(best_item["epoch"])
        logging.info("Resumed from %s", resume_path)
    else:
        transfer_report = transfer_a1_checkpoint(args.init_a1.resolve(), g, d, device)
        logging.info("Transferred A1 best checkpoint: %s", transfer_report)

    stage1 = 1 if smoke else args.stage1_epochs
    stage2 = 0 if smoke else args.stage2_epochs
    total_epochs = stage1 + stage2
    logging.info("Experiment: sensor_mode=%s, sensor_ids=%s, sensor_rows=%s", args.sensor_mode, sensor_ids, sensor_rows)
    logging.info("Data train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))
    logging.info("Selection metric: validation L1 on 148 unobserved nodes")

    for epoch in range(start_epoch, total_epochs + 1):
        g.train(); d.train(); stage = 1 if epoch <= stage1 else 2
        lambda_fm = profile.lambda_fm_stage1 if stage == 1 else profile.lambda_fm_stage2
        adversarial = epoch > profile.warmup_epochs
        sums = {k: 0.0 for k in ("g_total","d_total","g_gan","weighted_l1","g1_l1","feature_matching")}; batches=0; d_updates=0
        for batch_index, batch in enumerate(train_loader):
            inputs = batch["input"].to(device, non_blocking=True); targets = batch["target"].to(device, non_blocking=True)
            update_d = adversarial and batch_index % profile.discriminator_update_interval == 0
            losses = train_step(generator=g, discriminator=d, optimizer_g=og, optimizer_d=od, gan_loss=gan, inputs=inputs, targets=targets, profile=profile, lambda_fm=lambda_fm, adversarial_enabled=adversarial, update_d=update_d, row_weights=row_weights)
            for k in sums: sums[k] += losses[k]
            batches += 1; d_updates += int(update_d)
            if args.max_train_batches is not None and batch_index + 1 >= args.max_train_batches: break
        train_metrics = {k: v / max(1, d_updates if k=="d_total" else batches) for k,v in sums.items()}
        val = validate(g, val_loader, device, row_weights, unobserved, args.max_val_batches)
        save_previews(g, val_loader, device, preview_dir, f"epoch_{epoch:03d}", 2 if smoke else 3)
        metrics = {"epoch": epoch, "stage": stage, "sensor_mode": args.sensor_mode, "adversarial_enabled": adversarial, "lambda_fm": lambda_fm, **{f"train_{k}":v for k,v in train_metrics.items()}, **{f"val_{k}":v for k,v in val.items()}}
        history.append(metrics)
        save_checkpoint(checkpoint_dir / "latest.pt", g, d, og, od, epoch, metrics)
        improved = val["unobserved_l1"] < best - args.min_improvement
        if improved:
            best = val["unobserved_l1"]; best_epoch = epoch; no_improve = 0
            save_checkpoint(checkpoint_dir / "best.pt", g, d, og, od, epoch, metrics)
        elif stage == 2:
            no_improve += 1
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        state = {"run_name": args.run_name, "sensor_mode": args.sensor_mode, "latest_epoch": epoch, "best_epoch": best_epoch, "best_val_unobserved_l1": best, "stage2_no_improvement": no_improve, "sensor_node_ids": sensor_ids, "sensor_node_rows": sensor_rows, "training_profile": asdict(profile), "model_config": asdict(config), "a1_transfer": transfer_report}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Epoch %03d/%03d | stage=%d adv=%s | G=%.5f D=%.5f weightedL1=%.5f | val_unobs=%.5f RGB=(%.5f,%.5f,%.5f) diversity=%.3f | best=%d/%.5f", epoch,total_epochs,stage,adversarial,train_metrics["g_total"],train_metrics["d_total"],train_metrics["weighted_l1"],val["unobserved_l1"],val["unobserved_l1_r"],val["unobserved_l1_g"],val["unobserved_l1_b"],val["diversity_ratio"],best_epoch,best)
        if stage == 2 and args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            logging.info("EARLY STOPPING IN STAGE 2: no unobserved-node validation improvement for %d epochs.", no_improve); break

    logging.info("TRAINING FINISHED"); logging.info("Best epoch: %d", best_epoch); logging.info("Best validation unobserved L1: %.6f", best); logging.info("Best checkpoint: %s", checkpoint_dir / "best.pt")


def main() -> None:
    args = parse_args(); profile = TrainProfile(sensor_row_weight=args.sensor_row_weight)
    result_root = (args.output_root or (PACKAGE_ROOT / "results" / args.run_name)).resolve()
    if args.overwrite_run and args.resume is None and result_root.exists(): shutil.rmtree(result_root)
    setup_logging(result_root / "logs" / ("summary.log" if args.mode=="summary" else "smoke.log" if args.mode=="smoke" else "train.log"), append=bool(args.resume))
    set_seed(42)
    if not DATASET_ROOT.is_dir() or not MAPPING_PATH.is_file():
        raise FileNotFoundError("Run 10_prepare_sensor_update_S5_U_rgb.py first.")
    device = torch.device(args.device)
    logging.info("Device: %s", device); logging.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available(): logging.info("GPU: %s", torch.cuda.get_device_name(0))
    logging.info("Result root: %s", result_root)
    if args.mode == "summary": architecture_summary(device, profile)
    else: run_training(args, result_root, device, profile, smoke=args.mode=="smoke")


if __name__ == "__main__":
    main()
