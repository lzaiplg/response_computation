# -*- coding: utf-8 -*-
"""
Train the corrected/stabilized pix2pixHD-style baseline on the fixed 7:2:1 RGB data.

This script separates two kinds of changes:

A. Architecture corrections that are closer to the paper description:
   - real U-Net skip connections in G1;
   - parallel multi-scale local branches in G2;
   - reflection padding to 160x1024 and exact cropping to 153x1000.

B. Optional engineering stabilization because the paper does not disclose every
   optimizer detail and the previous run showed a discriminator collapse:
   - lower discriminator learning rate;
   - reconstruction warm-up;
   - stronger L1 reconstruction;
   - discriminator update interval;
   - gradient clipping;
   - early stopping only after stage 2 has started.

The recommended profile is "stable". Use "paper_closer" only as an additional
ablation because it may reproduce the former adversarial instability.
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

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pix2pixHD_seismic_model import (
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
FORMAL_ROOT = PACKAGE_ROOT / "data" / "processed" / "baseline_displacement"
DATASET_ROOT = FORMAL_ROOT / "dataset_rgb"


@dataclass
class TrainProfile:
    learning_rate_g: float
    learning_rate_d: float
    lambda_l1: float
    lambda_g1_l1: float
    lambda_fm_stage1: float
    lambda_fm_stage2: float
    warmup_epochs: int
    discriminator_update_interval: int
    real_label: float
    gradient_clip: float


def build_profile(name: str) -> TrainProfile:
    if name == "paper_closer":
        return TrainProfile(
            learning_rate_g=2.0e-4,
            learning_rate_d=2.0e-4,
            lambda_l1=1.0,
            lambda_g1_l1=1.0,
            lambda_fm_stage1=1.0,
            lambda_fm_stage2=10.0,
            warmup_epochs=0,
            discriminator_update_interval=1,
            real_label=1.0,
            gradient_clip=0.0,
        )

    # Recommended engineering-stable reproduction baseline.
    return TrainProfile(
        learning_rate_g=1.0e-4,
        learning_rate_d=2.5e-5,
        lambda_l1=50.0,
        lambda_g1_l1=10.0,
        lambda_fm_stage1=1.0,
        lambda_fm_stage2=10.0,
        warmup_epochs=5,
        discriminator_update_interval=2,
        real_label=0.9,
        gradient_clip=5.0,
    )


class FixedSplitRGBDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        limit: int | None = None,
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

        input_files = {p.name: p for p in input_dir.glob("*.png")}
        target_files = {p.name: p for p in target_dir.glob("*.png")}
        names = sorted(input_files.keys() & target_files.keys())
        if limit is not None:
            names = names[: max(0, limit)]
        if not names:
            raise FileNotFoundError(
                f"No paired RGB files found:\n{input_dir}\n{target_dir}"
            )
        self.samples = [
            (input_files[name], target_files[name], name)
            for name in names
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        input_path, target_path, name = self.samples[index]
        with Image.open(input_path) as image:
            input_image = image.convert("RGB")
        with Image.open(target_path) as image:
            target_image = image.convert("RGB")

        if input_image.size != (1000, 153):
            raise ValueError(f"{name}: input size={input_image.size}")
        if target_image.size != (1000, 153):
            raise ValueError(f"{name}: target size={target_image.size}")

        return {
            "input": self.transform(input_image),
            "target": self.transform(target_image),
            "name": name,
        }


def setup_logging(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(
        path,
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_for(
    split: str,
    batch_size: int,
    limit: int | None,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = FixedSplitRGBDataset(
        DATASET_ROOT / split / "input",
        DATASET_ROOT / split / "target",
        limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def pairwise_diversity(tensors: list[torch.Tensor]) -> float:
    if len(tensors) < 2:
        return 0.0
    values = [
        F.l1_loss(left, right).item()
        for left, right in combinations(tensors, 2)
    ]
    return float(np.mean(values))


@torch.no_grad()
def validate(
    generator,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    diversity_samples: int = 4,
) -> dict[str, float]:
    generator.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_channel_l1 = np.zeros(3, dtype=np.float64)
    count = 0

    diversity_predictions: list[torch.Tensor] = []
    diversity_targets: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        generated, _ = generator(inputs)

        total_l1 += F.l1_loss(generated, targets).item()
        total_mse += F.mse_loss(generated, targets).item()
        for channel in range(3):
            total_channel_l1[channel] += F.l1_loss(
                generated[:, channel],
                targets[:, channel],
            ).item()
        count += 1

        if len(diversity_predictions) < diversity_samples:
            for item_index in range(generated.shape[0]):
                if len(diversity_predictions) >= diversity_samples:
                    break
                diversity_predictions.append(
                    generated[item_index].detach().cpu()
                )
                diversity_targets.append(
                    targets[item_index].detach().cpu()
                )

        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    if count == 0:
        raise RuntimeError("Validation loader is empty.")

    prediction_diversity = pairwise_diversity(diversity_predictions)
    target_diversity = pairwise_diversity(diversity_targets)
    ratio = prediction_diversity / max(target_diversity, 1.0e-12)

    return {
        "l1": total_l1 / count,
        "mse": total_mse / count,
        "l1_r": float(total_channel_l1[0] / count),
        "l1_g": float(total_channel_l1[1] / count),
        "l1_b": float(total_channel_l1[2] / count),
        "prediction_diversity": prediction_diversity,
        "target_diversity": target_diversity,
        "diversity_ratio": ratio,
    }


@torch.no_grad()
def save_fixed_previews(
    generator,
    loader: DataLoader,
    device: torch.device,
    directory: Path,
    prefix: str,
    max_samples: int = 3,
) -> None:
    generator.eval()
    directory.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        generated, _ = generator(inputs)

        for index in range(inputs.shape[0]):
            name = Path(batch["name"][index]).stem
            tensor_to_image(inputs[index]).save(
                directory / f"{prefix}_{name}_input.png"
            )
            tensor_to_image(generated[index]).save(
                directory / f"{prefix}_{name}_prediction.png"
            )
            tensor_to_image(targets[index]).save(
                directory / f"{prefix}_{name}_target.png"
            )
            saved += 1
            if saved >= max_samples:
                return


def run_one_step(
    *,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    gan_loss,
    inputs,
    targets,
    profile: TrainProfile,
    lambda_fm: float,
    adversarial_enabled: bool,
    update_discriminator: bool,
) -> dict[str, float]:
    loss_d_value = 0.0
    if adversarial_enabled and update_discriminator:
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)

        with torch.no_grad():
            generated_for_d, _ = generator(inputs)

        pred_real = discriminator(inputs, targets)
        pred_fake = discriminator(inputs, generated_for_d)
        loss_d = 0.5 * (
            gan_loss(pred_real, True)
            + gan_loss(pred_fake, False)
        )
        loss_d.backward()

        if profile.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                profile.gradient_clip,
            )
        optimizer_d.step()
        loss_d_value = float(loss_d.item())

    set_requires_grad(discriminator, False)
    optimizer_g.zero_grad(set_to_none=True)

    # Recompute G output after the optional D update.
    generated, g1_low = generator(inputs)
    low_targets = F.interpolate(
        targets,
        size=g1_low.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    loss_l1 = F.l1_loss(generated, targets)
    loss_g1_l1 = F.l1_loss(g1_low, low_targets)

    if adversarial_enabled:
        pred_fake_for_g = discriminator(inputs, generated)
        with torch.no_grad():
            pred_real_for_fm = discriminator(inputs, targets)
        loss_g_gan = gan_loss(pred_fake_for_g, True)
        loss_fm = feature_matching_loss(
            pred_fake_for_g,
            pred_real_for_fm,
        )
    else:
        loss_g_gan = torch.zeros((), device=inputs.device)
        loss_fm = torch.zeros((), device=inputs.device)

    loss_g = (
        profile.lambda_l1 * loss_l1
        + profile.lambda_g1_l1 * loss_g1_l1
        + loss_g_gan
        + lambda_fm * loss_fm
    )
    loss_g.backward()

    if profile.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(),
            profile.gradient_clip,
        )
    optimizer_g.step()
    set_requires_grad(discriminator, True)

    losses = {
        "g_total": float(loss_g.item()),
        "d_total": loss_d_value,
        "g_gan": float(loss_g_gan.item()),
        "l1": float(loss_l1.item()),
        "g1_l1": float(loss_g1_l1.item()),
        "feature_matching": float(loss_fm.item()),
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError(f"Non-finite loss: {losses}")
    return losses


def resolve_checkpoint(value: str | None, result_root: Path) -> Path | None:
    if value is None:
        return None
    if value.lower() == "latest":
        return result_root / "checkpoints" / "latest.pt"
    if value.lower() == "best":
        return result_root / "checkpoints" / "best.pt"
    return Path(value).resolve()


def build_models_and_optimizers(
    device: torch.device,
    profile: TrainProfile,
):
    model_config = ModelConfig()
    generator, discriminator = make_models(model_config, device)

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=profile.learning_rate_g,
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=profile.learning_rate_d,
        betas=(0.5, 0.999),
    )
    gan_loss = GANLoss(real_label=profile.real_label).to(device)
    return (
        model_config,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        gan_loss,
    )


def architecture_summary(
    device: torch.device,
    profile: TrainProfile,
) -> None:
    (
        _,
        generator,
        discriminator,
        _,
        _,
        _,
    ) = build_models_and_optimizers(device, profile)

    for line in corrected_architecture_report(
        generator,
        discriminator,
    ):
        logging.info(line)

    dummy = torch.zeros(1, 3, 153, 1000, device=device)
    with torch.no_grad():
        output, g1_low = generator(dummy)
        d_outputs = discriminator(dummy, output)

    logging.info("Input shape: %s", tuple(dummy.shape))
    logging.info("Output shape: %s", tuple(output.shape))
    logging.info("G1 low output: %s", tuple(g1_low.shape))
    logging.info(
        "Discriminator outputs: %s",
        [tuple(scale[-1].shape) for scale in d_outputs],
    )


def smoke_test(
    args,
    result_root: Path,
    device: torch.device,
    profile: TrainProfile,
) -> None:
    train_loader = loader_for(
        "train",
        args.batch_size,
        args.smoke_train_samples,
        True,
        args.num_workers,
    )
    val_loader = loader_for(
        "val",
        args.batch_size,
        args.smoke_val_samples,
        False,
        args.num_workers,
    )

    (
        model_config,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        gan_loss,
    ) = build_models_and_optimizers(device, profile)

    batch = next(iter(train_loader))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

    losses = run_one_step(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        gan_loss=gan_loss,
        inputs=inputs,
        targets=targets,
        profile=profile,
        lambda_fm=profile.lambda_fm_stage1,
        adversarial_enabled=False,
        update_discriminator=False,
    )

    with torch.no_grad():
        outputs, _ = generator(inputs)

    if outputs.shape != targets.shape:
        raise RuntimeError(
            f"Output {outputs.shape} != target {targets.shape}"
        )

    metrics = validate(
        generator,
        val_loader,
        device,
        max_batches=1,
    )

    smoke_dir = result_root / "smoke"
    save_fixed_previews(
        generator,
        val_loader,
        device,
        smoke_dir / "previews",
        "smoke",
        max_samples=2,
    )

    checkpoint_path = smoke_dir / "smoke_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        epoch=0,
        metrics={**losses, **{f"val_{k}": v for k, v in metrics.items()}},
    )

    # Check that a new model can reload the checkpoint.
    (
        _,
        new_generator,
        new_discriminator,
        _,
        _,
        _,
    ) = build_models_and_optimizers(device, profile)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    new_generator.load_state_dict(checkpoint["generator"])
    new_discriminator.load_state_dict(checkpoint["discriminator"])

    new_generator.eval()
    generator.eval()
    with torch.no_grad():
        original_output, _ = generator(inputs)
        reloaded_output, _ = new_generator(inputs)
    reload_difference = torch.max(
        torch.abs(original_output - reloaded_output)
    ).item()
    if reload_difference > 1.0e-6:
        raise RuntimeError(
            f"Checkpoint reload difference={reload_difference}"
        )

    report = {
        "pass": True,
        "profile": args.profile,
        "model_config": asdict(model_config),
        "training_profile": asdict(profile),
        "device": str(device),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "output_shape": list(outputs.shape),
        "losses": losses,
        "validation": metrics,
        "checkpoint_reload_max_difference": reload_difference,
    }
    (smoke_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info("Input shape: %s", tuple(inputs.shape))
    logging.info("Target shape: %s", tuple(targets.shape))
    logging.info("Output shape: %s", tuple(outputs.shape))
    logging.info("Losses: %s", losses)
    logging.info("Validation: %s", metrics)
    logging.info("SMOKE TEST PASS")


def train(
    args,
    result_root: Path,
    device: torch.device,
    profile: TrainProfile,
) -> None:
    train_loader = loader_for(
        "train",
        args.batch_size,
        args.limit_train_samples,
        True,
        args.num_workers,
    )
    val_loader = loader_for(
        "val",
        args.batch_size,
        args.limit_val_samples,
        False,
        args.num_workers,
    )

    (
        model_config,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        gan_loss,
    ) = build_models_and_optimizers(device, profile)

    for line in corrected_architecture_report(
        generator,
        discriminator,
    ):
        logging.info(line)

    checkpoint_dir = result_root / "checkpoints"
    preview_dir = result_root / "validation_previews"
    history_path = result_root / "training_history.json"
    state_path = result_root / "training_state.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    start_epoch = 1
    best_val_l1 = float("inf")
    best_epoch = 0
    stage2_no_improvement = 0

    resume_path = resolve_checkpoint(args.resume, result_root)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)

        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("format") != "pix2pixHD_paper_corrected_v1":
            raise RuntimeError(
                "This checkpoint belongs to the old architecture and cannot "
                "be loaded into the corrected model."
            )

        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        start_epoch = int(checkpoint["epoch"]) + 1
        if history_path.is_file():
            history = json.loads(
                history_path.read_text(encoding="utf-8")
            )
        valid = [
            item for item in history
            if isinstance(item, dict) and "val_l1" in item
        ]
        if valid:
            best_item = min(valid, key=lambda item: item["val_l1"])
            best_val_l1 = float(best_item["val_l1"])
            best_epoch = int(best_item["epoch"])

        logging.info("Resumed from: %s", resume_path)
        logging.info("Starting epoch: %d", start_epoch)
    else:
        if (
            (result_root / "checkpoints" / "latest.pt").exists()
            and not args.overwrite_run
        ):
            raise FileExistsError(
                f"Existing training run found: {result_root}\n"
                "Use --resume latest to continue or --overwrite-run to restart."
            )

    total_epochs = args.stage1_epochs + args.stage2_epochs
    logging.info(
        "Data: train=%d, validation=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    logging.info(
        "Training schedule: warmup=%d, stage1=%d, stage2=%d, total=%d",
        profile.warmup_epochs,
        args.stage1_epochs,
        args.stage2_epochs,
        total_epochs,
    )
    logging.info(
        "LR G=%.2e, LR D=%.2e, lambda_L1=%.1f, lambda_G1=%.1f",
        profile.learning_rate_g,
        profile.learning_rate_d,
        profile.lambda_l1,
        profile.lambda_g1_l1,
    )

    d_collapse_epochs = 0

    for epoch in range(start_epoch, total_epochs + 1):
        generator.train()
        discriminator.train()

        stage = 1 if epoch <= args.stage1_epochs else 2
        lambda_fm = (
            profile.lambda_fm_stage1
            if stage == 1
            else profile.lambda_fm_stage2
        )
        adversarial_enabled = epoch > profile.warmup_epochs

        sums = {
            "g_total": 0.0,
            "d_total": 0.0,
            "g_gan": 0.0,
            "l1": 0.0,
            "g1_l1": 0.0,
            "feature_matching": 0.0,
        }
        batches = 0
        d_updates = 0

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            update_d = (
                adversarial_enabled
                and batch_index % profile.discriminator_update_interval == 0
            )

            losses = run_one_step(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                gan_loss=gan_loss,
                inputs=inputs,
                targets=targets,
                profile=profile,
                lambda_fm=lambda_fm,
                adversarial_enabled=adversarial_enabled,
                update_discriminator=update_d,
            )

            for key in sums:
                sums[key] += losses[key]
            batches += 1
            if update_d:
                d_updates += 1

            if (
                args.max_train_batches is not None
                and batch_index + 1 >= args.max_train_batches
            ):
                break

        train_metrics = {
            key: (
                value / max(1, d_updates)
                if key == "d_total"
                else value / max(1, batches)
            )
            for key, value in sums.items()
        }
        val_metrics = validate(
            generator,
            val_loader,
            device,
            args.max_val_batches,
        )

        save_fixed_previews(
            generator,
            val_loader,
            device,
            preview_dir,
            f"epoch_{epoch:03d}",
            max_samples=3,
        )

        metrics = {
            "epoch": epoch,
            "stage": stage,
            "adversarial_enabled": adversarial_enabled,
            "lambda_fm": lambda_fm,
            "d_updates": d_updates,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(metrics)

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch,
            metrics,
        )

        improved = val_metrics["l1"] < best_val_l1 - args.min_improvement
        if improved:
            best_val_l1 = val_metrics["l1"]
            best_epoch = epoch
            stage2_no_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch,
                metrics,
            )
        elif stage == 2:
            stage2_no_improvement += 1

        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state = {
            "profile": args.profile,
            "latest_epoch": epoch,
            "current_stage": stage,
            "best_epoch": best_epoch,
            "best_val_l1": best_val_l1,
            "stage2_epochs_without_improvement": stage2_no_improvement,
            "training_profile": asdict(profile),
            "model_config": asdict(model_config),
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logging.info(
            "Epoch %03d/%03d | stage=%d adv=%s | "
            "G=%.5f D=%.5f L1=%.5f | "
            "val_L1=%.5f val_RGB=(%.5f,%.5f,%.5f) | "
            "diversity ratio=%.3f | best=%d/%.5f",
            epoch,
            total_epochs,
            stage,
            adversarial_enabled,
            train_metrics["g_total"],
            train_metrics["d_total"],
            train_metrics["l1"],
            val_metrics["l1"],
            val_metrics["l1_r"],
            val_metrics["l1_g"],
            val_metrics["l1_b"],
            val_metrics["diversity_ratio"],
            best_epoch,
            best_val_l1,
        )

        if adversarial_enabled and train_metrics["d_total"] < 1.0e-3:
            d_collapse_epochs += 1
        else:
            d_collapse_epochs = 0

        if d_collapse_epochs >= 3:
            logging.warning(
                "WARNING: discriminator loss has remained below 1e-3 for "
                "three epochs. Inspect prediction diversity and previews."
            )

        if val_metrics["diversity_ratio"] < 0.10:
            logging.warning(
                "WARNING: prediction diversity is less than 10%% of target "
                "diversity; possible conditional mode collapse."
            )

        # Do not early-stop during stage 1. Stage 2 must actually run.
        if (
            stage == 2
            and args.early_stop_patience > 0
            and stage2_no_improvement >= args.early_stop_patience
        ):
            logging.info(
                "EARLY STOPPING IN STAGE 2: no validation improvement for %d epochs.",
                stage2_no_improvement,
            )
            break

    logging.info("TRAINING FINISHED")
    logging.info("Best epoch: %d", best_epoch)
    logging.info("Best validation L1: %.6f", best_val_l1)
    logging.info("Best checkpoint: %s", checkpoint_dir / "best.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["summary", "smoke", "train"],
        default="smoke",
    )
    parser.add_argument(
        "--profile",
        choices=["stable", "paper_closer"],
        default="stable",
    )
    parser.add_argument(
        "--run-name",
        default="pix2pixHD_paper_corrected_stable",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="RGB dataset root; defaults to processed/baseline_displacement/dataset_rgb.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Output directory; defaults to package/results/<run-name>.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-improvement", type=float, default=1.0e-5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--smoke-train-samples", type=int, default=4)
    parser.add_argument("--smoke-val-samples", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT
    args = parse_args()
    DATASET_ROOT = (args.dataset_root or DATASET_ROOT).resolve()
    result_root = (args.result_root or (PACKAGE_ROOT / "results" / args.run_name)).resolve()
    profile = build_profile(args.profile)

    if args.overwrite_run and args.resume is None and result_root.exists():
        shutil.rmtree(result_root)

    log_name = (
        "summary.log"
        if args.mode == "summary"
        else "smoke.log"
        if args.mode == "smoke"
        else "train.log"
    )
    setup_logging(
        result_root / "logs" / log_name,
        append=bool(args.resume),
    )
    set_seed(42)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Formal RGB dataset not found: {DATASET_ROOT}"
        )

    device = torch.device(args.device)
    logging.info("Device: %s", device)
    logging.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
    logging.info("Profile: %s", args.profile)
    logging.info("Result root: %s", result_root)

    if args.mode == "summary":
        architecture_summary(device, profile)
    elif args.mode == "smoke":
        smoke_test(args, result_root, device, profile)
    else:
        train(args, result_root, device, profile)


if __name__ == "__main__":
    main()
