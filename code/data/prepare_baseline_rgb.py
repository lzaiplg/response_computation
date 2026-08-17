# -*- coding: utf-8 -*-
"""
Prepare the formal RGB data set using scales calculated from TRAINING records only.

RGB meaning:
Input:
    R = H1 ground acceleration
    G = H2 ground acceleration
    B = vertical ground acceleration
Target:
    R = X dynamic displacement
    G = Y dynamic displacement
    B = Z dynamic displacement

Images are 1000 pixels wide (time) by 153 pixels high (nodes).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: Path) -> str:
    """Store package-relative metadata so generated mappings survive relocation."""
    try:
        return path.resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=base / "data" / "processed" / "response" / "response_npz",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=base / "config" / "splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "processed" / "baseline_displacement" / "dataset_rgb",
    )
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument(
        "--sample-rows-per-record",
        type=int,
        default=10000,
        help="Deterministic rows sampled per record when estimating target scales.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def estimate_train_scales(
    npz_dir: Path,
    train_names: list[str],
    percentile: float,
    sample_rows_per_record: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_parts: list[np.ndarray] = []
    disp_parts: list[np.ndarray] = []

    for index, name in enumerate(train_names, start=1):
        path = npz_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path) as data:
            input_data = np.abs(
                np.asarray(data["input"], dtype=np.float32).reshape(-1, 3)
            )
            disp = np.abs(
                np.asarray(data["disp"], dtype=np.float32).reshape(-1, 3)
            )

        input_parts.append(input_data)

        if disp.shape[0] > sample_rows_per_record:
            chosen = rng.choice(
                disp.shape[0],
                size=sample_rows_per_record,
                replace=False,
            )
            disp = disp[chosen]
        disp_parts.append(disp)

        if index % 50 == 0 or index == len(train_names):
            print(f"Scale scan {index}/{len(train_names)}")

    input_values = np.concatenate(input_parts, axis=0)
    disp_values = np.concatenate(disp_parts, axis=0)

    input_scales = np.percentile(input_values, percentile, axis=0).astype(
        np.float32
    )
    disp_scales = np.percentile(disp_values, percentile, axis=0).astype(
        np.float32
    )

    for label, scales in (
        ("input", input_scales),
        ("displacement", disp_scales),
    ):
        if np.any(~np.isfinite(scales)) or np.any(scales <= 1e-12):
            raise RuntimeError(f"Invalid {label} scales: {scales}")

    return input_scales, disp_scales


def to_rgb(array: np.ndarray, scales: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        array.astype(np.float32) / scales.reshape(1, 1, 3),
        -1.0,
        1.0,
    )
    return np.rint((normalized + 1.0) * 127.5).astype(np.uint8)


def save_rgb(array: np.ndarray, scales: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb(array, scales), mode="RGB").save(path)


def prepare_one_split(
    split: str,
    names: list[str],
    npz_dir: Path,
    output_dir: Path,
    input_scales: np.ndarray,
    disp_scales: np.ndarray,
) -> dict[str, object]:
    input_dir = output_dir / split / "input"
    target_dir = output_dir / split / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    clipped_input = np.zeros(3, dtype=np.int64)
    clipped_disp = np.zeros(3, dtype=np.int64)
    input_total = np.zeros(3, dtype=np.int64)
    disp_total = np.zeros(3, dtype=np.int64)

    for index, name in enumerate(names, start=1):
        npz_path = npz_dir / f"{name}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path) as data:
            input_data = np.asarray(data["input"], dtype=np.float32)
            disp = np.asarray(data["disp"], dtype=np.float32)

        if input_data.shape != (1000, 3):
            raise ValueError(f"{name}: input shape {input_data.shape}")
        if disp.shape != (153, 1000, 3):
            raise ValueError(f"{name}: disp shape {disp.shape}")
        if np.any(~np.isfinite(input_data)) or np.any(~np.isfinite(disp)):
            raise ValueError(f"{name}: NaN or Inf")

        nodal_input = np.broadcast_to(
            input_data[None, :, :],
            (153, 1000, 3),
        ).copy()

        clipped_input += np.sum(
            np.abs(nodal_input) > input_scales.reshape(1, 1, 3),
            axis=(0, 1),
        )
        clipped_disp += np.sum(
            np.abs(disp) > disp_scales.reshape(1, 1, 3),
            axis=(0, 1),
        )
        input_total += nodal_input.shape[0] * nodal_input.shape[1]
        disp_total += disp.shape[0] * disp.shape[1]

        save_rgb(
            nodal_input,
            input_scales,
            input_dir / f"{name}.png",
        )
        save_rgb(
            disp,
            disp_scales,
            target_dir / f"{name}.png",
        )

        if index % 25 == 0 or index == len(names):
            print(f"{split}: {index}/{len(names)}")

    return {
        "records": len(names),
        "input_directory": portable_path(input_dir),
        "target_directory": portable_path(target_dir),
        "input_clipped_fraction_xyz": (
            clipped_input / np.maximum(input_total, 1)
        ).tolist(),
        "target_clipped_fraction_xyz": (
            clipped_disp / np.maximum(disp_total, 1)
        ).tolist(),
    }


def verify_dataset(
    output_dir: Path,
    expected_counts: dict[str, int],
) -> None:
    for split, expected in expected_counts.items():
        input_files = sorted((output_dir / split / "input").glob("*.png"))
        target_files = sorted((output_dir / split / "target").glob("*.png"))
        if len(input_files) != expected or len(target_files) != expected:
            raise RuntimeError(
                f"{split}: count mismatch input={len(input_files)}, "
                f"target={len(target_files)}, expected={expected}"
            )
        if {p.name for p in input_files} != {p.name for p in target_files}:
            raise RuntimeError(f"{split}: input/target filenames differ")

        for path in input_files[:3] + target_files[:3]:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise RuntimeError(f"{path}: mode={image.mode}")
                if image.size != (1000, 153):
                    raise RuntimeError(f"{path}: size={image.size}")


def main() -> None:
    args = parse_args()
    npz_dir = args.npz_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()

    splits = {
        "train": read_split(split_dir / "train.txt"),
        "val": read_split(split_dir / "val.txt"),
        "test": read_split(split_dir / "test.txt"),
    }

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite only when intentionally regenerating formal RGB."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Calculating normalization scales from TRAINING records only...")
    input_scales, disp_scales = estimate_train_scales(
        npz_dir,
        splits["train"],
        args.percentile,
        args.sample_rows_per_record,
        args.seed,
    )

    split_reports = {}
    for split in ("train", "val", "test"):
        split_reports[split] = prepare_one_split(
            split,
            splits[split],
            npz_dir,
            output_dir,
            input_scales,
            disp_scales,
        )

    verify_dataset(
        output_dir,
        {split: len(names) for split, names in splits.items()},
    )

    mapping = {
        "rgb_channel_meaning": {
            "input_R": "H1 ground acceleration",
            "input_G": "H2 ground acceleration",
            "input_B": "vertical ground acceleration",
            "target_R": "X dynamic displacement",
            "target_G": "Y dynamic displacement",
            "target_B": "Z dynamic displacement",
        },
        "image_width": 1000,
        "image_height": 153,
        "mapping": "clip(value/scale,-1,1), then RGB=round((normalized+1)*127.5)",
        "scale_source": "training split only",
        "percentile": args.percentile,
        "sampling_seed": args.seed,
        "sample_rows_per_training_record": args.sample_rows_per_record,
        "input_scales_m_s2": input_scales.tolist(),
        "dynamic_displacement_scales_m": disp_scales.tolist(),
        "splits": split_reports,
    }
    mapping_path = output_dir.parent / "training_rgb_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    split_copy_dir = output_dir.parent / "splits"
    split_copy_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        shutil.copy2(split_dir / f"{split}.txt", split_copy_dir / f"{split}.txt")

    print("")
    print("FORMAL RGB PREPARATION PASS")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Input scales: {input_scales.tolist()}")
    print(f"Displacement scales: {disp_scales.tolist()}")
    print(f"Dataset: {output_dir}")
    print(f"Mapping: {mapping_path}")


if __name__ == "__main__":
    main()
