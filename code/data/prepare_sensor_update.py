# -*- coding: utf-8 -*-
"""Build the 7-channel image-style S5-U sensor-update data set.

Files per sample:
    ground/*.png  : original 3-channel ground-motion RGB
    sensor/*.png  : sparse 3-channel sensor-acceleration RGB
    mask/*.png    : 1-channel sensor-position mask
    target/*.png  : original 3-channel displacement RGB

The network concatenates ground(3)+sensor(3)+mask(1)=7 channels in memory.
The target remains the original 153x1000 RGB displacement image.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT
BASELINE_ROOT = PACKAGE_ROOT / "data" / "processed" / "baseline_displacement"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "processed" / "sensor_update_displacement"
FIXED_NODE_IDS = list(range(91, 105)) + list(range(11284, 11292))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz-dir", type=Path, default=PACKAGE_ROOT / "data" / "processed" / "response" / "response_npz")
    p.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    p.add_argument("--layout", type=Path, default=PACKAGE_ROOT / "config" / "sensor_layout.json")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--source-accel-mode", choices=("auto", "relative", "absolute"), default="auto")
    p.add_argument("--sensor-accel-mode", choices=("relative", "absolute"), default="absolute")
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--detection-records", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def detect_source_mode(npz_dir: Path, names: list[str], max_records: int) -> tuple[str, dict[str, float]]:
    rel_scores: list[float] = []
    abs_scores: list[float] = []
    for name in names[: max(1, max_records)]:
        with np.load(npz_dir / f"{name}.npz") as data:
            ground = np.asarray(data["input"], dtype=np.float64)
            accel = np.asarray(data["accel"], dtype=np.float64)
            node_ids = np.asarray(data["node_ids"]).astype(int)
        row_map = {int(node): row for row, node in enumerate(node_ids)}
        rows = [row_map[node] for node in FIXED_NODE_IDS if node in row_map]
        if not rows:
            raise RuntimeError("Cannot auto-detect acceleration definition: fixed nodes are absent.")
        fixed = accel[rows]
        ground_map = np.broadcast_to(ground[None, :, :], fixed.shape)
        scale = max(rms(ground_map), 1.0e-12)
        rel_scores.append(rms(fixed) / scale)
        abs_scores.append(rms(fixed - ground_map) / scale)
    rel = float(np.median(rel_scores))
    absolute = float(np.median(abs_scores))
    mode = "relative" if rel < absolute else "absolute"
    return mode, {"relative_score": rel, "absolute_score": absolute}


def convert_acceleration(node_accel: np.ndarray, ground: np.ndarray, source: str, target: str) -> np.ndarray:
    if source == target:
        return node_accel.astype(np.float32, copy=False)
    ground_map = ground[None, :, :]
    if source == "relative" and target == "absolute":
        return (node_accel + ground_map).astype(np.float32)
    if source == "absolute" and target == "relative":
        return (node_accel - ground_map).astype(np.float32)
    raise ValueError((source, target))


def to_rgb(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    normalized = np.clip(values.astype(np.float32) / scales.reshape(1, 1, 3), -1.0, 1.0)
    return np.rint((normalized + 1.0) * 127.5).astype(np.uint8)


def save_preview(path: Path, ground: np.ndarray, sensor: np.ndarray, mask: np.ndarray, target: np.ndarray, title: str) -> None:
    panels = [("Ground RGB", ground), ("Sparse sensor RGB", sensor), ("Sensor mask", np.repeat(mask[..., None], 3, axis=2)), ("Target displacement RGB", target)]
    width, height = 1000, 153
    title_h = 24
    canvas = Image.new("RGB", (width, (height + title_h) * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, array in panels:
        draw.text((8, y + 5), f"{title} | {label}", fill="black")
        canvas.paste(Image.fromarray(array.astype(np.uint8), mode="RGB"), (0, y + title_h))
        y += height + title_h
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    a = parse_args()
    split_dir = a.baseline_root / "splits"
    baseline_dataset = a.baseline_root / "dataset_rgb"
    original_mapping_path = a.baseline_root / "training_rgb_mapping.json"
    if not original_mapping_path.is_file():
        raise FileNotFoundError(original_mapping_path)
    original_mapping = json.loads(original_mapping_path.read_text(encoding="utf-8"))
    layout = json.loads(a.layout.read_text(encoding="utf-8"))
    sensor_ids = [int(x) for x in layout["sensor_node_ids"]]
    splits = {s: read_lines(split_dir / f"{s}.txt") for s in ("train", "val", "test")}

    source_mode = a.source_accel_mode
    detection = None
    if source_mode == "auto":
        source_mode, detection = detect_source_mode(a.npz_dir, splits["train"], a.detection_records)
    print(f"NPZ acceleration definition: {source_mode}")
    print(f"Sensor image quantity: {a.sensor_accel_mode}")

    if a.output_root.exists():
        if a.overwrite:
            shutil.rmtree(a.output_root)
        elif any(a.output_root.iterdir()):
            raise FileExistsError(f"Output is not empty: {a.output_root}; use --overwrite")
    a.output_root.mkdir(parents=True, exist_ok=True)

    # Sensor scales are computed using only the 549 training records.
    training_values: list[np.ndarray] = []
    sensor_rows: list[int] | None = None
    sensor_coordinates: dict[str, list[float]] = {}
    reference_node_ids: np.ndarray | None = None
    for index, name in enumerate(splits["train"], start=1):
        with np.load(a.npz_dir / f"{name}.npz") as data:
            ground = np.asarray(data["input"], dtype=np.float32)
            accel = np.asarray(data["accel"], dtype=np.float32)
            node_ids = np.asarray(data["node_ids"]).astype(int)
            coords = np.asarray(data["node_coordinates"], dtype=np.float64)
        row_map = {int(node): row for row, node in enumerate(node_ids)}
        missing = [node for node in sensor_ids if node not in row_map]
        if missing:
            raise KeyError(f"{name}: missing sensor nodes {missing}")
        rows = [row_map[node] for node in sensor_ids]
        if sensor_rows is None:
            sensor_rows = rows
            reference_node_ids = node_ids.copy()
            sensor_coordinates = {str(node): coords[row_map[node]].tolist() for node in sensor_ids}
        elif rows != sensor_rows or not np.array_equal(node_ids, reference_node_ids):
            raise RuntimeError(f"{name}: node ordering changed")
        converted = convert_acceleration(accel, ground, source_mode, a.sensor_accel_mode)
        training_values.append(np.abs(converted[rows].reshape(-1, 3)))
        if index % 50 == 0 or index == len(splits["train"]):
            print(f"Sensor scale scan {index}/{len(splits['train'])}")

    sensor_scales = np.percentile(np.concatenate(training_values, axis=0), a.percentile, axis=0).astype(np.float32)
    if np.any(~np.isfinite(sensor_scales)) or np.any(sensor_scales <= 1.0e-12):
        raise RuntimeError(f"Invalid sensor scales: {sensor_scales}")
    assert sensor_rows is not None

    manifest_rows: list[dict[str, object]] = []
    split_reports: dict[str, object] = {}
    for split, names in splits.items():
        split_root = a.output_root / "dataset_rgb" / split
        for sub in ("ground", "sensor", "mask", "target"):
            (split_root / sub).mkdir(parents=True, exist_ok=True)
        clipped = np.zeros(3, dtype=np.int64)
        total = np.zeros(3, dtype=np.int64)
        for index, name in enumerate(names, start=1):
            source_ground = baseline_dataset / split / "input" / f"{name}.png"
            source_target = baseline_dataset / split / "target" / f"{name}.png"
            if not source_ground.is_file() or not source_target.is_file():
                raise FileNotFoundError(f"Missing baseline RGB for {name}")
            with np.load(a.npz_dir / f"{name}.npz") as data:
                ground = np.asarray(data["input"], dtype=np.float32)
                accel = np.asarray(data["accel"], dtype=np.float32)
                node_ids = np.asarray(data["node_ids"]).astype(int)
            row_map = {int(node): row for row, node in enumerate(node_ids)}
            rows = [row_map[node] for node in sensor_ids]
            converted = convert_acceleration(accel, ground, source_mode, a.sensor_accel_mode)
            sparse = np.zeros((153, 1000, 3), dtype=np.float32)
            sparse[rows] = converted[rows]
            mask = np.zeros((153, 1000), dtype=np.uint8)
            mask[rows, :] = 255
            clipped += np.sum(np.abs(sparse) > sensor_scales.reshape(1, 1, 3), axis=(0, 1))
            total += sparse.shape[0] * sparse.shape[1]
            sensor_rgb = to_rgb(sparse, sensor_scales)

            shutil.copy2(source_ground, split_root / "ground" / source_ground.name)
            shutil.copy2(source_target, split_root / "target" / source_target.name)
            Image.fromarray(sensor_rgb, mode="RGB").save(split_root / "sensor" / f"{name}.png")
            Image.fromarray(mask, mode="L").save(split_root / "mask" / f"{name}.png")

            if index <= 2:
                with Image.open(source_ground) as im:
                    ground_rgb = np.asarray(im.convert("RGB"))
                with Image.open(source_target) as im:
                    target_rgb = np.asarray(im.convert("RGB"))
                save_preview(a.output_root / "previews" / f"{split}_{index:02d}_{name}.png", ground_rgb, sensor_rgb, mask, target_rgb, name)

            manifest_rows.append({
                "split": split,
                "record": name,
                "ground_png": str((split_root / "ground" / f"{name}.png").relative_to(a.output_root)),
                "sensor_png": str((split_root / "sensor" / f"{name}.png").relative_to(a.output_root)),
                "mask_png": str((split_root / "mask" / f"{name}.png").relative_to(a.output_root)),
                "target_png": str((split_root / "target" / f"{name}.png").relative_to(a.output_root)),
            })
            if index % 25 == 0 or index == len(names):
                print(f"{split}: {index}/{len(names)}")
        split_reports[split] = {
            "records": len(names),
            "sensor_clipped_fraction_xyz": (clipped / np.maximum(total, 1)).tolist(),
        }

    with (a.output_root / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)

    shutil.copy2(a.layout, a.output_root / "sensor_layout.json")
    mapping = {
        "dataset_name": "formal_sensor_update_S5_U_7ch",
        "input_channels": {
            "0_2": "ground acceleration RGB, exactly matching A1",
            "3_5": f"sparse {a.sensor_accel_mode} sensor acceleration RGB",
            "6": "sensor position mask (0=no sensor, 1=sensor)",
        },
        "target_channels": "original A1 displacement RGB (R=X,G=Y,B=Z)",
        "network_input_shape_chw": [7, 153, 1000],
        "network_target_shape_chw": [3, 153, 1000],
        "sensor_node_ids": sensor_ids,
        "sensor_node_rows": sensor_rows,
        "sensor_coordinates": sensor_coordinates,
        "unobserved_node_count": 148,
        "source_npz_acceleration_definition": source_mode,
        "sensor_input_acceleration_definition": a.sensor_accel_mode,
        "source_mode_detection": detection,
        "sensor_acceleration_scales": sensor_scales.tolist(),
        "sensor_acceleration_scale_unit": "same unit as NPZ accel/input",
        "sensor_scale_percentile": a.percentile,
        "sensor_scale_source": "training split only",
        "ground_acceleration_scales_m_s2": original_mapping.get("input_scales_m_s2"),
        "dynamic_displacement_scales_m": original_mapping.get("dynamic_displacement_scales_m"),
        "rgb_mapping": "clip(value/scale,-1,1), RGB=round((normalized+1)*127.5)",
        "mask_mapping": "PNG 0/255; loaded as tensor 0/1",
        "splits": split_reports,
    }
    (a.output_root / "sensor_update_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    # Verify counts and sizes.
    for split, names in splits.items():
        for sub in ("ground", "sensor", "mask", "target"):
            files = list((a.output_root / "dataset_rgb" / split / sub).glob("*.png"))
            if len(files) != len(names):
                raise RuntimeError(f"{split}/{sub}: {len(files)} != {len(names)}")
        with Image.open(a.output_root / "dataset_rgb" / split / "sensor" / f"{names[0]}.png") as im:
            if im.size != (1000, 153) or im.mode != "RGB":
                raise RuntimeError(f"Invalid sensor PNG: {im.size}, {im.mode}")

    print("\nS5-U SENSOR UPDATE RGB PREPARATION PASS")
    print(f"Source accel={source_mode}; sensor input={a.sensor_accel_mode}")
    print(f"Sensor nodes={sensor_ids}; rows={sensor_rows}")
    print(f"Sensor scales={sensor_scales.tolist()}")
    print(f"Output={a.output_root.resolve()}")


if __name__ == "__main__":
    main()
