# -*- coding: utf-8 -*-
"""Read-only integrity checks for the standalone pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=package)
    parser.add_argument("--require-datasets", action="store_true")
    return parser.parse_args()


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    root = args.package_root.resolve()
    failures: list[str] = []
    split_counts = {"train": 549, "val": 157, "test": 78}
    for split, expected in split_counts.items():
        path = root / "config" / "splits" / f"{split}.txt"
        actual = len(lines(path)) if path.is_file() else -1
        if actual != expected:
            failures.append(f"split {split}: {actual} != {expected}")

    npz_dir = root / "data" / "processed" / "response" / "response_npz"
    npz_files = sorted(npz_dir.glob("*.npz"))
    if len(npz_files) != 784:
        failures.append(f"response NPZ count: {len(npz_files)} != 784")
    if npz_files:
        with np.load(npz_files[0]) as data:
            required = {"input", "disp", "accel", "time", "node_ids", "node_coordinates"}
            missing = required - set(data.files)
            if missing:
                failures.append(f"NPZ missing keys: {sorted(missing)}")
            if data["input"].shape != (1000, 3) or data["disp"].shape != (153, 1000, 3) or data["accel"].shape != (153, 1000, 3):
                failures.append(f"NPZ shape mismatch in {npz_files[0].name}")

    raw_at2 = list((root / "data" / "raw" / "peer_earthquake_wav").rglob("*.AT2"))
    if len(raw_at2) < 784:
        failures.append(f"raw AT2 count unexpectedly small: {len(raw_at2)}")
    layout_path = root / "config" / "sensor_layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if len(layout.get("sensor_node_ids", [])) != 5:
            failures.append("sensor layout does not contain five sensor nodes")
    else:
        failures.append(f"missing {layout_path}")

    if args.require_datasets:
        dataset_root = root / "data" / "processed" / "sensor_update_acceleration" / "dataset_rgb"
        for split, expected in split_counts.items():
            for folder in ("ground", "sensor", "mask", "target"):
                actual = len(list((dataset_root / split / folder).glob("*.png")))
                if actual != expected:
                    failures.append(f"dataset {split}/{folder}: {actual} != {expected}")

    if failures:
        print("PREFLIGHT FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"PREFLIGHT PASS\nPackage: {root}\nResponse NPZ: {len(npz_files)}\nRaw AT2: {len(raw_at2)}\nSplit: 549/157/78")


if __name__ == "__main__":
    main()
