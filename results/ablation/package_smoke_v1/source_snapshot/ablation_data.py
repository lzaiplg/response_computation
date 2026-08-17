from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_ROOT = PROJECT_ROOT / "data" / "processed" / "response" / "response_npz"
DISP_ROOT = PROJECT_ROOT / "data" / "processed" / "sensor_update_displacement"
ACC_ROOT = PROJECT_ROOT / "data" / "processed" / "sensor_update_acceleration"
DISP_MAPPING = DISP_ROOT / "sensor_update_mapping.json"
ACC_MAPPING = ACC_ROOT / "sensor_update_acc_mapping.json"
SPLIT_ROOT = PROJECT_ROOT / "config" / "splits"
TRAIN_LIST = SPLIT_ROOT / "train.txt"
VAL_LIST = SPLIT_ROOT / "val.txt"
TEST_LIST = SPLIT_ROOT / "test.txt"


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def deterministic_development_split(seed: int = 20260814, tune_count: int = 69) -> tuple[list[str], list[str]]:
    names = read_names(TRAIN_LIST)
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{seed}|{name}".encode("utf-8")).hexdigest(),
    )
    tune = sorted(ranked[:tune_count])
    train = sorted(ranked[tune_count:])
    if set(train) & set(tune):
        raise RuntimeError("Development split overlap detected")
    if set(train) | set(tune) != set(names):
        raise RuntimeError("Development split does not cover the original training split")
    return train, tune


def resolve_partition(partition: str) -> list[str]:
    train, tune = deterministic_development_split()
    if partition == "train":
        return train
    if partition == "tune":
        return tune
    if partition == "external":
        return read_names(VAL_LIST)
    if partition == "formal":
        return read_names(TEST_LIST)
    raise ValueError(partition)


def load_mapping(target: str) -> dict[str, Any]:
    path = DISP_MAPPING if target == "disp" else ACC_MAPPING
    return json.loads(path.read_text(encoding="utf-8"))


def target_scales(target: str) -> np.ndarray:
    mapping = load_mapping(target)
    key = "dynamic_displacement_scales_m" if target == "disp" else "absolute_acceleration_scales_m_s2"
    return np.asarray(mapping[key], dtype=np.float32)


def fixed_geometry() -> tuple[np.ndarray, np.ndarray]:
    first = NPZ_ROOT / f"{read_names(TRAIN_LIST)[0]}.npz"
    with np.load(first, allow_pickle=False) as data:
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        coords = np.asarray(data["node_coordinates"], dtype=np.float32)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    normalized = 2.0 * (coords - lo) / span - 1.0
    return node_ids, normalized.astype(np.float32)


class BridgeResponseDataset(Dataset):
    """Raw-float dataset. Truth is returned only as a supervised target, never as input."""

    def __init__(self, partition: str, target: str = "disp", limit: int | None = None) -> None:
        if target not in {"disp", "accel_abs"}:
            raise ValueError(target)
        self.partition = partition
        self.target_kind = target
        self.names = resolve_partition(partition)
        if limit is not None:
            self.names = self.names[: int(limit)]
        self.mapping = load_mapping(target)
        self.sensor_rows = np.asarray(self.mapping["sensor_node_rows"], dtype=np.int64)
        self.ground_scales = np.asarray(self.mapping["ground_acceleration_scales_m_s2"], dtype=np.float32)
        self.sensor_scales = np.asarray(self.mapping["sensor_acceleration_scales"], dtype=np.float32)
        self.output_scales = target_scales(target)
        self.node_ids, self.coords = fixed_geometry()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        path = NPZ_ROOT / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            ground = np.asarray(data["input"], dtype=np.float32)
            relative_accel = np.asarray(data["accel"], dtype=np.float32)
            if not np.array_equal(np.asarray(data["node_ids"], dtype=np.int64), self.node_ids):
                raise RuntimeError(f"Node ordering changed in {name}")
            absolute_accel = relative_accel + ground[None, :, :]
            if self.target_kind == "disp":
                target = np.asarray(data["disp"], dtype=np.float32)
            else:
                target = absolute_accel

        ground_norm = ground / self.ground_scales.reshape(1, 3)
        sensor_norm = absolute_accel[self.sensor_rows] / self.sensor_scales.reshape(1, 1, 3)
        target_norm = target / self.output_scales.reshape(1, 1, 3)
        return {
            "name": name,
            "ground": torch.from_numpy(ground_norm.T.copy()),
            "sensor": torch.from_numpy(sensor_norm.transpose(0, 2, 1).copy()),
            "target": torch.from_numpy(target_norm.transpose(2, 0, 1).copy()),
        }


def dataset_manifest() -> dict[str, Any]:
    train, tune = deterministic_development_split()
    external = read_names(VAL_LIST)
    node_ids, coords = fixed_geometry()
    return {
        "npz_root": str(NPZ_ROOT.resolve()),
        "train_count": len(train),
        "tune_count": len(tune),
        "external_count": len(external),
        "formal_test_available": True,
        "formal_test_count": len(read_names(TEST_LIST)),
        "node_count": int(node_ids.size),
        "coordinate_shape": list(coords.shape),
        "split_seed": 20260814,
        "split_rule": "SHA256 rank within original 549 training records; 69 tune and 480 train",
        "external_warning": "The 157 records were used by historical models and are not a pristine final test.",
    }
