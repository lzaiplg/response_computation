from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def count_files(path: Path, pattern: str) -> int:
    return sum(item.is_file() for item in path.rglob(pattern)) if path.is_dir() else 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def main() -> None:
    split_root = PACKAGE_ROOT / "config" / "splits"
    response_root = PACKAGE_ROOT / "data" / "processed" / "response"
    raw_root = PACKAGE_ROOT / "data" / "raw" / "peer_earthquake_wav"
    payload: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "package_root_rule": "All paths are relative to the package root.",
        "records": {
            name: sum(
                1
                for line in (split_root / f"{name}.txt")
                .read_text(encoding="utf-8-sig")
                .splitlines()
                if line.strip()
            )
            for name in ("train", "val", "test")
        },
        "expected_response_records": 784,
        "expected_shapes": {
            "response_npz": [153, 1000, 3],
            "rgb_input": [153, 1000, 3],
            "rgb_target": [153, 1000, 3],
        },
        "paths": {
            "raw_earthquake": relative(raw_root),
            "response_npz": relative(response_root / "response_npz"),
            "splits": relative(split_root),
        },
        "observed_files": {
            "raw_files": count_files(raw_root, "*") if raw_root.is_dir() else 0,
            "raw_at2_files": sum(
                item.is_file() and item.suffix.lower() == ".at2"
                for item in raw_root.rglob("*")
            )
            if raw_root.is_dir()
            else 0,
            "response_npz_files": count_files(response_root / "response_npz", "*.npz"),
        },
        "split_sha256": {
            name: sha256(split_root / f"{name}.txt")
            for name in ("train", "val", "test")
        },
        "note": "Large raw/NPZ/RGB artifacts are intentionally ignored by Git; copy or provision them separately.",
    }
    output = PACKAGE_ROOT / "data" / "DATA_MANIFEST.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
