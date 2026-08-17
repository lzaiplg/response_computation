# -*- coding: utf-8 -*-
"""Prepare an image-to-image MAIN10 V2 absolute-acceleration target dataset.

Input RGB remains exactly the existing ground-motion image.
Target RGB: R/G/B = X/Y/Z absolute nodal acceleration.
All scales are estimated from the 549 training records only.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parents[2]
FIXED_NODE_IDS = list(range(91, 105)) + list(range(11284, 11292))


def portable_path(path: Path) -> str:
    """Store package-relative metadata so generated mappings survive relocation."""
    try:
        return path.resolve().relative_to(SCRIPT_DIR.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "response" / "response_npz",
    )
    p.add_argument(
        "--source-baseline-root",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "baseline_displacement",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "baseline_acceleration",
    )
    p.add_argument(
        "--source-accel-mode", choices=("auto", "relative", "absolute"), default="auto"
    )
    p.add_argument(
        "--target-accel-mode", choices=("relative", "absolute"), default="absolute"
    )
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--sample-rows-per-record", type=int, default=12000)
    p.add_argument("--detection-records", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def lines(path):
    return [
        x.strip()
        for x in path.read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def detect(npz_dir, names, max_records):
    rel = []
    ab = []
    for name in names[: max(1, max_records)]:
        with np.load(npz_dir / f"{name}.npz") as d:
            ground = np.asarray(d["input"], dtype=np.float64)
            accel = np.asarray(d["accel"], dtype=np.float64)
            ids = np.asarray(d["node_ids"]).astype(int)
        rm = {int(n): r for r, n in enumerate(ids)}
        rows = [rm[n] for n in FIXED_NODE_IDS if n in rm]
        if not rows:
            raise RuntimeError(
                "Fixed nodes unavailable for acceleration-definition detection"
            )
        fixed = accel[rows]
        gm = np.broadcast_to(ground[None, :, :], fixed.shape)
        scale = max(rms(gm), 1e-12)
        rel.append(rms(fixed) / scale)
        ab.append(rms(fixed - gm) / scale)
    rs = float(np.median(rel))
    aps = float(np.median(ab))
    mode = "relative" if rs < aps else "absolute"
    return mode, {"relative_score": rs, "absolute_score": aps}


def convert(accel, ground, source, target):
    if source == target:
        return accel.astype(np.float32, copy=False)
    if source == "relative" and target == "absolute":
        return (accel + ground[None, :, :]).astype(np.float32)
    if source == "absolute" and target == "relative":
        return (accel - ground[None, :, :]).astype(np.float32)
    raise ValueError((source, target))


def to_rgb(values, scales):
    n = np.clip(values.astype(np.float32) / scales.reshape(1, 1, 3), -1, 1)
    return np.rint((n + 1) * 127.5).astype(np.uint8)


def preview(path, inp, target, title):
    h, w = 153, 1000
    th = 25
    canvas = Image.new("RGB", (w, 2 * (h + th)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(
        (("Ground-motion input RGB", inp), ("Absolute acceleration target RGB", target))
    ):
        y = i * (h + th)
        draw.text((8, y + 5), f"{title} | {label}", fill="black")
        canvas.paste(Image.fromarray(array, mode="RGB"), (0, y + th))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    a = parse_args()
    source = a.source_baseline_root.resolve()
    out = a.output_root.resolve()
    npz = a.npz_dir.resolve()
    splits = {
        s: lines(source / "splits" / f"{s}.txt") for s in ("train", "val", "test")
    }
    if [len(splits[s]) for s in ("train", "val", "test")] != [549, 157, 78]:
        raise RuntimeError({s: len(v) for s, v in splits.items()})
    source_mode = a.source_accel_mode
    detection = None
    if source_mode == "auto":
        source_mode, detection = detect(npz, splits["train"], a.detection_records)
    print(f"NPZ acceleration definition: {source_mode}; target: {a.target_accel_mode}")
    if out.exists():
        if a.overwrite:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            raise FileExistsError(f"{out} is not empty; use --overwrite")
    (out / "dataset_rgb").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    samples = []
    for i, name in enumerate(splits["train"], 1):
        with np.load(npz / f"{name}.npz") as d:
            g = np.asarray(d["input"], dtype=np.float32)
            ac = np.asarray(d["accel"], dtype=np.float32)
        target = np.abs(convert(ac, g, source_mode, a.target_accel_mode).reshape(-1, 3))
        if len(target) > a.sample_rows_per_record:
            target = target[
                rng.choice(len(target), a.sample_rows_per_record, replace=False)
            ]
        samples.append(target)
        if i % 50 == 0 or i == len(splits["train"]):
            print(f'Acceleration scale scan {i}/{len(splits["train"])}')
    scales = np.percentile(
        np.concatenate(samples, axis=0), a.percentile, axis=0
    ).astype(np.float32)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 1e-12):
        raise RuntimeError(f"Invalid acceleration scales: {scales}")
    base_map = json.loads(
        (source / "training_rgb_mapping.json").read_text(encoding="utf-8")
    )
    reports = {}
    for split, names in splits.items():
        inp_dir = out / "dataset_rgb" / split / "input"
        tar_dir = out / "dataset_rgb" / split / "target"
        inp_dir.mkdir(parents=True, exist_ok=True)
        tar_dir.mkdir(parents=True, exist_ok=True)
        clipped = np.zeros(3, dtype=np.int64)
        total = np.zeros(3, dtype=np.int64)
        for i, name in enumerate(names, 1):
            src = source / "dataset_rgb" / split / "input" / f"{name}.png"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, inp_dir / src.name)
            with np.load(npz / f"{name}.npz") as d:
                g = np.asarray(d["input"], dtype=np.float32)
                ac = np.asarray(d["accel"], dtype=np.float32)
            target = convert(ac, g, source_mode, a.target_accel_mode)
            if target.shape != (153, 1000, 3) or np.any(~np.isfinite(target)):
                raise ValueError(f"{name}: bad target {target.shape}")
            clipped += np.sum(np.abs(target) > scales.reshape(1, 1, 3), axis=(0, 1))
            total += 153 * 1000
            target_rgb = to_rgb(target, scales)
            Image.fromarray(target_rgb, mode="RGB").save(tar_dir / f"{name}.png")
            if i <= 2:
                with Image.open(src) as im:
                    inp = np.asarray(im.convert("RGB"))
                preview(
                    out / "previews" / f"{split}_{i:02d}_{name}.png",
                    inp,
                    target_rgb,
                    name,
                )
            if i % 25 == 0 or i == len(names):
                print(f"{split}: {i}/{len(names)}")
        reports[split] = {
            "records": len(names),
            "target_clipped_fraction_xyz": (clipped / np.maximum(total, 1)).tolist(),
        }
    shutil.copytree(source / "splits", out / "splits", dirs_exist_ok=True)
    mapping = {
        "dataset_name": "formal_baseline_MAIN10_V2_ACC",
        "image_width": 1000,
        "image_height": 153,
        "rgb_channel_meaning": {
            "input_R": "H1 ground acceleration",
            "input_G": "H2 ground acceleration",
            "input_B": "vertical ground acceleration",
            "target_R": "X absolute nodal acceleration",
            "target_G": "Y absolute nodal acceleration",
            "target_B": "Z absolute nodal acceleration",
        },
        "mapping": "clip(value/scale,-1,1), RGB=round((normalized+1)*127.5)",
        "scale_source": "training split only",
        "percentile": a.percentile,
        "input_scales_m_s2": base_map["input_scales_m_s2"],
        "absolute_acceleration_scales_m_s2": scales.tolist(),
        "source_npz_acceleration_definition": source_mode,
        "target_acceleration_definition": a.target_accel_mode,
        "source_mode_detection": detection,
        "splits": reports,
        "source_npz_directory": portable_path(npz),
        "window_definition": base_map.get("window_definition", "MAIN10 V2"),
    }
    (out / "training_acc_rgb_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nMAIN10 V2 A1 ABSOLUTE-ACCELERATION RGB PREPARATION PASS")
    print(f"Scales={scales.tolist()}")
    print(f"Output={out}")


if __name__ == "__main__":
    main()
