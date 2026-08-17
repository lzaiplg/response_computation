# -*- coding: utf-8 -*-
"""Build the 7-channel acceleration-output dataset by reusing the validated
sensor condition images and replacing only the target with full-field absolute
acceleration RGB."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[2]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-sensor-root",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "sensor_update_displacement",
    )
    p.add_argument(
        "--acc-baseline-root",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "baseline_acceleration",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "sensor_update_acceleration",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def lines(p):
    return [
        x.strip() for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()
    ]


def main():
    a = parse()
    src = a.source_sensor_root.resolve()
    acc = a.acc_baseline_root.resolve()
    out = a.output_root.resolve()
    if out.exists():
        if a.overwrite:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            raise FileExistsError(f"{out} not empty; use --overwrite")
    for split in ("train", "val", "test"):
        names = lines(acc / "splits" / f"{split}.txt")
        root = out / "dataset_rgb" / split
        for sub in ("ground", "sensor", "mask", "target"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names, 1):
            for sub in ("ground", "sensor", "mask"):
                p = src / "dataset_rgb" / split / sub / f"{name}.png"
                if not p.is_file():
                    raise FileNotFoundError(p)
                shutil.copy2(p, root / sub / p.name)
            t = acc / "dataset_rgb" / split / "target" / f"{name}.png"
            if not t.is_file():
                raise FileNotFoundError(t)
            shutil.copy2(t, root / "target" / t.name)
            if i % 50 == 0 or i == len(names):
                print(f"{split}: {i}/{len(names)}")
    shutil.copytree(acc / "splits", out / "splits", dirs_exist_ok=True)
    sm = json.loads((src / "sensor_update_mapping.json").read_text(encoding="utf-8"))
    am = json.loads((acc / "training_acc_rgb_mapping.json").read_text(encoding="utf-8"))
    sm.update(
        {
            "dataset_name": "formal_sensor_update_MAIN10_V2_ACC_S5_U_7ch",
            "target_channels": "absolute nodal acceleration RGB (R=X,G=Y,B=Z)",
            "absolute_acceleration_scales_m_s2": am[
                "absolute_acceleration_scales_m_s2"
            ],
            "target_acceleration_definition": am["target_acceleration_definition"],
            "source_npz_acceleration_definition": am[
                "source_npz_acceleration_definition"
            ],
        }
    )
    sm.pop("dynamic_displacement_scales_m", None)
    (out / "sensor_update_acc_mapping.json").write_text(
        json.dumps(sm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        names = lines(acc / "splits" / f"{split}.txt")
        for sub in ("ground", "sensor", "mask", "target"):
            files = list((out / "dataset_rgb" / split / sub).glob("*.png"))
            if len(files) != len(names):
                raise RuntimeError(f"{split}/{sub}: {len(files)} != {len(names)}")
        with Image.open(
            out / "dataset_rgb" / split / "target" / f"{names[0]}.png"
        ) as im:
            if im.size != (1000, 153) or im.mode != "RGB":
                raise RuntimeError((im.size, im.mode))
    print("MAIN10 V2 SENSOR ACCELERATION-OUTPUT DATASET PASS")
    print(f"Output={out}")


if __name__ == "__main__":
    main()
