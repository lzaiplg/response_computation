# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_FIELDS = {
    "input", "disp", "disp_total", "gravity_disp", "accel", "strain",
    "node_ids", "node_coordinates", "time", "modal_frequencies_hz",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "SeismicConversionOutput_FIXED",
    )
    parser.add_argument("--expected-count", type=int, default=784)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-pixel-compare", action="store_true")
    return parser.parse_args()


def scales3(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,) or np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"Invalid {name}: {array}")
    return array


def field_to_rgb(array, scales):
    normalized = np.clip(
        array.astype(np.float32) / scales.reshape(1, 1, 3),
        -1.0,
        1.0,
    )
    return np.rint((normalized + 1.0) * 127.5).astype(np.uint8)


def read_rgb(path, expected_size):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path.name}: mode={image.mode}, expected RGB")
        if image.size != expected_size:
            raise ValueError(f"{path.name}: size={image.size}, expected {expected_size}")
        return np.asarray(image, dtype=np.uint8)


def main():
    args = parse_args()
    root = args.root.resolve()
    npz_dir = root / "response_npz"
    input_dir = root / "output_nodal" / "input_images_rgb"
    response_dir = root / "output_nodal" / "response_disp_rgb"
    mapping_path = root / "rgb_mapping.json"

    for folder in (npz_dir, input_dir, response_dir):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    input_scales = scales3(mapping["input_scales_m_s2"], "input_scales_m_s2")
    disp_scales = scales3(
        mapping["dynamic_displacement_scales_m"],
        "dynamic_displacement_scales_m",
    )

    npz_files = sorted(npz_dir.glob("*.npz"))
    input_pngs = sorted(input_dir.glob("*.png"))
    response_pngs = sorted(response_dir.glob("*.png"))

    npz_names = {p.stem for p in npz_files}
    input_names = {p.stem for p in input_pngs}
    response_names = {p.stem for p in response_pngs}

    missing_input = sorted(npz_names - input_names)
    missing_response = sorted(npz_names - response_names)
    extra_input = sorted(input_names - npz_names)
    extra_response = sorted(response_names - npz_names)

    selected = npz_files
    if args.max_records is not None:
        selected = selected[: max(0, args.max_records)]
    if not selected:
        raise RuntimeError("No NPZ files selected.")

    errors = []
    rows = []
    worst_reconstruction = 0.0
    worst_pixel_diff = 0
    different_channel_values = 0
    global_disp_max = np.zeros(3, dtype=np.float64)

    reference_node_ids = None
    reference_coords = None
    reference_time = None

    for index, npz_path in enumerate(selected, start=1):
        name = npz_path.stem
        row = {"record": name, "status": "PASS", "errors": ""}
        record_errors = []

        try:
            with np.load(npz_path) as data:
                missing = REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise ValueError(f"missing fields: {sorted(missing)}")

                input_data = np.asarray(data["input"], dtype=np.float32)
                disp = np.asarray(data["disp"], dtype=np.float32)
                disp_total = np.asarray(data["disp_total"], dtype=np.float32)
                gravity = np.asarray(data["gravity_disp"], dtype=np.float32)
                accel = np.asarray(data["accel"], dtype=np.float32)
                strain = np.asarray(data["strain"], dtype=np.float32)
                node_ids = np.asarray(data["node_ids"])
                coords = np.asarray(data["node_coordinates"])
                time = np.asarray(data["time"], dtype=np.float64)

            expected_shapes = {
                "input": (1000, 3),
                "disp": (153, 1000, 3),
                "disp_total": (153, 1000, 3),
                "gravity_disp": (153, 3),
                "accel": (153, 1000, 3),
                "time": (1000,),
            }
            actual_shapes = {
                "input": input_data.shape,
                "disp": disp.shape,
                "disp_total": disp_total.shape,
                "gravity_disp": gravity.shape,
                "accel": accel.shape,
                "time": time.shape,
            }
            for key, expected in expected_shapes.items():
                if actual_shapes[key] != expected:
                    record_errors.append(
                        f"{key} shape={actual_shapes[key]}, expected={expected}"
                    )

            arrays = (input_data, disp, disp_total, gravity, accel, strain, time)
            if any(np.any(~np.isfinite(array)) for array in arrays):
                record_errors.append("NaN or Inf detected")

            dt = float(np.median(np.diff(time)))
            final_time = float(time[-1])
            if not math.isclose(dt, 0.01, rel_tol=0.0, abs_tol=1e-8):
                record_errors.append(f"median dt={dt}")
            if not math.isclose(final_time, 10.0, rel_tol=0.0, abs_tol=1e-6):
                record_errors.append(f"final time={final_time}")

            reconstruction_error = float(
                np.max(np.abs(disp - (disp_total - gravity[:, None, :])))
            )
            worst_reconstruction = max(worst_reconstruction, reconstruction_error)
            if reconstruction_error > 1e-5:
                record_errors.append(
                    f"gravity reconstruction error={reconstruction_error:.3e}"
                )

            if reference_node_ids is None:
                reference_node_ids = node_ids.copy()
                reference_coords = coords.copy()
                reference_time = time.copy()
            else:
                if not np.array_equal(reference_node_ids, node_ids):
                    record_errors.append("node_ids differ")
                if not np.allclose(reference_coords, coords, rtol=0.0, atol=1e-10):
                    record_errors.append("node coordinates differ")
                if not np.allclose(reference_time, time, rtol=0.0, atol=1e-10):
                    record_errors.append("time vector differs")

            disp_max_xyz = np.max(np.abs(disp), axis=(0, 1))
            global_disp_max = np.maximum(global_disp_max, disp_max_xyz)

            input_png = input_dir / f"{name}.png"
            response_png = response_dir / f"{name}.png"
            input_rgb = read_rgb(input_png, (1000, 153))
            response_rgb = read_rgb(response_png, (1000, 153))

            if not args.skip_pixel_compare:
                nodal_input = np.broadcast_to(
                    input_data[None, :, :], (153, 1000, 3)
                )
                expected_input = field_to_rgb(nodal_input, input_scales)
                expected_response = field_to_rgb(disp, disp_scales)

                input_diff = np.abs(
                    input_rgb.astype(np.int16) - expected_input.astype(np.int16)
                )
                response_diff = np.abs(
                    response_rgb.astype(np.int16) - expected_response.astype(np.int16)
                )

                record_pixel_max = max(
                    int(np.max(input_diff)),
                    int(np.max(response_diff)),
                )
                record_diff_values = (
                    int(np.count_nonzero(input_diff))
                    + int(np.count_nonzero(response_diff))
                )

                worst_pixel_diff = max(worst_pixel_diff, record_pixel_max)
                different_channel_values += record_diff_values

                if record_pixel_max != 0:
                    record_errors.append(
                        f"RGB max pixel difference={record_pixel_max}"
                    )

                row["rgb_different_channel_values"] = record_diff_values

            row.update({
                "final_time_s": final_time,
                "median_dt_s": dt,
                "gravity_reconstruction_error_m": reconstruction_error,
                "disp_max_x_m": float(disp_max_xyz[0]),
                "disp_max_y_m": float(disp_max_xyz[1]),
                "disp_max_z_m": float(disp_max_xyz[2]),
            })

        except Exception as exc:
            record_errors.append(str(exc))

        if record_errors:
            row["status"] = "FAIL"
            row["errors"] = " | ".join(record_errors)
            errors.append(f"{name}: {row['errors']}")

        rows.append(row)

        if index % 25 == 0 or index == len(selected):
            print(f"Validated {index}/{len(selected)}")

    count_ok = (
        len(npz_files) == args.expected_count
        and len(input_pngs) == args.expected_count
        and len(response_pngs) == args.expected_count
    )
    names_ok = not (
        missing_input or missing_response or extra_input or extra_response
    )
    full_pass = bool(count_ok and names_ok and not errors)

    report = {
        "pass": full_pass,
        "expected_count": args.expected_count,
        "npz_count": len(npz_files),
        "input_rgb_count": len(input_pngs),
        "response_rgb_count": len(response_pngs),
        "records_checked": len(selected),
        "missing_input_rgb": missing_input,
        "missing_response_rgb": missing_response,
        "extra_input_rgb": extra_input,
        "extra_response_rgb": extra_response,
        "input_scales_m_s2": input_scales.tolist(),
        "dynamic_displacement_scales_m": disp_scales.tolist(),
        "global_dynamic_disp_max_abs_xyz_m": global_disp_max.tolist(),
        "worst_gravity_reconstruction_error_m": worst_reconstruction,
        "worst_rgb_pixel_difference": worst_pixel_diff,
        "total_rgb_different_channel_values": different_channel_values,
        "failed_validation_records": len(errors),
        "errors": errors,
    }

    report_path = root / "validation_full_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = root / "validation_record_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("========== FULL VALIDATION SUMMARY ==========")
    print(f"NPZ count:          {len(npz_files)}")
    print(f"Input RGB count:    {len(input_pngs)}")
    print(f"Response RGB count: {len(response_pngs)}")
    print(f"Name matching:      {'PASS' if names_ok else 'FAIL'}")
    print(f"Records checked:    {len(selected)}")
    print(f"Validation errors:  {len(errors)}")
    print(f"Worst gravity reconstruction error: {worst_reconstruction:.3e} m")
    if args.skip_pixel_compare:
        print("RGB pixel reproduction: SKIPPED")
    else:
        print(f"Worst RGB pixel difference: {worst_pixel_diff}")
        print(f"Different RGB channel-values: {different_channel_values}")
    print(f"JSON report: {report_path}")
    print(f"CSV report:  {csv_path}")
    print("")
    if full_pass:
        print("FULL VALIDATION PASS")
    else:
        print("FULL VALIDATION FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
