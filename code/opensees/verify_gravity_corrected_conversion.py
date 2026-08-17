# -*- coding: utf-8 -*-
"""Verify gravity-baseline-corrected OpenSees conversion outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).resolve().parent / 'SeismicConversionOutput_FIXED'),
    )
    parser.add_argument('--node-id', type=int, default=None)
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    npz_files = sorted((root / 'response_npz').glob('*.npz'))
    if not npz_files:
        raise FileNotFoundError(f'No NPZ files found in {root / "response_npz"}')

    path = npz_files[0]
    with np.load(path) as data:
        required = {
            'time', 'disp', 'disp_total', 'gravity_disp', 'input',
            'node_ids', 'modal_frequencies_hz',
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f'{path.name} is not a gravity-corrected NPZ. Missing fields: {sorted(missing)}'
            )

        time = np.asarray(data['time'], dtype=np.float64)
        disp = np.asarray(data['disp'], dtype=np.float64)
        disp_total = np.asarray(data['disp_total'], dtype=np.float64)
        gravity_disp = np.asarray(data['gravity_disp'], dtype=np.float64)
        input_acc = np.asarray(data['input'], dtype=np.float64)
        node_ids = np.asarray(data['node_ids'])
        frequencies = np.asarray(data['modal_frequencies_hz'], dtype=np.float64)

    if time.size < 2:
        raise RuntimeError('FAILED: fewer than two time samples.')
    if disp.shape != disp_total.shape:
        raise RuntimeError(
            f'FAILED: disp shape {disp.shape} differs from disp_total shape {disp_total.shape}.'
        )
    if gravity_disp.shape != (disp.shape[0], 3):
        raise RuntimeError(
            f'FAILED: gravity_disp shape {gravity_disp.shape}; expected {(disp.shape[0], 3)}.'
        )

    dt_values = np.diff(time)
    reconstructed = disp_total - gravity_disp[:, None, :]
    reconstruction_error = float(np.max(np.abs(reconstructed - disp)))

    if args.node_id is not None:
        matches = np.where(node_ids == args.node_id)[0]
        if matches.size == 0:
            raise ValueError(f'Node {args.node_id} is not in node mapping.')
        node_row = int(matches[0])
    else:
        # Select by dynamic displacement, not by gravity-deformed total displacement.
        node_row = int(np.unravel_index(np.argmax(np.abs(disp)), disp.shape)[0])

    node_peak_flat = int(np.argmax(np.abs(disp[node_row])))
    node_peak_time_index = int(np.unravel_index(node_peak_flat, disp[node_row].shape)[0])

    report = {
        'npz_file': str(path),
        'time_steps': int(time.size),
        'first_time_s': float(time[0]),
        'last_time_s': float(time[-1]),
        'median_dt_s': float(np.median(dt_values)),
        'input_shape': list(input_acc.shape),
        'dynamic_displacement_shape': list(disp.shape),
        'selected_node_row': node_row,
        'selected_node_id': int(node_ids[node_row]),
        'selected_node_dynamic_peak_time_s': float(time[node_peak_time_index]),
        'gravity_max_abs_xyz_m': [
            float(value) for value in np.max(np.abs(gravity_disp), axis=0)
        ],
        'dynamic_max_abs_xyz_m': [
            float(value) for value in np.max(np.abs(disp), axis=(0, 1))
        ],
        'total_max_abs_xyz_m': [
            float(value) for value in np.max(np.abs(disp_total), axis=(0, 1))
        ],
        'gravity_subtraction_reconstruction_error_m': reconstruction_error,
        'modal_frequencies_hz': [float(value) for value in frequencies],
        'nan_or_inf_in_dynamic_disp': bool(np.any(~np.isfinite(disp))),
        'nan_or_inf_in_total_disp': bool(np.any(~np.isfinite(disp_total))),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not np.isclose(np.median(dt_values), 0.01, atol=1e-8):
        raise RuntimeError('FAILED: recorder time step is not 0.01 s.')
    if not np.isclose(time[-1], 10.0, atol=1e-6):
        raise RuntimeError('FAILED: final response time is not 10.0 s.')
    if np.any(~np.isfinite(disp)) or np.any(~np.isfinite(disp_total)):
        raise RuntimeError('FAILED: displacement contains NaN or Inf.')
    if reconstruction_error > 1.0e-6:
        raise RuntimeError(
            f'FAILED: gravity subtraction is inconsistent; max error={reconstruction_error:.6e} m.'
        )

    dynamic_figure = root / 'verification_dynamic_time_history.png'
    plt.figure(figsize=(11, 5))
    for direction, label in enumerate(('X', 'Y', 'Z')):
        plt.plot(time, disp[node_row, :, direction], label=label)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Dynamic displacement (m)')
    plt.title(
        f'{path.stem} | dynamic response | '
        f'node row={node_row}, node id={int(node_ids[node_row])}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(dynamic_figure, dpi=180)
    plt.close()

    comparison_figure = root / 'verification_total_vs_dynamic_z.png'
    plt.figure(figsize=(11, 5))
    plt.plot(time, disp_total[node_row, :, 2], label='Total Z')
    plt.plot(time, disp[node_row, :, 2], label='Dynamic Z')
    plt.axhline(float(gravity_disp[node_row, 2]), linestyle='--', label='Gravity Z baseline')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title(
        f'{path.stem} | gravity baseline check | '
        f'node row={node_row}, node id={int(node_ids[node_row])}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_figure, dpi=180)
    plt.close()

    print('PASS: gravity-baseline and time-axis checks passed.')
    print(f'Dynamic plot: {dynamic_figure}')
    print(f'Comparison plot: {comparison_figure}')


if __name__ == '__main__':
    main()
