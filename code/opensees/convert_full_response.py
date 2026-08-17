
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:57:51 2026

@author: Gemini
"""

import os
import re
import numpy as np
from PIL import Image
import openseespy.opensees as ops
import logging

import math
import contextlib
import io
import time
import argparse
import csv
import json
import shutil
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASE_WAV_DIR = str(PACKAGE_ROOT / 'data' / 'raw' / 'peer_earthquake_wav')
OUTPUT_DIR = str(PACKAGE_ROOT / 'data' / 'processed' / 'response')

# --- Nodal Output Directories ---
NODAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'output_nodal')
NODAL_IMG_INPUT_DIR = os.path.join(NODAL_OUTPUT_DIR, 'input_images')
NODAL_IMG_DISP_DIR = os.path.join(NODAL_OUTPUT_DIR, 'response_disp')
NODAL_IMG_ACCEL_DIR = os.path.join(NODAL_OUTPUT_DIR, 'response_accel')

# --- Elemental Output Directories ---
ELEMENTAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'output_elemental')
ELEMENTAL_IMG_INPUT_DIR = os.path.join(ELEMENTAL_OUTPUT_DIR, 'input_images')
ELEMENTAL_IMG_STRAIN_DIR = os.path.join(ELEMENTAL_OUTPUT_DIR, 'response_strain')

TCL_DIR = str(PACKAGE_ROOT / 'code' / 'opensees' / 'TCL')
RESPONSE_DATA_DIR = os.path.join(OUTPUT_DIR, 'response_data')

# Create output directories
os.makedirs(NODAL_IMG_INPUT_DIR, exist_ok=True)
os.makedirs(NODAL_IMG_DISP_DIR, exist_ok=True)
os.makedirs(NODAL_IMG_ACCEL_DIR, exist_ok=True)
os.makedirs(ELEMENTAL_IMG_INPUT_DIR, exist_ok=True)
os.makedirs(ELEMENTAL_IMG_STRAIN_DIR, exist_ok=True)
os.makedirs(RESPONSE_DATA_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'conversion_log.md')

# Analysis Parameters
TARGET_DT = 0.01
TARGET_DURATION = 10.0
G_ACCEL = 9.80665
TARGET_PGA_G = 0.3  # common scale: max horizontal PGA -> 0.3 g
GRAVITY_ACCEL = 9.81
GRAVITY_STEPS = 10
GRAVITY_TOL = 1.0e-6
GRAVITY_MAX_ITERS = 100
DYNAMIC_TOL = 1.0e-7
DYNAMIC_MAX_ITERS = 50
DAMPING_RATIO = 0.05
DAMPING_MODE_I = 1
DAMPING_MODE_J = 2
NUM_MODES_TO_REPORT = 10

TCL_FILES = [
    "0_material.tcl", "1_node.tcl", "2_Section.tcl", "3_fiber.tcl",
    "4_Element.tcl", "5_Fix.tcl", "6_mass_load.tcl",
]

# --- Logger Setup ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Logging is configured after command-line paths are applied in ``main``.

# --- TclInterpreter Class (from reference code) ---
class TclInterpreter:
    def __init__(self):
        self.vars = {}
        self.nonlinear_elements = []
    def _resolve(self, text):
        def _sub(m):
            v = self.vars.get(m.group(1))
            return str(v) if v is not None else m.group(0)
        return re.sub(r'\$(\w+)', _sub, str(text))
    def _eval_expr(self, expr):
        expr = self._resolve(expr)
        expr = re.sub(r'\bsqrt\b', 'math.sqrt', expr)
        expr = re.sub(r'\bpow\b', 'math.pow', expr)
        expr = re.sub(r'\bacos\b', 'math.acos', expr)
        expr = re.sub(r'\babs\b', 'abs', expr)
        try:
            return eval(expr, {'math': math, '__builtins__': {}})
        except Exception:
            return 0.0
    def _process(self, token):
        MAX_ITER = 10
        for _ in range(MAX_ITER):
            m = re.search(r'\[expr\s+([^\[\]]+)\]', token)
            if m:
                val = self._eval_expr(m.group(1))
                token = token[:m.start()] + str(val) + token[m.end():]
                continue
            m = re.search(r'\[lindex\s+\$(\w+)\s+(\d+)\]', token)
            if m:
                lst = self.vars.get(m.group(1), [])
                idx = int(m.group(2))
                val = lst[idx] if isinstance(lst, list) and idx < len(lst) else 0.0
                token = token[:m.start()] + str(val) + token[m.end():]
                continue
            break
        return self._resolve(token)
    def _num(self, token):
        return float(self._process(token))
    def _int(self, token):
        return int(self._num(token))
    def _preprocess(self, content):
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        commands = []
        current = []
        depth = 0
        i = 0
        while i < len(content):
            ch = content[i]
            if ch == '#' and not depth:
                while i < len(content) and content[i] != '\n':
                    i += 1
                i += 1
                continue
            if ch == '{':
                depth += 1
                current.append(ch)
            elif ch == '}':
                depth -= 1
                current.append(ch)
                if depth < 0:
                    depth = 0
            elif ch == ';' and depth == 0:
                cmd = ''.join(current).strip()
                if cmd:
                    commands.append(cmd)
                current = []
            elif ch == '\n' and depth == 0:
                cmd = ''.join(current).strip()
                if cmd:
                    commands.append(cmd)
                current = []
            else:
                current.append(ch)
            i += 1
        cmd = ''.join(current).strip()
        if cmd:
            commands.append(cmd)
        cleaned = []
        for cmd in commands:
            depth2 = 0
            out = []
            j = 0
            while j < len(cmd):
                c = cmd[j]
                if c == '{':
                    depth2 += 1
                    out.append(c)
                elif c == '}':
                    depth2 -= 1
                    out.append(c)
                elif c in (';', '#') and depth2 == 0:
                    break
                else:
                    out.append(c)
                j += 1
            s = ''.join(out).strip()
            if s:
                cleaned.append(s)
        return cleaned
    def _tokenize(self, cmd):
        tokens = []
        current = []
        depth = 0
        in_q = False
        for ch in cmd:
            if in_q:
                if ch == '"':
                    in_q = False
                    tokens.append(''.join(current))
                    current = []
                else:
                    current.append(ch)
            elif ch == '"':
                in_q = True
            elif ch == '{':
                if depth > 0:
                    current.append(ch)
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    tokens.append(''.join(current))
                    current = []
                else:
                    current.append(ch)
            elif ch in (' ', '\t', '\n') and depth == 0:
                s = ''.join(current).strip()
                if s:
                    tokens.append(s)
                current = []
            else:
                current.append(ch)
        s = ''.join(current).strip()
        if s:
            tokens.append(s)
        return [t for t in tokens if t]
    def _execute(self, cmd):
        cmd = cmd.strip()
        if not cmd or cmd.startswith('#'):
            return
        tokens = self._tokenize(cmd)
        if not tokens:
            return
        kw = tokens[0].lower()
        if kw == 'model':
            ops.model('BasicBuilder', '-ndm', 3, '-ndf', 6)
        elif kw == 'set':
            if len(tokens) < 3:
                return
            name = tokens[1]
            val_raw = ' '.join(tokens[2:])
            val_str = self._process(val_raw)
            try:
                self.vars[name] = float(val_str)
            except ValueError:
                self.vars[name] = val_str
        elif kw == 'node':
            nid = self._int(tokens[1])
            x, y, z = self._num(tokens[2]), self._num(tokens[3]), self._num(tokens[4])
            ops.node(nid, x, y, z)
        elif kw == 'mass':
            nid = self._int(tokens[1])
            vals = [self._num(t) for t in tokens[2:]]
            ops.mass(nid, *vals)
        elif kw == 'fix':
            nid = self._int(tokens[1])
            dofs = [int(self._num(t)) for t in tokens[2:]]
            ops.fix(nid, *dofs)
        elif kw == 'uniaxialmaterial':
            mat_type = tokens[1]
            mat_id = self._int(tokens[2])
            args = [self._num(t) for t in tokens[3:]]
            ops.uniaxialMaterial(mat_type, mat_id, *args)
        elif kw == 'geomtransf':
            gt_type = tokens[1]
            gt_id = self._int(tokens[2])
            args = [self._num(t) for t in tokens[3:]]
            ops.geomTransf(gt_type, gt_id, *args)
        elif kw == 'element':
            self._element(tokens)
        elif kw == 'section':
            self._section(tokens)
        elif kw == 'patch':
            sub = tokens[1]
            nums = [self._num(t) for t in tokens[2:]]
            if sub == 'rect':
                ops.patch(sub, int(nums[0]), int(nums[1]), int(nums[2]), nums[3], nums[4], nums[5], nums[6])
        elif kw == 'layer':
            sub = tokens[1]
            nums = [self._num(t) for t in tokens[2:]]
            if sub == 'straight':
                ops.layer(sub, int(nums[0]), int(nums[1]), nums[2], nums[3], nums[4], nums[5], nums[6])
        elif kw == 'rigidlink':
            link_type = tokens[1]
            n1 = self._int(tokens[2])
            n2 = self._int(tokens[3])
            ops.rigidLink(link_type, n1, n2)
        # Model-only parser: analysis/load commands in TCL are intentionally ignored.
        # Gravity and transient analysis are configured explicitly in Python below.
        elif kw in {
            'system', 'numberer', 'constraints', 'test', 'algorithm',
            'integrator', 'analysis', 'analyze', 'loadconst', 'wipeanalysis',
            'pattern', 'load', 'timeseries', 'eigen', 'puts', 'for',
            'lappend', 'source'
        }:
            pass
    def _element(self, tokens):
        etype = tokens[1]
        eid = self._int(tokens[2])
        el = etype.lower()
        if el == 'elasticbeamcolumn':
            iN, jN, A, E, G, J, Iy, Iz, trT = [self._num(t) for t in tokens[3:12]]
            ops.element('elasticBeamColumn', eid, int(iN), int(jN), A, E, G, J, Iy, Iz, int(trT))
        elif el == 'nonlinearbeamcolumn':
            iN, jN, nIP, secT, trT = [self._int(t) for t in tokens[3:8]]
            self.nonlinear_elements.append(eid) # Record nonlinear element ID
            extra_args = []
            k = 8
            while k < len(tokens):
                if tokens[k] == '-iter':
                    extra_args += ['-iter', self._int(tokens[k+1]), self._num(tokens[k+2])]
                    k += 3
                else: k += 1
            ops.element('nonlinearBeamColumn', eid, iN, jN, nIP, secT, trT, *extra_args)
        elif el == 'zerolength':
            iN, jN = self._int(tokens[3]), self._int(tokens[4])
            mats, dirs = [], []
            k, mode = 5, None
            while k < len(tokens):
                t = tokens[k]
                if t == '-mat': mode = 'mat'
                elif t == '-dir': mode = 'dir'
                elif mode == 'mat': mats.append(self._int(t))
                elif mode == 'dir': dirs.append(self._int(t))
                k += 1
            ops.element('zeroLength', eid, iN, jN, '-mat', *mats, '-dir', *dirs)
    def _section(self, tokens):
        if tokens[1].lower() != 'fiber': return
        sec_id = self._int(tokens[2])
        gj, block = 0.0, ''
        k = 3
        while k < len(tokens):
            if tokens[k] == '-GJ':
                gj = self._num(tokens[k+1])
                k += 2
            elif any(sub in tokens[k] for sub in ['patch', 'layer', 'fiber']):
                block = tokens[k]
                k += 1
            else: k += 1
        ops.section('Fiber', sec_id, '-GJ', gj)
        if block:
            for sc in self._preprocess(block): self._execute(sc)
    def parse_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        for cmd in self._preprocess(content):
            try:
                self._execute(cmd)
            except Exception as e:
                logging.error(f"Failed to execute TCL command: '{cmd}'. Error: {e}", exc_info=True)

# --- Data Loading ---
def get_seismic_records(directory):
    """
    Scans a directory for .AT2 files, groups them by RSN, and identifies components.
    It intelligently assigns horizontal components (H1, H2) based on common naming conventions.
    """
    logging.info(f"Scanning for seismic records in: {directory}")
    all_files = [f for f in os.listdir(directory) if f.upper().endswith('.AT2')]
    
    grouped_by_rsn = {}
    for filename in all_files:
        rsn_match = re.match(r"^(RSN\d+)_", filename, re.IGNORECASE)
        if not rsn_match:
            rsn_match = re.match(r"^([a-zA-Z0-9.-]+)_", filename)

        if rsn_match:
            rsn = rsn_match.group(1)
            if rsn not in grouped_by_rsn:
                grouped_by_rsn[rsn] = []
            grouped_by_rsn[rsn].append(filename)
        else:
            logging.warning(f"Could not determine RSN for file: {filename}. Skipping.")

    records = []
    for rsn, files in grouped_by_rsn.items():
        if len(files) < 3:
            logging.warning(f"RSN {rsn} has fewer than 3 components ({len(files)} found). Skipping.")
            continue

        # --- Vertical Component Identification ---
        vertical_file = None
        for f in files:
            f_upper = f.upper()
            # Check for common vertical component suffix features
            if ('UP.AT2' in f_upper or
                'DWN.AT2' in f_upper or
                'DOWN.AT2' in f_upper or
                'UD.AT2' in f_upper or
                'V.AT2' in f_upper or
                'V1.AT2' in f_upper or
                'V2.AT2' in f_upper or
                'VERT.AT2' in f_upper or
                'VRT.AT2' in f_upper or
                'VTC.AT2' in f_upper):
                vertical_file = f
                break
        
        if not vertical_file:
            logging.warning(f"Could not identify vertical component for RSN {rsn}. Skipping.")
            continue

        # --- Horizontal Component Identification ---
        horizontal_files = [f for f in files if f != vertical_file]
        if len(horizontal_files) < 2:
            logging.warning(f"RSN {rsn} has a vertical component but fewer than 2 horizontal. Skipping.")
            continue

        h1_file, h2_file = None, None
        
        # Define pairing rules for horizontal components
        # List of tuples, where each tuple is a pair of identifiers (e.g., ('000', '090'))
        direction_pairs = [
            ('000', '090'), ('N', 'E'), ('NS', 'EW'), 
            ('H1', 'H2'), ('LONG', 'TRANS')
        ]
        
        f_upper = [f.upper() for f in horizontal_files]

        # Try to find a match based on direction pairs
        for p1, p2 in direction_pairs:
            f1_match, f2_match = None, None
            for i, fname_upper in enumerate(f_upper):
                if p1 in fname_upper:
                    f1_match = horizontal_files[i]
                if p2 in fname_upper:
                    f2_match = horizontal_files[i]
            
            if f1_match and f2_match and f1_match != f2_match:
                h1_file, h2_file = f1_match, f2_match
                logging.info(f"For RSN {rsn}, identified horizontal pair based on '{p1}'/'{p2}'.")
                break
        
        # Fallback to alphabetical sorting if no specific pair is found
        if not h1_file or not h2_file:
            horizontal_files.sort()
            h1_file, h2_file = horizontal_files[0], horizontal_files[1]
            logging.warning(
                f"For RSN {rsn}, could not definitively identify horizontal component pairs. "
                f"Falling back to alphabetical sort: H1='{h1_file}', H2='{h2_file}'. "
                "Verify this order matches the intended model axes."
            )

        record_name = rsn.replace('.', '_')
        records.append({
            "name": record_name,
            "H1_file": os.path.join(directory, h1_file),
            "H2_file": os.path.join(directory, h2_file),
            "V_file": os.path.join(directory, vertical_file),
        })
        logging.info(f"Successfully grouped RSN {rsn}: H1='{h1_file}', H2='{h2_file}', V='{vertical_file}'")

    logging.info(f"Found {len(records)} complete 3-component seismic records.")
    return records

def read_at2_file(filepath, target_dt=TARGET_DT, target_duration=TARGET_DURATION):
    """Read a PEER AT2 component and return a fixed-length array in original AT2 units.

    PEER acceleration ordinates are normally in g. This function does not scale each
    component separately. A single common factor is applied later to H1/H2/V together.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        header_index = None
        dt = None
        npts = None
        for index, line in enumerate(lines[:20]):
            dt_match = re.search(r'DT\s*=\s*([\d.Ee+\-]+)', line, re.IGNORECASE)
            npts_match = re.search(r'NPTS\s*=\s*(\d+)', line, re.IGNORECASE)
            if dt_match and npts_match:
                header_index = index
                dt = float(dt_match.group(1))
                npts = int(npts_match.group(1))
                break
        if header_index is None or dt is None or npts is None:
            raise ValueError('Could not find DT and NPTS in AT2 header.')
        if dt <= 0 or npts <= 0:
            raise ValueError(f'Invalid AT2 header: DT={dt}, NPTS={npts}')

        values = [
            float(value)
            for line in lines[header_index + 1:]
            for value in line.split()
        ]
        accel = np.asarray(values, dtype=np.float64)
        if accel.size == 0:
            raise ValueError('No acceleration values found.')
        if accel.size != npts:
            logging.warning(
                'File %s: header NPTS=%d but found %d values; using actual count.',
                os.path.basename(filepath), npts, accel.size,
            )
            npts = accel.size

        final_npts = int(round(target_duration / target_dt))
        original_time = np.arange(npts, dtype=np.float64) * dt
        target_time = np.arange(final_npts, dtype=np.float64) * target_dt

        # np.interp right=0.0 performs zero padding after the original record ends.
        accel_resampled = np.interp(
            target_time,
            original_time,
            accel,
            left=float(accel[0]),
            right=0.0,
        )
        return accel_resampled
    except Exception as exc:
        logging.error('Failed to read AT2 file %s: %s', filepath, exc)
        return None


def prepare_three_components(h1_file, h2_file, v_file):
    """Read three components and preserve their original amplitude ratios.

    A single scale factor is based on the larger horizontal-component PGA so that
    max(PGA_H1, PGA_H2) becomes TARGET_PGA_G * g. The same factor is applied to V.
    """
    h1 = read_at2_file(h1_file)
    h2 = read_at2_file(h2_file)
    vertical = read_at2_file(v_file)
    if h1 is None or h2 is None or vertical is None:
        return None

    horizontal_reference = max(float(np.max(np.abs(h1))), float(np.max(np.abs(h2))))
    if horizontal_reference <= 1.0e-12:
        logging.error('Both horizontal components have near-zero PGA.')
        return None

    common_factor = TARGET_PGA_G * G_ACCEL / horizontal_reference
    h1_si = h1 * common_factor
    h2_si = h2 * common_factor
    v_si = vertical * common_factor
    metadata = {
        'common_scale_factor': common_factor,
        'raw_h1_pga': float(np.max(np.abs(h1))),
        'raw_h2_pga': float(np.max(np.abs(h2))),
        'raw_v_pga': float(np.max(np.abs(vertical))),
        'scaled_h1_pga_m_s2': float(np.max(np.abs(h1_si))),
        'scaled_h2_pga_m_s2': float(np.max(np.abs(h2_si))),
        'scaled_v_pga_m_s2': float(np.max(np.abs(v_si))),
    }
    return h1_si, h2_si, v_si, metadata


# --- OpenSees Analysis ---
@contextlib.contextmanager
def suppress_opensees_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            yield
        finally:
            pass

def _sorted_nodes_and_elements(interpreter):
    all_nodes_raw = [int(node_id) for node_id in ops.getNodeTags()]
    node_coords = {node_id: ops.nodeCoord(node_id) for node_id in all_nodes_raw}
    all_nodes = sorted(
        all_nodes_raw,
        key=lambda node_id: (
            node_coords[node_id][0],
            node_coords[node_id][1],
            node_coords[node_id][2],
        ),
    )

    all_elements_raw = [int(element_id) for element_id in interpreter.nonlinear_elements]
    element_centers = {}
    for element_id in all_elements_raw:
        connected_nodes = ops.eleNodes(element_id)
        coordinates = [ops.nodeCoord(node_id) for node_id in connected_nodes]
        element_centers[element_id] = np.mean(coordinates, axis=0)
    all_elements = sorted(
        all_elements_raw,
        key=lambda element_id: (
            element_centers[element_id][0],
            element_centers[element_id][1],
            element_centers[element_id][2],
        ),
    )
    return all_nodes, node_coords, all_elements, element_centers


def _apply_gravity(all_nodes):
    """Apply gravity, freeze it, and return the gravity-equilibrium nodal displacement.

    The returned array has shape ``(num_nodes, 3)`` and contains the absolute
    translational displacement after the static gravity analysis.  Dynamic
    displacement saved for surrogate-model training is later calculated as

        dynamic displacement = total displacement - gravity displacement.
    """
    ops.wipeAnalysis()
    gravity_series_tag = 90001
    gravity_pattern_tag = 90001
    ops.timeSeries('Constant', gravity_series_tag)
    ops.pattern('Plain', gravity_pattern_tag, gravity_series_tag)

    loaded_nodes = 0
    total_mass_z = 0.0
    for node_id in all_nodes:
        try:
            mass_z = float(ops.nodeMass(node_id, 3))
        except Exception:
            mass_z = 0.0
        if abs(mass_z) > 0.0:
            ops.load(node_id, 0.0, 0.0, -mass_z * GRAVITY_ACCEL, 0.0, 0.0, 0.0)
            loaded_nodes += 1
            total_mass_z += mass_z

    logging.info(
        '  Gravity loads created from nodal masses: nodes=%d, total translational mass=%.6e kg',
        loaded_nodes, total_mass_z,
    )
    if loaded_nodes == 0:
        raise RuntimeError('No nodal masses were found; gravity analysis cannot be performed.')

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('UmfPack')
    ops.test('NormDispIncr', GRAVITY_TOL, GRAVITY_MAX_ITERS, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0 / GRAVITY_STEPS)
    ops.analysis('Static')

    ok = ops.analyze(GRAVITY_STEPS)
    if ok != 0:
        raise RuntimeError(f'Gravity analysis failed with OpenSees status {ok}.')

    # Freeze the converged gravity state and reset analysis time to zero.
    ops.loadConst('-time', 0.0)

    gravity_disp = np.asarray(
        [
            [
                float(ops.nodeDisp(node_id, 1)),
                float(ops.nodeDisp(node_id, 2)),
                float(ops.nodeDisp(node_id, 3)),
            ]
            for node_id in all_nodes
        ],
        dtype=np.float64,
    )
    if gravity_disp.shape != (len(all_nodes), 3):
        raise RuntimeError(
            f'Unexpected gravity displacement shape: {gravity_disp.shape}; '
            f'expected {(len(all_nodes), 3)}.'
        )
    if np.any(~np.isfinite(gravity_disp)):
        raise RuntimeError('Gravity-equilibrium displacement contains NaN or Inf.')

    gravity_max_by_direction = np.max(np.abs(gravity_disp), axis=0)
    logging.info(
        '  Maximum absolute gravity displacement: X=%.6e m, Y=%.6e m, Z=%.6e m',
        gravity_max_by_direction[0],
        gravity_max_by_direction[1],
        gravity_max_by_direction[2],
    )

    ops.wipeAnalysis()
    logging.info('  Gravity analysis completed, loadConst applied, and baseline displacement captured.')
    return gravity_disp


def _configure_damping():
    """Calculate modal frequencies and apply 5% Rayleigh damping."""
    eigenvalues = ops.eigen(NUM_MODES_TO_REPORT)
    if eigenvalues is None or len(eigenvalues) < max(DAMPING_MODE_I, DAMPING_MODE_J):
        raise RuntimeError('Eigenvalue analysis failed or returned too few modes.')

    frequencies = []
    for value in eigenvalues:
        if value > 0.0:
            frequencies.append(math.sqrt(value) / (2.0 * math.pi))
        else:
            frequencies.append(float('nan'))
    logging.info('  Modal frequencies (Hz): %s', ', '.join(f'{value:.6g}' for value in frequencies))

    mode_i_value = eigenvalues[DAMPING_MODE_I - 1]
    mode_j_value = eigenvalues[DAMPING_MODE_J - 1]
    if mode_i_value <= 0.0 or mode_j_value <= 0.0:
        raise RuntimeError('Selected damping modes have non-positive eigenvalues.')

    omega_i = math.sqrt(mode_i_value)
    omega_j = math.sqrt(mode_j_value)
    alpha_m = DAMPING_RATIO * 2.0 * omega_i * omega_j / (omega_i + omega_j)
    beta_k_init = DAMPING_RATIO * 2.0 / (omega_i + omega_j)

    # Use initial stiffness proportional damping for nonlinear response.
    ops.rayleigh(alpha_m, 0.0, beta_k_init, 0.0)
    logging.info(
        '  Rayleigh damping: modes %d/%d, alphaM=%.6e, betaKinit=%.6e',
        DAMPING_MODE_I, DAMPING_MODE_J, alpha_m, beta_k_init,
    )
    if frequencies[DAMPING_MODE_I - 1] < 0.1 or frequencies[DAMPING_MODE_J - 1] < 0.1:
        logging.warning(
            '  One selected damping frequency is below 0.1 Hz. Check whether it is a bearing/rigid-body mode.'
        )
    return frequencies


def _configure_transient_analysis():
    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('UmfPack')
    ops.test('NormDispIncr', DYNAMIC_TOL, DYNAMIC_MAX_ITERS, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')


def _analyze_one_step(dt):
    """Run one transient step with conservative fallback algorithms."""
    ok = ops.analyze(1, dt)
    if ok == 0:
        return 0

    fallback_algorithms = [
        ('NewtonLineSearch',),
        ('ModifiedNewton', '-initial'),
        ('KrylovNewton',),
    ]
    for algorithm_args in fallback_algorithms:
        ops.test('NormDispIncr', 1.0e-6, 100, 0)
        ops.algorithm(*algorithm_args)
        ok = ops.analyze(1, dt)
        if ok == 0:
            ops.test('NormDispIncr', DYNAMIC_TOL, DYNAMIC_MAX_ITERS, 0)
            ops.algorithm('Newton')
            return 0

    ops.test('NormDispIncr', DYNAMIC_TOL, DYNAMIC_MAX_ITERS, 0)
    ops.algorithm('Newton')
    return ok


def _load_and_align_recorder(path, target_times, expected_min_columns=4):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < expected_min_columns:
        raise ValueError(f'Unexpected recorder shape {data.shape} in {path}')

    recorder_times = data[:, 0]
    values = data[:, 1:]
    if np.any(~np.isfinite(data)):
        raise ValueError(f'NaN or Inf found in {path}')

    aligned = np.empty((target_times.size, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        aligned[:, column] = np.interp(
            target_times,
            recorder_times,
            values[:, column],
            left=float(values[0, column]),
            right=float(values[-1, column]),
        )
    return aligned


def run_opensees_analysis(acc_h1, acc_h2, acc_v, record_name):
    """Build model, initialize gravity, and run a true Newmark transient analysis."""
    logging.info('  Setting up OpenSees model...')
    ops.wipe()

    interpreter = TclInterpreter()
    for tcl_file in TCL_FILES:
        interpreter.parse_file(os.path.join(TCL_DIR, tcl_file))

    all_nodes, node_coords, all_elements, element_centers = _sorted_nodes_and_elements(interpreter)
    num_nodes = len(all_nodes)
    num_elements = len(all_elements)
    num_steps = len(acc_h1)
    if len(acc_h2) != num_steps or len(acc_v) != num_steps:
        raise ValueError('Three acceleration components have different lengths.')

    logging.info(
        '  Model built: nodes=%d, nonlinear elements=%d, transient steps=%d, dt=%.5f s',
        num_nodes, num_elements, num_steps, TARGET_DT,
    )

    gravity_disp = _apply_gravity(all_nodes)
    frequencies = _configure_damping()

    recorder_dir = os.path.join(RESPONSE_DATA_DIR, record_name)
    if os.path.isdir(recorder_dir):
        shutil.rmtree(recorder_dir)
    os.makedirs(recorder_dir, exist_ok=True)

    for node_id in all_nodes:
        ops.recorder(
            'Node', '-file', os.path.join(recorder_dir, f'node_{node_id}_disp.out'),
            '-time', '-node', node_id, '-dof', 1, 2, 3, 'disp',
        )
        ops.recorder(
            'Node', '-file', os.path.join(recorder_dir, f'node_{node_id}_accel.out'),
            '-time', '-node', node_id, '-dof', 1, 2, 3, 'accel',
        )
    for element_id in all_elements:
        ops.recorder(
            'Element', '-file', os.path.join(recorder_dir, f'ele_{element_id}_section_deformation.out'),
            '-time', '-ele', element_id, 'section', 'deformation',
        )

    ops.timeSeries('Path', 91001, '-dt', TARGET_DT, '-values', *acc_h1, '-factor', 1.0)
    ops.timeSeries('Path', 91002, '-dt', TARGET_DT, '-values', *acc_h2, '-factor', 1.0)
    ops.timeSeries('Path', 91003, '-dt', TARGET_DT, '-values', *acc_v, '-factor', 1.0)
    ops.pattern('UniformExcitation', 91001, 1, '-accel', 91001)
    ops.pattern('UniformExcitation', 91002, 2, '-accel', 91002)
    ops.pattern('UniformExcitation', 91003, 3, '-accel', 91003)

    _configure_transient_analysis()
    start_time = time.perf_counter()
    for step in range(num_steps):
        ok = _analyze_one_step(TARGET_DT)
        if ok != 0:
            current_time = ops.getTime()
            logging.error(
                '  Transient analysis failed: record=%s, step=%d/%d, time=%.6f s, status=%s',
                record_name, step + 1, num_steps, current_time, ok,
            )
            ops.remove('recorders')
            return None
    duration = time.perf_counter() - start_time
    final_time = float(ops.getTime())
    ops.remove('recorders')
    logging.info('  Transient analysis successful: final time=%.6f s, runtime=%.2f s', final_time, duration)

    expected_final_time = num_steps * TARGET_DT
    if not math.isclose(final_time, expected_final_time, rel_tol=0.0, abs_tol=TARGET_DT * 0.1):
        logging.warning(
            '  Final OpenSees time %.6f differs from expected %.6f s.',
            final_time, expected_final_time,
        )

    target_times = np.arange(1, num_steps + 1, dtype=np.float64) * TARGET_DT
    # Recorder displacement is absolute and therefore still contains the static
    # gravity deformation.  Preserve it as disp_total, but use the gravity-
    # subtracted displacement as disp for surrogate-model training.
    all_disps_total = np.zeros((num_nodes, num_steps, 3), dtype=np.float32)
    all_disps = np.zeros((num_nodes, num_steps, 3), dtype=np.float32)
    all_accels = np.zeros((num_nodes, num_steps, 3), dtype=np.float32)
    all_strains = np.zeros((num_elements, num_steps, 3), dtype=np.float32)

    for node_index, node_id in enumerate(all_nodes):
        try:
            displacement = _load_and_align_recorder(
                os.path.join(recorder_dir, f'node_{node_id}_disp.out'), target_times, 4
            )
            acceleration = _load_and_align_recorder(
                os.path.join(recorder_dir, f'node_{node_id}_accel.out'), target_times, 4
            )
            total_displacement = displacement[:, :3]
            dynamic_displacement = total_displacement - gravity_disp[node_index][None, :]

            all_disps_total[node_index] = total_displacement.astype(np.float32)
            all_disps[node_index] = dynamic_displacement.astype(np.float32)
            all_accels[node_index] = acceleration[:, :3].astype(np.float32)
        except Exception as exc:
            logging.error('  Failed reading node %s recorders: %s', node_id, exc)
            return None

    for element_index, element_id in enumerate(all_elements):
        try:
            deformation = _load_and_align_recorder(
                os.path.join(recorder_dir, f'ele_{element_id}_section_deformation.out'),
                target_times,
                2,
            )
            components = min(3, deformation.shape[1])
            all_strains[element_index, :, :components] = deformation[:, :components]
        except Exception as exc:
            logging.error('  Failed reading element %s recorder: %s', element_id, exc)
            return None

    if np.any(~np.isfinite(all_disps)):
        raise RuntimeError('Gravity-subtracted dynamic displacement contains NaN or Inf.')
    if np.any(~np.isfinite(all_disps_total)):
        raise RuntimeError('Total displacement contains NaN or Inf.')

    reconstruction_error = float(
        np.max(
            np.abs(
                all_disps_total.astype(np.float64)
                - gravity_disp[:, None, :]
                - all_disps.astype(np.float64)
            )
        )
    )
    logging.info(
        '  Gravity-baseline subtraction check: max reconstruction error=%.6e m',
        reconstruction_error,
    )

    dynamic_max_by_direction = np.max(np.abs(all_disps), axis=(0, 1))
    logging.info(
        '  Maximum absolute dynamic displacement: X=%.6e m, Y=%.6e m, Z=%.6e m',
        dynamic_max_by_direction[0],
        dynamic_max_by_direction[1],
        dynamic_max_by_direction[2],
    )

    metadata = {
        'node_ids': np.asarray(all_nodes, dtype=np.int64),
        'node_coordinates': np.asarray([node_coords[node_id] for node_id in all_nodes], dtype=np.float64),
        'element_ids': np.asarray(all_elements, dtype=np.int64),
        'element_centers': np.asarray([element_centers[element_id] for element_id in all_elements], dtype=np.float64),
        'time': target_times,
        'modal_frequencies_hz': np.asarray(frequencies, dtype=np.float64),
        'runtime_seconds': duration,
        'gravity_disp': gravity_disp.astype(np.float32),
        'disp_total': all_disps_total,
        'displacement_definition': (
            'disp = disp_total - gravity_disp; disp is the dynamic increment '
            'used for surrogate-model training'
        ),
        'gravity_subtraction_max_reconstruction_error_m': reconstruction_error,
    }
    return all_disps, all_accels, all_strains, metadata


# --- Image Conversion ---
def array_to_image(data_array, output_path, global_max=None):
    """
    Normalizes a data array to the [-1, 1] range using a global maximum value,
    then maps it to [0, 255] for storage as a PNG image. This is the required
    format for training pix2pixHD models.

    Args:
        data_array (np.ndarray): The input data array, expected to be in (H, W, C) format.
        output_path (str): The path to save the output image.
        global_max (float, optional): The global maximum absolute value to use for normalization.
                                      If None, the maximum absolute value of the current array is used.
    """
    if data_array is None or data_array.size == 0:
        logging.warning(f"Input array for image conversion is empty. Cannot save {output_path}.")
        return

    if global_max is None:
        logging.warning("`global_max` not provided. Normalizing based on local max. This may not be suitable for model training.")
        global_max = np.max(np.abs(data_array))
    
    if global_max < 1e-9:
        logging.warning("`global_max` is near zero. The resulting image will be gray.")
        global_max = 1.0

    # Normalize the data to the [-1, 1] range
    normalized_array = np.clip(data_array / global_max, -1.0, 1.0)

    # Map the [-1, 1] range to [0, 255] for uint8 image format
    image_array = ((normalized_array + 1.0) / 2.0 * 255).astype(np.uint8)
    
    # Ensure the array is in (H, W, 3) format for RGB saving
    if image_array.ndim == 2:
        image_array = np.stack([image_array]*3, axis=-1)
    elif image_array.shape[2] == 1:
        image_array = np.concatenate([image_array]*3, axis=-1)

    # Create and save the image
    img = Image.fromarray(image_array, 'RGB')
    img.save(output_path)
    logging.info(f"  Successfully saved image: {output_path}")

# --- Main Execution ---
def configure_output_directory(output_directory):
    global OUTPUT_DIR, NODAL_OUTPUT_DIR, NODAL_IMG_INPUT_DIR, NODAL_IMG_DISP_DIR
    global NODAL_IMG_ACCEL_DIR, ELEMENTAL_OUTPUT_DIR, ELEMENTAL_IMG_INPUT_DIR
    global ELEMENTAL_IMG_STRAIN_DIR, RESPONSE_DATA_DIR, LOG_FILE_PATH

    OUTPUT_DIR = os.path.abspath(output_directory)
    NODAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'output_nodal')
    NODAL_IMG_INPUT_DIR = os.path.join(NODAL_OUTPUT_DIR, 'input_images')
    NODAL_IMG_DISP_DIR = os.path.join(NODAL_OUTPUT_DIR, 'response_disp')
    NODAL_IMG_ACCEL_DIR = os.path.join(NODAL_OUTPUT_DIR, 'response_accel')
    ELEMENTAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'output_elemental')
    ELEMENTAL_IMG_INPUT_DIR = os.path.join(ELEMENTAL_OUTPUT_DIR, 'input_images')
    ELEMENTAL_IMG_STRAIN_DIR = os.path.join(ELEMENTAL_OUTPUT_DIR, 'response_strain')
    RESPONSE_DATA_DIR = os.path.join(OUTPUT_DIR, 'response_data')
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'conversion_log.md')

    for path in [
        NODAL_IMG_INPUT_DIR, NODAL_IMG_DISP_DIR, NODAL_IMG_ACCEL_DIR,
        ELEMENTAL_IMG_INPUT_DIR, ELEMENTAL_IMG_STRAIN_DIR, RESPONSE_DATA_DIR,
        os.path.join(OUTPUT_DIR, 'response_npz'),
    ]:
        os.makedirs(path, exist_ok=True)
    setup_logging()


def save_node_mapping(metadata):
    path = os.path.join(OUTPUT_DIR, 'node_mapping.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['node_row', 'node_id', 'x', 'y', 'z'])
        for row, (node_id, coordinate) in enumerate(zip(metadata['node_ids'], metadata['node_coordinates'])):
            writer.writerow([row, int(node_id), *[float(value) for value in coordinate]])


def save_element_mapping(metadata):
    path = os.path.join(OUTPUT_DIR, 'element_mapping.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['element_row', 'element_id', 'center_x', 'center_y', 'center_z'])
        for row, (element_id, center) in enumerate(zip(metadata['element_ids'], metadata['element_centers'])):
            writer.writerow([row, int(element_id), *[float(value) for value in center]])


def build_argument_parser():
    parser = argparse.ArgumentParser(description='Corrected seismic OpenSees transient converter.')
    parser.add_argument(
        '--wav-dir',
        default=BASE_WAV_DIR,
        help='Directory containing the PEER three-component waveform folders.',
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help='Separate output directory; the original output is not overwritten.',
    )
    parser.add_argument('--limit-records', type=int, default=None, help='Use 1-3 for a smoke test.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Zero-based inclusive start index after all records are sorted.')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Zero-based exclusive end index after all records are sorted.')
    parser.add_argument('--count-only', action='store_true',
                        help='Scan records, print TOTAL_RECORDS=<n>, and exit without analysis.')
    parser.add_argument('--skip-images', action='store_true',
                        help='Do not generate PNG files. Recommended for NPZ-based training and parallel runs.')
    parser.add_argument('--cleanup-recorders', action='store_true',
                        help='Delete per-record OpenSees text recorder files after the NPZ is safely saved.')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing NPZ records.')
    parser.add_argument('--damping-mode-i', type=int, default=DAMPING_MODE_I)
    parser.add_argument('--damping-mode-j', type=int, default=DAMPING_MODE_J)
    return parser


def main():
    global BASE_WAV_DIR, DAMPING_MODE_I, DAMPING_MODE_J
    args = build_argument_parser().parse_args()
    BASE_WAV_DIR = str(Path(args.wav_dir).resolve())
    DAMPING_MODE_I = args.damping_mode_i
    DAMPING_MODE_J = args.damping_mode_j
    if DAMPING_MODE_I < 1 or DAMPING_MODE_J < 1 or DAMPING_MODE_I == DAMPING_MODE_J:
        raise ValueError('Damping modes must be distinct positive 1-based mode numbers.')

    configure_output_directory(args.output_dir)
    logging.info('--- Corrected seismic converter: gravity + Newmark transient analysis ---')
    logging.info('Input wave directory: %s', BASE_WAV_DIR)
    logging.info('Output directory: %s', OUTPUT_DIR)

    subfolders = sorted(entry.path for entry in os.scandir(BASE_WAV_DIR) if entry.is_dir())
    if not subfolders:
        raise FileNotFoundError(f'No subdirectories found in {BASE_WAV_DIR}')

    records = []
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        folder_records = get_seismic_records(folder_path)
        for record in folder_records:
            record['name'] = f"{folder_name}_{record['name']}"
        records.extend(folder_records)
    records.sort(key=lambda item: item['name'])
    total_records = len(records)
    logging.info('Total complete three-component records before slicing: %d', total_records)

    if args.count_only:
        print(f'TOTAL_RECORDS={total_records}')
        return

    start_index = max(0, int(args.start_index))
    end_index = total_records if args.end_index is None else min(total_records, max(start_index, int(args.end_index)))
    records = records[start_index:end_index]

    if args.limit_records is not None:
        records = records[:max(0, args.limit_records)]

    logging.info(
        'Selected record slice: start=%d, end=%d, selected=%d',
        start_index, end_index, len(records),
    )
    if not records:
        raise RuntimeError(
            f'No records selected. Total={total_records}, requested slice=[{start_index}:{end_index}].'
        )

    npz_dir = os.path.join(OUTPUT_DIR, 'response_npz')
    scale_rows = []
    failed_records = []
    successful_names = []
    global_max_input = 0.0
    global_max_disp = 0.0
    global_max_accel = 0.0
    global_max_strain = 0.0
    mapping_saved = False

    for index, record in enumerate(records, start=1):
        name = record['name']
        npz_path = os.path.join(npz_dir, f'{name}.npz')
        logging.info('\n[%d/%d] Processing %s', index, len(records), name)

        if os.path.exists(npz_path) and not args.overwrite:
            with np.load(npz_path) as cache:
                required_new_fields = {'disp_total', 'gravity_disp'}
                missing_new_fields = required_new_fields.difference(cache.files)
                if missing_new_fields:
                    logging.warning(
                        '  Legacy NPZ lacks %s; recomputing so gravity baseline is not mixed with new data.',
                        sorted(missing_new_fields),
                    )
                else:
                    logging.info('  Existing gravity-corrected NPZ found; loading without re-analysis.')
                    input_data = cache['input']
                    disp = cache['disp']
                    accel = cache['accel']
                    strain = cache['strain']
                    successful_names.append(name)
                    global_max_input = max(global_max_input, float(np.max(np.abs(input_data))))
                    global_max_disp = max(global_max_disp, float(np.max(np.abs(disp))))
                    global_max_accel = max(global_max_accel, float(np.max(np.abs(accel))))
                    global_max_strain = max(global_max_strain, float(np.max(np.abs(strain))))
                    continue

        prepared = prepare_three_components(record['H1_file'], record['H2_file'], record['V_file'])
        if prepared is None:
            failed_records.append(name)
            continue
        acc_h1, acc_h2, acc_v, scale_metadata = prepared

        try:
            result = run_opensees_analysis(acc_h1, acc_h2, acc_v, name)
        except Exception as exc:
            logging.exception('  Analysis raised an exception for %s: %s', name, exc)
            result = None
        if result is None:
            failed_records.append(name)
            continue

        disp, accel, strain, metadata = result
        input_data = np.stack([acc_h1, acc_h2, acc_v], axis=-1).astype(np.float32)
        np.savez_compressed(
            npz_path,
            input=input_data,
            # Dynamic earthquake-induced displacement used for model training.
            disp=disp,
            # Absolute displacement including gravity deformation, retained for engineering checks.
            disp_total=metadata['disp_total'],
            # Static gravity-equilibrium displacement, shape (num_nodes, 3).
            gravity_disp=metadata['gravity_disp'],
            accel=accel,
            strain=strain,
            node_ids=metadata['node_ids'],
            node_coordinates=metadata['node_coordinates'],
            element_ids=metadata['element_ids'],
            element_centers=metadata['element_centers'],
            time=metadata['time'],
            modal_frequencies_hz=metadata['modal_frequencies_hz'],
            displacement_definition=np.asarray(metadata['displacement_definition']),
            gravity_subtraction_max_reconstruction_error_m=np.asarray(
                metadata['gravity_subtraction_max_reconstruction_error_m'], dtype=np.float64
            ),
        )

        if not mapping_saved:
            save_node_mapping(metadata)
            save_element_mapping(metadata)
            mapping_saved = True

        scale_rows.append({'record': name, **scale_metadata})
        successful_names.append(name)
        global_max_input = max(global_max_input, float(np.max(np.abs(input_data))))
        global_max_disp = max(global_max_disp, float(np.max(np.abs(disp))))
        global_max_accel = max(global_max_accel, float(np.max(np.abs(accel))))
        global_max_strain = max(global_max_strain, float(np.max(np.abs(strain))))

        if args.cleanup_recorders:
            recorder_dir = os.path.join(RESPONSE_DATA_DIR, name)
            shutil.rmtree(recorder_dir, ignore_errors=True)
            logging.info('  Removed recorder text files after NPZ save: %s', recorder_dir)

    if scale_rows:
        with open(os.path.join(OUTPUT_DIR, 'record_scaling.csv'), 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(scale_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scale_rows)

    if not successful_names:
        raise RuntimeError('No successful analyses were produced.')

    normalization = {
        'input_acceleration_global_max_m_s2': global_max_input,
        'displacement_global_max_m': global_max_disp,
        'response_acceleration_global_max_m_s2': global_max_accel,
        'section_deformation_global_max_mixed_units': global_max_strain,
        'target_dt_s': TARGET_DT,
        'target_duration_s': TARGET_DURATION,
        'target_horizontal_pga_g': TARGET_PGA_G,
        'component_scaling': 'one common scale factor based on max horizontal PGA',
        'gravity_analysis': True,
        'displacement_definition': 'dynamic increment = total displacement - gravity equilibrium displacement',
        'npz_total_displacement_field': 'disp_total',
        'npz_gravity_displacement_field': 'gravity_disp',
        'training_displacement_field': 'disp',
        'transient_integrator': 'Newmark gamma=0.5 beta=0.25',
        'damping_ratio': DAMPING_RATIO,
        'damping_modes': [DAMPING_MODE_I, DAMPING_MODE_J],
        'record_slice_start_index': start_index,
        'record_slice_end_index': end_index,
        'skip_images': bool(args.skip_images),
        'cleanup_recorders': bool(args.cleanup_recorders),
        'successful_records': len(successful_names),
        'failed_records': len(failed_records),
    }
    with open(os.path.join(OUTPUT_DIR, 'normalization.json'), 'w', encoding='utf-8') as file:
        json.dump(normalization, file, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'successful_records.txt'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(successful_names) + '\n')
    with open(os.path.join(OUTPUT_DIR, 'failed_records.txt'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(failed_records) + ('\n' if failed_records else ''))

    if args.skip_images:
        logging.info('\nSkipping PNG generation (--skip-images). NPZ files are the formal training data.')
    else:
        logging.info('\nGenerating PNG images from float32 NPZ responses...')
        for index, name in enumerate(successful_names, start=1):
            logging.info('  Image %d/%d: %s', index, len(successful_names), name)
            with np.load(os.path.join(npz_dir, f'{name}.npz')) as cache:
                input_data = cache['input']
                disp = cache['disp']
                accel = cache['accel']
                strain = cache['strain']

            num_nodes, num_steps, _ = disp.shape
            num_elements = strain.shape[0]
            nodal_input = np.broadcast_to(input_data, (num_nodes, num_steps, 3)).copy()
            elemental_input = np.broadcast_to(input_data, (num_elements, num_steps, 3)).copy()

            array_to_image(nodal_input, os.path.join(NODAL_IMG_INPUT_DIR, f'{name}.png'), global_max_input)
            array_to_image(disp, os.path.join(NODAL_IMG_DISP_DIR, f'{name}.png'), global_max_disp)
            array_to_image(accel, os.path.join(NODAL_IMG_ACCEL_DIR, f'{name}.png'), global_max_accel)
            array_to_image(elemental_input, os.path.join(ELEMENTAL_IMG_INPUT_DIR, f'{name}.png'), global_max_input)
            array_to_image(strain, os.path.join(ELEMENTAL_IMG_STRAIN_DIR, f'{name}.png'), global_max_strain)

    logging.info('\n--- Corrected conversion finished ---')
    logging.info('Successful records: %d', len(successful_names))
    logging.info('Failed records: %d', len(failed_records))
    logging.info('Global Max Input Accel: %.12g m/s^2', global_max_input)
    logging.info('Global Max Dynamic Output Disp: %.12g m', global_max_disp)
    logging.info('Global Max Output Accel: %.12g m/s^2', global_max_accel)
    logging.info('Output: %s', OUTPUT_DIR)


if __name__ == '__main__':
    main()
