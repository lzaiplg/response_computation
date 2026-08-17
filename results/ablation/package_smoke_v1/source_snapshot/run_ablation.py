from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

import torch

from ablation_data import NPZ_ROOT, TEST_LIST, TRAIN_LIST, VAL_LIST, dataset_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
REGISTRY = SCRIPT_DIR / "experiment_registry.json"
SOURCE_FILES = (
    "ablation_data.py",
    "ablation_model.py",
    "ablation_losses.py",
    "train_ablation_case.py",
    "run_ablation.py",
    "preflight_ablation.py",
    "analyze_campaign.py",
    "analyze_strategy_screen.py",
    "analyze_ablation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="screen_v1")
    parser.add_argument("--stage", choices=("screen", "confirmation", "screen_v2", "screen_v3", "confirmation_v2", "screen_v4"), default="screen")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--strategy-register", type=Path, default=SCRIPT_DIR / "innovation_strategy_register_v3.csv")
    parser.add_argument("--problem-contract", type=Path, default=SCRIPT_DIR / "problem_contract_v3.json")
    parser.add_argument("--target", choices=("disp", "accel_abs"), default="disp")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--direction-score-weight", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "results" / "ablation")
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def command_for(case: dict[str, Any], run_root: Path, args: argparse.Namespace) -> list[str]:
    architecture = str(case.get("architecture", "sacno"))
    sensor_noise_std = float(case.get("sensor_noise_std", 0.0))
    sensor_dropout_prob = float(case.get("sensor_dropout_prob", 0.0))
    eval_noise_std = float(case.get("eval_noise_std", 0.0))
    eval_dropout_prob = float(case.get("eval_dropout_prob", 0.0))
    peak_score_weight = float(case.get("peak_score_weight", 0.0))
    direction_score_weight = float(case.get("direction_score_weight", args.direction_score_weight))
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_ablation_case.py"),
        "--case-id", str(case["case_id"]),
        "--run-root", str(run_root),
        "--sensor-training", str(case["sensor_training"]),
        "--fusion", str(case["fusion"]),
        "--architecture", architecture,
        "--loss-profile", str(case["loss_profile"]),
        "--seed", str(case["seed"]),
        "--target", args.target,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--width", str(args.width),
        "--blocks", str(args.blocks),
        "--sensor-noise-std", str(sensor_noise_std),
        "--sensor-dropout-prob", str(sensor_dropout_prob),
        "--eval-noise-std", str(eval_noise_std),
        "--eval-dropout-prob", str(eval_dropout_prob),
        "--peak-score-weight", str(peak_score_weight),
        "--direction-score-weight", str(direction_score_weight),
        "--device", str(args.device),
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def pump(stream: TextIO, destination: TextIO, terminal: TextIO) -> None:
    try:
        for line in iter(stream.readline, ""):
            destination.write(line)
            destination.flush()
            terminal.write(line)
            terminal.flush()
    finally:
        stream.close()


def run_process(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(target=pump, args=(process.stdout, stdout_file, sys.stdout), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, stderr_file, sys.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return_code = process.wait()
        for thread in threads:
            thread.join()
        return return_code


def refresh_partial_summary(run_root: Path) -> None:
    rows = []
    for summary_path in sorted((run_root / "cases").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "case_id": summary["case_id"],
            "target": summary["target"],
            "sensor_training": summary["sensor_training"],
            "fusion": summary["fusion"],
            "architecture": summary.get("architecture", "sacno"),
            "loss_profile": summary["loss_profile"],
            "seed": summary["seed"],
            "best_epoch": summary["best_epoch"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for split in ("external", "formal"):
            if split not in summary:
                continue
            for mode in ("zero", "real"):
                if mode in summary[split]:
                    row[f"{split}_{mode}_nrmse"] = summary[split][mode]["nrmse_normalized"]["mean"]
                    row[f"{split}_{mode}_rmse_physical"] = summary[split][mode]["rmse_physical"]["mean"]
                    row[f"{split}_{mode}_peak_abs_physical"] = summary[split][mode]["peak_absolute_error_physical"]["mean"]
                    row[f"{split}_{mode}_peak_are"] = summary[split][mode]["peak_relative_error_pct"]["median"]
        gain = summary.get("formal", {}).get("sensor_information_gain") or summary.get("external", {}).get("sensor_information_gain")
        if gain:
            row["b1_win_rate_pct"] = gain["b1_win_rate_pct"]
            row["sensor_rmse_improvement_pct"] = gain["rmse_improvement_pct"]["mean"]
        rows.append(row)
    if not rows:
        return
    all_fields = sorted({key for row in rows for key in row})
    temporary = run_root / "partial_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, run_root / "partial_summary.csv")


def initialize(
    run_root: Path,
    commands: list[dict[str, Any]],
    args: argparse.Namespace,
    registry_path: Path,
    strategy_register_path: Path,
    problem_contract_path: Path,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "experiment_manifest.json"
    proposed_identity = {
        "campaign": args.campaign,
        "stage": args.stage,
        "target": args.target,
        "smoke": args.smoke,
        "case_ids": [item["case"]["case_id"] for item in commands],
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_identity = {
            "campaign": existing.get("campaign"),
            "stage": existing.get("stage", "screen"),
            "target": existing.get("target"),
            "smoke": existing.get("smoke"),
            "case_ids": [item["case"]["case_id"] for item in existing.get("planned_cases", [])],
        }
        if existing_identity != proposed_identity:
            raise RuntimeError(
                "Campaign directory already belongs to a different immutable experiment: "
                f"existing={existing_identity}, requested={proposed_identity}"
            )
    snapshot = run_root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        source = SCRIPT_DIR / name
        hashes[name] = sha256(source)
        destination = snapshot / name
        if not destination.exists():
            shutil.copy2(source, destination)
    for name in ("problem_contract.json",):
        destination = run_root / name
        if not destination.exists():
            shutil.copy2(problem_contract_path, destination)
    registry_destination = run_root / "experiment_registry.json"
    if not registry_destination.exists():
        shutil.copy2(registry_path, registry_destination)
    strategy_destination = run_root / "innovation_strategy_register.csv"
    if not strategy_destination.exists():
        shutil.copy2(strategy_register_path, strategy_destination)
    environment = {
        "created_at": now(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_hashes": hashes,
    }
    atomic_json(run_root / "environment.json", environment)
    fingerprint = dataset_manifest() | {
        "train_list_sha256": sha256(TRAIN_LIST),
        "val_list_sha256": sha256(VAL_LIST),
        "test_list_sha256": sha256(TEST_LIST),
        "npz_file_count": len(list(NPZ_ROOT.glob("*.npz"))),
    }
    atomic_json(run_root / "data_fingerprints.json", fingerprint)
    if not manifest_path.exists():
        launcher_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        atomic_json(
            manifest_path,
            {
            "campaign": args.campaign,
            "stage": args.stage,
            "created_at": now(),
            "target": args.target,
            "smoke": args.smoke,
            "launcher_args": launcher_args,
            "planned_cases": commands,
            "formal_test_available": True,
            "formal_test_count": dataset_manifest()["formal_test_count"],
            },
        )


def main() -> int:
    args = parse_args()
    registry_path = args.registry.resolve()
    strategy_register_path = args.strategy_register.resolve()
    problem_contract_path = args.problem_contract.resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    if not strategy_register_path.is_file():
        raise FileNotFoundError(strategy_register_path)
    if not problem_contract_path.is_file():
        raise FileNotFoundError(problem_contract_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cases = list(registry[args.stage])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    run_root = (args.output_root / args.campaign).resolve()
    planned = [{"case": case, "command": command_for(case, run_root, args)} for case in cases]
    initialize(run_root, planned, args, registry_path, strategy_register_path, problem_contract_path)
    execution_path = run_root / "execution.jsonl"
    for item in planned:
        case = item["case"]
        command = item["command"]
        case_dir = run_root / "cases" / case["case_id"]
        status_path = case_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed":
                append_jsonl(execution_path, {"status": "skipped_complete", "timestamp": now(), "case_id": case["case_id"], "command": command})
                continue
            if status.get("status") == "failed":
                if not args.retry_failed:
                    append_jsonl(execution_path, {"status": "skipped_failed", "timestamp": now(), "case_id": case["case_id"], "command": command})
                    continue
                archive = case_dir.with_name(case_dir.name + ".failed." + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
                case_dir.rename(archive)
            elif status.get("status") != "started":
                append_jsonl(execution_path, {"status": "skipped_unknown_status", "timestamp": now(), "case_id": case["case_id"], "command": command})
                continue
        stdout_path = run_root / "batch_error_logs" / f"{case['case_id']}.stdout.log"
        stderr_path = run_root / "batch_error_logs" / f"{case['case_id']}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            execution_path,
            {
                "status": "started",
                "timestamp": now(),
                "case_id": case["case_id"],
                "command": command,
                "run_dir": str(case_dir),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        return_code = run_process(command, stdout_path, stderr_path)
        append_jsonl(
            execution_path,
            {"status": "completed" if return_code == 0 else "failed", "timestamp": now(), "case_id": case["case_id"], "return_code": return_code},
        )
        refresh_partial_summary(run_root)
        if return_code != 0:
            print(f"CASE FAILED: {case['case_id']}", file=sys.stderr)
    refresh_partial_summary(run_root)
    print(f"CAMPAIGN FINISHED OR RESUMABLE: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
