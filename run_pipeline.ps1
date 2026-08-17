param(
    [ValidateSet("preflight", "opensees-smoke", "build-data", "train-a1", "train-b0", "train-b1", "test-b0", "test-b1", "plot", "ablation-preflight", "ablation-smoke", "ablation-screen", "all")]
    [string]$Stage = "preflight",
    [int]$Stage1Epochs = 100,
    [int]$Stage2Epochs = 100,
    [int]$Patience = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = "python"

function Invoke-Python([string]$Script, [string[]]$Arguments) {
    & $Python $Script @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed ($LASTEXITCODE): $Script" }
}

function Invoke-Stage([string]$Name) {
    switch ($Name) {
        "preflight" {
            Invoke-Python ".\code\evaluation\preflight_pipeline.py" @()
        }
        "opensees-smoke" {
            Invoke-Python ".\code\opensees\convert_full_response.py" @("--limit-records", "1", "--skip-images", "--cleanup-recorders", "--output-dir", ".\data\processed\response_smoke")
        }
        "build-data" {
            Invoke-Python ".\code\data\prepare_baseline_rgb.py" @("--overwrite")
            Invoke-Python ".\code\data\prepare_acceleration_rgb.py" @("--overwrite")
            Invoke-Python ".\code\data\prepare_sensor_update.py" @("--overwrite")
            Invoke-Python ".\code\data\prepare_sensor_acceleration.py" @("--overwrite")
            Invoke-Python ".\code\evaluation\preflight_pipeline.py" @("--require-datasets")
        }
        "train-a1" {
            Invoke-Python ".\code\baseline\train_baseline_rgb.py" @("--mode", "train", "--run-name", "A1_acceleration", "--dataset-root", ".\data\processed\baseline_acceleration\dataset_rgb", "--result-root", ".\models\A1_acceleration_run", "--stage1-epochs", $Stage1Epochs, "--stage2-epochs", $Stage2Epochs, "--early-stop-patience", $Patience, "--batch-size", "1", "--num-workers", "0", "--overwrite-run")
            Copy-Item ".\models\A1_acceleration_run\checkpoints\best.pt" ".\models\A1_acceleration_best.pt" -Force
        }
        "train-b0" {
            Invoke-Python ".\code\training\train_sensor_update.py" @("--mode", "train", "--sensor-mode", "zero", "--run-name", "B0", "--output-root", ".\results\B0", "--init-a1", ".\models\A1_acceleration_best.pt", "--stage1-epochs", $Stage1Epochs, "--stage2-epochs", $Stage2Epochs, "--early-stop-patience", $Patience, "--batch-size", "1", "--num-workers", "0", "--overwrite-run")
        }
        "train-b1" {
            Invoke-Python ".\code\training\train_sensor_update.py" @("--mode", "train", "--sensor-mode", "real", "--run-name", "B1", "--output-root", ".\results\B1", "--init-a1", ".\models\A1_acceleration_best.pt", "--stage1-epochs", $Stage1Epochs, "--stage2-epochs", $Stage2Epochs, "--early-stop-patience", $Patience, "--batch-size", "1", "--num-workers", "0", "--overwrite-run")
        }
        "test-b0" {
            Invoke-Python ".\code\training\test_sensor_update.py" @("--checkpoint", ".\results\B0\checkpoints\best.pt", "--sensor-mode", "zero", "--run-name", "B0", "--output-root", ".\results\B0\test", "--expected-count", "78", "--plots-per-record", "2", "--overwrite")
        }
        "test-b1" {
            Invoke-Python ".\code\training\test_sensor_update.py" @("--checkpoint", ".\results\B1\checkpoints\best.pt", "--sensor-mode", "real", "--run-name", "B1", "--output-root", ".\results\B1\test", "--expected-count", "78", "--plots-per-record", "2", "--overwrite")
        }
        "plot" {
            Invoke-Python ".\code\evaluation\plot_pier_curves.py" @("--b0-dir", ".\results\B0\test\predicted_npz", "--b1-dir", ".\results\B1\test\predicted_npz", "--output-dir", ".\results\curves", "--all-records")
        }
        "ablation-preflight" {
            Invoke-Python ".\code\ablation\preflight_ablation.py" @()
        }
        "ablation-smoke" {
            Invoke-Python ".\code\ablation\run_ablation.py" @("--campaign", "package_smoke_v1", "--registry", ".\code\ablation\experiment_registry_v3.json", "--strategy-register", ".\code\ablation\innovation_strategy_register_v3.csv", "--problem-contract", ".\code\ablation\problem_contract_v3.json", "--stage", "screen_v4", "--target", "accel_abs", "--smoke", "--max-cases", "2", "--batch-size", "1", "--num-workers", "0", "--device", "cpu")
        }
        "ablation-screen" {
            Invoke-Python ".\code\ablation\run_ablation.py" @("--campaign", "acceleration_screen_v4", "--registry", ".\code\ablation\experiment_registry_v3.json", "--strategy-register", ".\code\ablation\innovation_strategy_register_v3.csv", "--problem-contract", ".\code\ablation\problem_contract_v3.json", "--stage", "screen_v4", "--target", "accel_abs", "--epochs", "60", "--patience", "12", "--batch-size", "1", "--num-workers", "0", "--direction-score-weight", "0.35")
        }
        "all" {
            Invoke-Stage "build-data"; Invoke-Stage "train-a1"; Invoke-Stage "train-b0"; Invoke-Stage "train-b1"; Invoke-Stage "test-b0"; Invoke-Stage "test-b1"; Invoke-Stage "plot"
        }
    }
}

Invoke-Stage $Stage
