# Next-round ablation plan

## Scope

The package uses 480 records for training, 69 for checkpoint selection, 157 historical records for external reporting and 78 untouched records for the formal test. B0 and B1 are always matched by architecture, loss, seed, optimizer budget and split.

## Screen matrix

`experiment_registry_v3.json` defines the paired screen for SACNO, phase-balanced loss, physics-proxy loss, spectral SACNO and AMS variants. The output remains the complete acceleration field, with primary metrics on 148 unobserved nodes and secondary metrics on the four pier tops.

## Commands

```powershell
python .\code\ablation\preflight_ablation.py
python .\code\ablation\run_ablation.py --campaign acceleration_screen_v4 --registry .\code\ablation\experiment_registry_v3.json --strategy-register .\code\ablation\innovation_strategy_register_v3.csv --problem-contract .\code\ablation\problem_contract_v3.json --stage screen_v4 --target accel_abs --epochs 60 --patience 12 --batch-size 1 --num-workers 0 --direction-score-weight 0.35
python .\code\ablation\analyze_ablation.py --campaign acceleration_screen_v4 --evaluation-split external
python .\code\ablation\analyze_ablation.py --campaign acceleration_screen_v4 --evaluation-split formal
```

Use a new campaign name when changing any scientific condition. The launcher writes to `results/ablation/<campaign>`, archives source hashes and skips completed immutable cases on resume.

## Retention rule

Do not retain a method from one favorable case. Require lower full-field and worst-direction error, no unacceptable peak degradation, B1 improvement over matched B0 on most records, three seeds and formal-test confirmation. If validated M/C/K or modal data become available, add a separate FE equilibrium-loss factor rather than relabeling the current proxy loss.
