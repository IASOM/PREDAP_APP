# Hydra Experiment Runner

This tutorial covers the Hydra-based experiment runner (`main_experiments_hydra.py`), which provides declarative YAML-driven training with automatic parameter sweeps and MLflow integration.

---


## YAML Configuration Structure

### Base Config (`conf/config.yaml`)

```yaml
data:
  data_path: '../data/FINAL_DB/full_CAT1.parquet'
  codes_path: '../data/FINAL_DB/targets_CAT.json'

model:
  target_code: "T14"
  lookback: 7
  forecast: 7
  num_transformer_blocks: 2
  head_size: 32
  num_heads: 8
  activation: "gelu"
  covid_token: true
  ff_dim: 512
  mlp_units: [512, 256]
  dropout: 0.5
  learning_rate: 1e-5

training:
  cutoff_date: "2008-01-01"
  positional_encoding: true
  evaluate_model: true

mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "HYDRA_EXPERIMENT"
```

### Grid Search Config (`conf/grid_search_V1.yaml`)

Uses the `joblib` launcher for optional parallelism and pre-defined experiment pairs:

```yaml
defaults:
  - _self_
  - override hydra/launcher: joblib

experiment_setup: "7_7"

experiment_pairs:
  3_3:     { lb: 3,   fc: 3 }
  7_7:     { lb: 7,   fc: 7 }
  14_14:   { lb: 14,  fc: 14 }
  60_30:   { lb: 60,  fc: 30 }
  60_60:   { lb: 60,  fc: 60 }
  182_182: { lb: 182, fc: 182 }
  365_182: { lb: 182, fc: 365 }

model:
  lookback: ${experiment_pairs.${experiment_setup}.lb}
  forecast: ${experiment_pairs.${experiment_setup}.fc}
```

!!! info "Hydra Variable Interpolation"
    The `${experiment_pairs.${experiment_setup}.lb}` syntax dynamically resolves the lookback value based on the selected `experiment_setup`. Changing `experiment_setup` to `"60_30"` automatically sets `lookback=60` and `forecast=30`.

---

## Running Experiments

### Single Run

```bash
python main_experiments_hydra.py \
    model.target_code=J00 \
    experiment_setup=14_14
```

### Multi-Run Sweep

```bash
python main_experiments_hydra.py --multirun \
    model.target_code=J00,B34,I10 \
    experiment_setup=7_7,14_14,60_30
```

This generates $3 \times 3 = 9$ independent runs.

### Architecture Sweep

```bash
python main_experiments_hydra.py --multirun \
    model.head_size=32,64 \
    model.num_heads=4,8,16 \
    model.ff_dim=256,512 \
    model.target_code=J00
```

This generates $2 \times 3 \times 2 = 12$ architecture configurations.

---

## Hydra Output Structure

Each run creates a timestamped directory:

```
outputs/
└── 2026-03-03/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml       # Fully resolved configuration
        │   ├── hydra.yaml        # Hydra runtime metadata
        │   └── overrides.yaml    # CLI parameters used
        ├── main_experiments_hydra.log  # Training logs
        └── multirun.yaml         # Multi-run sweep definition (if --multirun)
```

---

## Parallel Execution

The `joblib` launcher enables parallel execution:

```yaml
hydra:
  launcher:
    n_jobs: 2         # Run 2 experiments simultaneously
    backend: loky     # Process-based parallelism
```

!!! warning "GPU Memory Constraints"
    With `n_jobs > 1`, multiple models load into GPU memory concurrently. Monitor GPU utilization with `nvidia-smi` and reduce `n_jobs` if OOM errors occur.

---

## Percentile-Based Sweeps

`conf/grid_search_percentiles.yaml` enables sweeping over percentile-stratified diagnostic codes:

```bash
python main_experiments_hydra.py \
    --config-name=grid_search_percentiles \
    --multirun \
    model.target_code=Z14,O40,F70,R82,L82
```

This is useful for evaluating model performance across diagnostic codes with different volume characteristics (rare vs. common conditions).

---

## Tips

1. **Start small**: Test with `experiment_setup=3_3` before launching long sweeps
2. **Use `--cfg job`** to preview the resolved configuration without running:
   ```bash
   python main_experiments_hydra.py --cfg job model.target_code=J00
   ```
3. **Resume from checkpoint**: If a sweep is interrupted, re-run with the same parameters — MLflow will track new runs alongside existing ones
4. **Monitor live**: Open `mlflow ui` in a separate terminal to watch metrics as runs complete
