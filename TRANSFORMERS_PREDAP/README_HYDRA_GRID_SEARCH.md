# Hydra Grid Search for Transformer Experiments

This guide explains how to use Hydra for running grid search experiments instead of nested for loops.

## Files Structure

```
TRANSFORMERS_PREDAP/
├── conf/
│   ├── config.yaml              # Base configuration
│   └── grid_search.yaml         # Grid search parameters
├── main_experiments_hydra.py    # Updated Hydra-based script
└── README_HYDRA_GRID_SEARCH.md # This file
```

## Configuration Files

### Base Configuration (`conf/config.yaml`)
Contains the default parameters for a single run:
- Data path and model parameters
- Training settings
- MLflow configuration

### Grid Search Configuration (`conf/grid_search.yaml`)  
Defines all parameter combinations for the sweep:
- Uses Hydra's sweep functionality
- Automatically generates all parameter combinations
- Saves results in organized directory structure

## How to Run Grid Search

### Option 1: Run All Parameter Combinations
```bash
# Run the complete grid search with all parameters from grid_search.yaml
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun

# Alternative syntax
python main_experiments_hydra.py -cd conf -cn grid_search --multirun
```

### Option 2: Run Specific Parameter Subsets
```bash
# Run only specific target codes
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    model.target_code=T14,J00

# Run specific lookback and forecast combinations
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    model.lookback=7,14,30 model.forecast=7,14

# Run with specific model architecture parameters
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    model.num_heads=2,4 model.head_size=2,4 model.ff_dim=8,16
```

### Option 3: Override Individual Parameters
```bash
# Override specific parameters while keeping grid search for others
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    model.activation=relu model.covid_token=false

# Use a different data path
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    data.data_path="path/to/your/data.csv"
```

## Key Advantages of Hydra Approach

### 1. **Automatic Parameter Combination Generation**
- No more nested for loops
- Hydra automatically generates all combinations from `grid_search.yaml`
- Much cleaner and more maintainable code

### 2. **Organized Output Structure**  
```
outputs/
├── 2025-11-03/          # Date
│   └── 15-30-45/        # Time
│       ├── 0/           # Job 0 results
│       ├── 1/           # Job 1 results
│       ├── ...
│       └── multirun.yaml # Summary of all runs
```

### 3. **Easy Configuration Management**
- Centralized configuration in YAML files
- Easy to modify parameters without changing code
- Version control friendly
- Environment-specific configurations

### 4. **Parallel Execution** (Optional)
```bash
# Run jobs in parallel using joblib launcher
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    hydra/launcher=joblib hydra.launcher.n_jobs=4
```

### 5. **Resume and Filtering**
```bash
# Resume from specific job number
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    hydra.job.num=42

# Run only failed jobs (if tracking is implemented)
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    hydra.sweep.subdir='${hydra.job.num}_retry'
```

## Parameters in Grid Search

From `grid_search.yaml`, the following parameters are automatically combined:

### Target Codes
```yaml
model.target_code: T14,J00,M54
```

### Temporal Parameters
```yaml  
model.lookback: 7,14,30,60,182,365
model.forecast: 7,14,30,60,182,365
```

### Model Architecture
```yaml
model.num_transformer_blocks: 2,4,6
model.head_size: 2,4,8  
model.num_heads: 2,4,8
model.ff_dim: 8,16,32
model.mlp_units: 32,64,128
```

### Training Parameters
```yaml
model.activation: tanh,relu
model.covid_token: True,False
model.dropout: 0.1,0.3,0.5
model.learning_rate: 0.001,0.005
```

## Total Combinations

The total number of runs = Product of all parameter options:
- 3 target codes × 6 lookbacks × 6 forecasts × 3 transformer blocks × 3 head sizes × 3 num heads × 3 ff dims × 3 mlp units × 2 activations × 2 covid tokens × 3 dropouts × 2 learning rates
- **Total: 3 × 6 × 6 × 3 × 3 × 3 × 3 × 3 × 2 × 2 × 3 × 2 = 314,928 runs**

## Monitoring and Results

### MLflow Integration
- Each run is logged to MLflow automatically
- View results at: `http://localhost:5000`
- Results are tagged with Hydra job information

### Result Files
- Individual best results: `best_hyperparameters_<CODE>.json`
- Consolidated summary: `consolidated_best_results_summary.json`  
- CSV summary: `best_results_summary.csv`

### Hydra Logs
- Configuration for each run: `outputs/YYYY-MM-DD/HH-MM-SS/<job_num>/.hydra/`
- Consolidated run information: `outputs/YYYY-MM-DD/HH-MM-SS/multirun.yaml`

## Tips for Large Grid Searches

### 1. **Start Small**
```bash
# Test with a small subset first
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    model.target_code=T14 model.lookback=7,14 model.forecast=7,14
```

### 2. **Use Parallel Processing**
```bash
# Install joblib launcher
pip install hydra-joblib-launcher

# Run with parallel jobs
python main_experiments_hydra.py --config-path=conf --config-name=grid_search --multirun \
    hydra/launcher=joblib hydra.launcher.n_jobs=4
```

### 3. **Monitor Progress**
- Check MLflow UI for real-time progress
- Monitor `outputs/` directory for completed jobs
- Use system monitoring tools for resource usage

### 4. **Handle Interruptions**
- Hydra automatically saves progress
- Can resume by excluding completed parameter combinations
- Results are saved incrementally

### Hydra usage
```python
@hydra.main(config_path="conf", config_name="config")
def main_experiment(cfg: DictConfig) -> None:
    # Training logic here using cfg parameters...

# Run with: python script.py --multirun
```
