# Installation

## Prerequisites

- **Python** 3.9 or higher
- **TensorFlow** 2.x (GPU support recommended)
- **CUDA** 11.8+ and **cuDNN** 8.6+ (for GPU acceleration)

---

## Installation Methods

=== "pip"

    Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/IASOM/PREDAP.git
    cd predap/TRANSFORMERS_PREDAP
    pip install -r requirements.txt
    ```

=== "conda"

    Create a dedicated Conda environment from the provided YAML:

    ```bash
    git clone https://github.com/IASOM/PREDAP.git
    cd predap/TRANSFORMERS_PREDAP
    conda env create -f environment.yml
    conda activate predap
    ```

=== "Docker"

    Build and run using the provided Dockerfile:

    ```bash
    git clone https://github.com/IASOM/PREDAP.git
    cd predap/TRANSFORMERS_PREDAP
    docker build -t predap:latest .
    docker run --gpus all -p 8000:8000 predap:latest
    ```

---

## Core Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow >= 2.x` | Model building, training, and inference |
| `pandas`, `numpy` | Data manipulation and numerical operations |
| `scikit-learn` | Preprocessing (MinMaxScaler, FunctionTransformer) |
| `mlflow` | Experiment tracking, model registry |
| `hydra-core` | Configuration management and grid search |
| `fastapi`, `uvicorn` | REST API for production inference |
| `matplotlib`, `seaborn` | Visualization and plotting |
| `openpyxl` | Excel I/O for diagnostic covariate files |
| `holidays` | Catalan public holiday calendar generation |

---

## GPU Setup

Predap automatically detects and configures available GPUs. Memory growth is enabled by default to prevent TensorFlow from allocating all GPU memory at once:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

Verify your GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Data Requirements

Before running experiments you need:

1. **Main dataset** — A `.parquet` file containing timestamped diagnostic visit counts (one column per diagnostic code, a `timestamp` column).
2. **Diagnostic covariates** — Excel files from the LMLR + GCausal feature selection pipeline, named `BEST_features_NOSMOOTH_{CODE}.xlsx`.
3. **Target codes JSON** (optional) — A JSON file listing all diagnostic codes for batch runs.

!!! note "Data Path Configuration"
    Update `data_path` and `diagnostic_covariates_path` in `src/config/base_transformer_config.py` or via Hydra YAML overrides.

---

## Verify Installation

```bash
# Check core imports
python -c "
from src.config.base_transformer_config import BaseTransformerConfig
from src.main_train_univ_transformer_class import UnivariateTransformerPipeline
print('Predap installed successfully!')
print(f'Default config: {BaseTransformerConfig().print_config()}')
"
```

---

## MLflow Tracking Server (Optional)

Start the MLflow UI to browse experiment results:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.
