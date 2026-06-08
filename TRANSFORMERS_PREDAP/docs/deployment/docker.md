# Docker

PREDAP uses Docker for reproducible GPU deployment of the FastAPI service and production reconstruction workflows.

The project Dockerfile is based on:

```dockerfile
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```

This NVIDIA NGC image already includes TensorFlow 2.17.0, Python 3.12.3, CUDA, cuDNN and TensorRT. For that reason, Docker installs `requirements-docker.txt`, not `requirements.txt`. Do not install `tensorflow==2.17.0+nv25.2` with normal pip; that build belongs to the NVIDIA container stack.

## Requirement Files

| File | Use |
| --- | --- |
| `requirements.txt` | Local CPU install. Uses `tensorflow-cpu>=2.16,<2.18`. |
| `requirements-gpu.txt` | Linux/WSL2 GPU install via pip. Uses `tensorflow[and-cuda]>=2.17,<2.18`. |
| `requirements-docker.txt` | Docker NVIDIA install. Excludes `tensorflow` and `keras` because the base image provides them. |
| `Inference/requirements.txt` | Lightweight CPU inference environment. |

`scikit-learn` is included as `scikit-learn>=1.4,<1.9`. The code uses stable preprocessing and metrics APIs such as `MinMaxScaler`, `FunctionTransformer`, `mean_absolute_error` and `mean_squared_error`, so it does not need to force only the newest scikit-learn release.

## Compose Environment

`docker-compose.yml` reads these values from `.env`:

```env
MODELS_FOLDER_NAME=quantized_models
READ_DATA_FOLDER_NAME=data
SAVE_DATA_FOLDER_NAME=production_predictions

RELATIVE_MODELS_PATH=../quantized_models
RELATIVE_READ_DATA_PATH=../data
RELATIVE_SAVE_DATA_PATH=../production_predictions
```

The host paths on the left are mounted into `/app/<folder>` inside the container:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `../quantized_models` | `/app/quantized_models` | Quantized model weights. |
| `../data` | `/app/data` | Input datasets. |
| `../production_predictions` | `/app/production_predictions` | Prediction outputs and production artifacts. |

Create the host folders before starting Compose:

```bash
mkdir -p ../quantized_models ../data ../production_predictions
```

On Windows without WSL2, create these folders manually or change `.env` to paths that Docker Desktop can mount.

## Start The API

From `TRANSFORMERS_PREDAP`:

```bash
docker compose build
docker compose up -d
docker compose logs -f mi-api-ia
```

The API is exposed only on localhost:

```text
http://127.0.0.1:8000
```

Redis is also started by Compose and is exposed only on localhost:

```text
127.0.0.1:6379
```

Stop the stack with:

```bash
docker compose down
```

## GPU Requirements

The Compose service reserves one NVIDIA GPU:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

The host machine needs:

1. NVIDIA driver compatible with the NGC TensorFlow 25.02 image.
2. Docker Engine or Docker Desktop with GPU support.
3. NVIDIA Container Toolkit on Linux/WSL2.

Verify TensorFlow and GPU visibility inside the running container:

```bash
docker compose exec mi-api-ia python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Expected TensorFlow version:

```text
2.17.0
```

If the GPU list is empty, first verify the host:

```bash
nvidia-smi
```

Then verify Docker GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## CPU-Only Alternative

For local CPU development, do not use the NVIDIA Docker route. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Training Jobs

For one-off training or CLI commands, reuse the built image and mount the same folders:

```bash
docker compose run --rm mi-api-ia python predap_cli.py --help
docker compose run --rm mi-api-ia python predap_cli.py reconstruct --code J00 --prediction-start 2025-12-23 --prediction-end 2025-12-31
```

For direct `docker run` usage:

```bash
docker build -t predap-api .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 ^
  -v "%cd%\..\quantized_models:/app/quantized_models" ^
  -v "%cd%\..\data:/app/data" ^
  -v "%cd%\..\production_predictions:/app/production_predictions" ^
  predap-api
```

Use the equivalent `$(pwd)` syntax on Linux/WSL2.
