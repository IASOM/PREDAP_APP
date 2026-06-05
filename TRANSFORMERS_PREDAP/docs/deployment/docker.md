# Docker Containers

Predap can be containerized for reproducible deployment across environments.

---

## Dockerfile

```dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Docker Compose

For a full stack with MLflow tracking:

```yaml
version: "3.8"

services:
  predap-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
      - ./production_predictions:/app/production_predictions
    environment:
      - MLFLOW_TRACKING_URI=file:./mlruns
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mlflow-ui:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: >
      mlflow ui
      --backend-store-uri file:///mlruns
      --host 0.0.0.0
      --port 5000
```

### Running

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f predap-api

# Stop
docker compose down
```

---

## Training Container

For training jobs (not serving), override the entrypoint:

```bash
# Single training run
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns \
    predap:latest python main.py

# Hydra grid search
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns \
    predap:latest python main_experiments_hydra.py --multirun \
    model.target_code=J00,B34 experiment_setup=7_7,14_14
```

---

## GPU Configuration

| Runtime | Flag | Notes |
|---------|------|-------|
| Docker CLI | `--gpus all` | Requires NVIDIA Container Toolkit |
| Docker Compose | `deploy.resources.reservations.devices` | See compose example above |
| CPU-only | No GPU flags | TensorFlow falls back to CPU automatically |

!!! tip "NVIDIA Container Toolkit"
    Install the toolkit to enable GPU passthrough:
    ```bash
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

---

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | Input datasets (`.parquet`) |
| `./models` | `/app/models` | Saved model weights (`.keras`, `.h5`) |
| `./mlruns` | `/app/mlruns` | MLflow experiment tracking |
| `./production_predictions` | `/app/production_predictions` | Production inference outputs |
