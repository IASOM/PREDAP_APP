# TRANSFORMERS_PREDAP

Pipeline PREDAP per entrenar, quantitzar, reconstruir i predir demanda sanitaria amb models transformer.

## Flux curt

Executa aquestes comandes des de `TRANSFORMERS_PREDAP`:

```bash
python predap_cli.py train --stage full --code TOTAL --lookback 7 --forecast 30
python predap_cli.py quantize --codes TOTAL --lookback 7 --forecast 30
python predap_cli.py predict --code TOTAL --lookback 7 --forecast 30 --prediction-dates 2025-12-31 --no-delete-old
```

Per validar-ho tot amb missatges i comprovacions:

```bash
bash scripts/validate_pipeline.sh
```

El script comprova dades, BEST_features, forecasts invalids, models entrenats, pesos quantitzats i parquet de prediccio.

## Defaults operatius

El CLI fa servir per defecte el parquet real d'AQUAS:

```text
AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet
```

Els artefactes surten fora de `TRANSFORMERS_PREDAP`, a l'arrel del workspace:

```text
../trained_models/<code>/<model_type>/*.keras
../quantized_models/<code>/<model_type>/*_f16.weights.h5
../production_predictions/final_output_predictions/
```

`model_type` pot ser:

```text
univariate_model
diagnostics_model
seasonal_model
```

## BEST_features i forecasts valids

Els forecasts operatius han d'existir com a `LAG` dins:

```text
../data/best_features/BEST_features_NOSMOOTH_<code>.xlsx
```

El CLI filtra `train`, `quantize`, `reconstruct` i `predict` contra aquests `LAG`. Si demanes una llista amb forecasts no suportats, els salta. Si no queda cap forecast valid, para abans de carregar models.

Per `TOTAL`, els `LAG` disponibles actuals son:

```text
0, 1, 3, 7, 14, 30, 60, 182, 365
```

Exemple valid:

```bash
python predap_cli.py train --stage full --code TOTAL --lookback 7 --forecast 30
```

Exemple que ha de fallar:

```bash
python predap_cli.py predict --code TOTAL --lookback 7 --forecasts 35,42 --no-delete-old
```

## Comandes principals

Entrenar un stack complet:

```bash
python predap_cli.py train \
  --stage full \
  --code TOTAL \
  --lookback 7 \
  --forecast 30 \
  --epochs 50 \
  --batch-size 32
```

Entrenar diversos forecasts valids:

```bash
python predap_cli.py train \
  --stage full \
  --code TOTAL \
  --lookback 7 \
  --forecasts 30,60
```

Quantitzar models locals:

```bash
python predap_cli.py quantize \
  --codes TOTAL \
  --lookback 7 \
  --forecast 30 \
  --trained-model-folder ../trained_models \
  --quantized-weights-folder ../quantized_models
```

Predir amb pesos quantitzats:

```bash
python predap_cli.py predict \
  --code TOTAL \
  --lookback 7 \
  --forecast 30 \
  --model-folder ../quantized_models \
  --prediction-dates 2025-12-31 \
  --no-delete-old
```

`reconstruct` i `predict` fan el mateix. `predict` es l'alias curt; `reconstruct` deixa explicit que es reconstrueixen arquitectures i es carreguen pesos.

## Validacio manual

Comprovar que TensorFlow i Keras importen:

```bash
python -c "import tensorflow as tf; import keras; print(tf.__version__); print(keras.__version__)"
```

Comprovar el parquet d'entrada:

```bash
python -c "import pandas as pd; p='AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet'; df=pd.read_parquet(p, columns=['timestamp']); print(df['timestamp'].min(), df['timestamp'].max(), len(df))"
```

Comprovar `BEST_features`:

```bash
python -c "import pandas as pd; p='../data/best_features/BEST_features_NOSMOOTH_DEMAND_DEMANDA_TOTAL.xlsx'; df=pd.read_excel(p); print(sorted(df['LAG'].dropna().astype(int).unique()))"
```

Comprovar sortida de prediccio:

```bash
python -c "import pandas as pd; df=pd.read_parquet('../production_predictions/final_output_predictions'); print(df.shape); print(df.head())"
```

## MLflow

Hi ha dos camins:

1. CLI local: entrena `.keras` a `../trained_models` i quantitza des de carpeta local. No necessita MLflow.
2. Hydra/experiments: `main_experiments_hydra.py` logueja models, parametres i metriques a MLflow.

Arrencar UI MLflow:

```bash
python -m mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Executar un experiment Hydra amb MLflow local:

```bash
python main_experiments_hydra.py \
  model.target_code=TOTAL \
  model.lookback=7 \
  model.forecast=30 \
  model.head_size=4 \
  model.num_heads=1 \
  model.ff_dim=8 \
  model.mlp_units=8 \
  model.num_transformer_blocks=1 \
  model.dropout=0.0 \
  training.evaluate_model=false \
  data.data_path=AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet \
  mlflow.tracking_uri=file:./mlruns \
  mlflow.experiment_name=PREDAP_HYDRA_SMOKE
```

Quantitzar des de MLflow:

```bash
python predap_cli.py quantize \
  --experiments PREDAP_HYDRA_SMOKE_YYYYMMDD \
  --codes DEMAND_DEMANDA_TOTAL \
  --lookback 7 \
  --forecast 30 \
  --quantized-weights-folder ../quantized_models
```

Canvia `PREDAP_HYDRA_SMOKE_YYYYMMDD` pel nom real de l'experiment creat per `main_experiments_hydra.py`.

## Hydra

Les configuracions viuen a:

```text
conf/config.yaml
conf/config_production.yaml
conf/grid_search.yaml
conf/grid_search_V1.yaml
conf/grid_search_percentiles.yaml
```

Smoke run:

```bash
python main_experiments_hydra.py model.target_code=TOTAL model.lookback=7 model.forecast=30 training.evaluate_model=false
```

Multirun:

```bash
python main_experiments_hydra.py -m \
  model.target_code=TOTAL \
  model.lookback=7 \
  model.forecast=7,14,30 \
  model.head_size=4 \
  model.num_heads=1 \
  model.ff_dim=8 \
  model.mlp_units=8 \
  training.evaluate_model=false
```

Recomanacio: fes servir el CLI per produccio i Hydra per experiments. Quan una configuracio Hydra sigui bona, reprodueix-la amb `predap_cli.py train`, quantitza i valida amb `scripts/validate_pipeline.sh`.

## Fallbacks habituals

- Falta el parquet: executa AQUAS o passa `--data-path`.
- Falta un predictor: revisa que el nom del predictor existeixi al parquet; la pipeline avisara del predictor que no pot resoldre.
- Falta un `LAG`: tria un forecast disponible a `data/best_features`.
- Falta un `.keras`: executa `train` abans de `quantize`.
- Falta un `.weights.h5`: executa `quantize` abans de `predict`.
- Falta MLflow/Hydra: instal.la `requirements.txt`.

## Fitxers importants

```text
predap_cli.py
scripts/validate_pipeline.sh
production/model_quantization_pipeline.py
production/model_reconstruction_pipeline.py
production/data_preparation_in_poduction.py
src/config/base_transformer_config.py
main_experiments_hydra.py
```
