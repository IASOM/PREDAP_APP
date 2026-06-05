# TRANSFORMERS_PREDAP

Pipeline de modelatge PREDAP per entrenar, quantitzar i executar models de prediccio de demanda sanitaria. El repositori integra el projecte `AQUAS_DATA_RETRIEVAL`, que s'encarrega d'obtenir i materialitzar les dades, i manté aquí tota la part de deep learning: preparacio de dades per model, entrenament, correccio de residuals, quantitzacio i reconstruccio en produccio.

## Visio general

El sistema queda dividit en dos blocs:

- `AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main`: ingesta i agregacio de dades de demanda i diagnostics. Genera Parquet finals com `demand_diagnosis_joined.parquet`.
- `TRANSFORMERS_PREDAP`: entrenament i inferencia amb tres fases de model: transformer univariant, residuals diagnostics i residuals estacionals.

Flux recomanat:

1. Generar o recuperar dades amb AQUAS.
2. Entrenar models amb `predap train`.
3. Quantitzar pesos per produccio amb `predap quantize`.
4. Reconstruir models i generar prediccions amb `predap reconstruct`.

## Estructura del projecte

```text
TRANSFORMERS_PREDAP/
  predap_cli.py                         # CLI principal
  scripts/predap.sh                     # wrapper Bash
  production/
    retrieve_and_reconstruct_data_pipeline.py
    model_reconstruction_pipeline.py
    model_quantization_pipeline.py
    data_preparation_in_poduction.py
  src/
    main_train_univ_transformer_class.py
    main_train_diagnostic_residual_transformer_class.py
    main_train_seasonal_residual_transformer_class.py
    config/base_transformer_config.py
    data_utils/
    univariate_transformer/
    residual_multivariate_transformers/
  AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/
    run_pipeline_optimized.py
    scripts/create_multiyear_sample.py
  docs/
  requirements.txt
```

## Instal·lacio

Des de la carpeta `TRANSFORMERS_PREDAP`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/requirements.txt
```

En Windows amb Git Bash, el wrapper funciona igual si `python` apunta a l'entorn correcte:

```bash
PYTHON_BIN=.venv/Scripts/python.exe bash scripts/predap.sh --help
```

## CLI rapida

Totes les comandes es poden executar des de Bash:

```bash
bash scripts/predap.sh --help
bash scripts/predap.sh sample-data --start 2010-01-01 --end 2023-10-31
bash scripts/predap.sh aquas -- --sample --all
bash scripts/predap.sh train --stage univariate --code J00 --lookbacks 7,14 --forecasts 7,14
bash scripts/predap.sh reconstruct --code DEMAND_demanda_SERVEI_CODI_INF --prediction-start 2025-12-23 --prediction-end 2025-12-31
```

També es pot cridar directament amb Python:

```bash
python predap_cli.py --help
```

## 1. Obtenir dades amb AQUAS

Per executar el pipeline AQUAS integrat, fes servir el subcomandament `aquas`. Els arguments que venen despres de `aquas` es passen al runner intern `run_pipeline_optimized.py`.

```bash
bash scripts/predap.sh aquas -- --all --start-date 2024-01-01 --end-date 2024-12-31
```

Mode sample sense base de dades:

```bash
bash scripts/predap.sh aquas -- --sample --all
```

Generar un dataset sintetic multi-any:

```bash
bash scripts/predap.sh sample-data \
  --start 2010-01-01 \
  --end 2023-10-31
```

Sortida esperada en sample:

```text
AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/sample/multiyear_output/finals/demand_diagnosis_joined.parquet
```

## 2. Entrenar models

El training stack esta dividit en tres fases:

- `univariate`: model transformer base per cada codi.
- `diagnostic`: model residual amb covariables diagnostiques.
- `seasonal`: model residual amb covariables de calendari/estacionalitat.
- `full`: executa les tres fases de forma consecutiva.

Exemple curt:

```bash
bash scripts/predap.sh train \
  --stage full \
  --codes J00,I10,M54 \
  --lookbacks 7,14 \
  --forecasts 7,14 \
  --data-path ../data/FINAL_DB/full_CAT1.parquet \
  --model-folder ../transformer_outputs/models_covid_token \
  --epochs 50 \
  --batch-size 32
```

Per entrenar nomes el model univariant:

```bash
bash scripts/predap.sh train \
  --stage univariate \
  --code J00 \
  --lookbacks 7 \
  --forecasts 7
```

## 3. Quantitzar models

La quantitzacio carrega models des de MLflow i guarda pesos `float16` en una estructura apta per produccio.

```bash
bash scripts/predap.sh quantize \
  --experiments 1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260514 \
  --codes demanda__SERVEI_CODI__INF,demanda__SERVEI_CODI__MF \
  --lookbacks 7,14,60,60,182,182 \
  --forecasts 7,14,30,60,182,365 \
  --data-path ../data/FINAL_DB/finals_combined.csv
```

Els pesos es guarden sota:

```text
../quantized_models/<code>/<model_type>/<code>_<model_type>_<forecast>fh_<lookback>lb_f16_weights.h5
```

## 4. Reconstruir i predir en produccio

`reconstruct` reconstrueix les arquitectures, carrega els pesos quantitzats i escriu prediccions particionades per codi.

```bash
bash scripts/predap.sh reconstruct \
  --code DEMAND_demanda_SERVEI_CODI_INF \
  --data-path AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/sample/multiyear_output/finals/demand_diagnosis_joined.parquet \
  --old-data-path ../data/FINAL_DB/finals_combined.csv \
  --model-folder ../quantized_models \
  --lookbacks 7,14,60,60,182,182 \
  --forecasts 7,14,30,60,182,365 \
  --prediction-start 2025-12-23 \
  --prediction-end 2025-12-31
```

Opcions utils:

- `--codes A,B,C`: predir una llista concreta de codis.
- `--all-codes`: llegir codis des del dataset d'entrada.
- `--prediction-dates 2025-12-23,2025-12-24`: passar dates concretes.
- `--no-delete-old`: no fer neteja/evaluacio posterior de prediccions antigues.
- `--output-dir`: canviar la carpeta de prediccions de produccio.

## Configuracio important

Els defaults principals viuen a `src/config/base_transformer_config.py`. Els mes rellevants son:

- `data_path`: dataset base d'entrenament o inferencia.
- `diagnostic_covariates_path`: prefix dels Excels `BEST_features_NOSMOOTH_*`.
- `model_folder`: carpeta de models o pesos.
- `LOOKBACK_LIST` i `FORECAST_LIST`: combinacions temporals a entrenar o reconstruir.
- `cutoff_date`, `max_date`, `covid_token` i `eliminate_covid_data`.

El CLI permet sobreescriure els valors mes comuns sense editar el codi.

## Notes de produccio

- AQUAS hauria de ser la font canonica de dades finals. Aquest projecte no hauria de tornar a implementar ingesta SQL si AQUAS ja ho resol.
- `production/retrieve_and_reconstruct_data_pipeline.py` encara conte un flux antic/hardcoded de simulacio. El nou `predap reconstruct` es mes net per execucio manual i automatitzacio.
- El projecte te imports historics que depenen del directori d'execucio. `predap_cli.py` afegeix `TRANSFORMERS_PREDAP` i `src` al `PYTHONPATH` en runtime per reduir aquests errors.
- Per entrenaments llargs, MLflow queda configurat localment a `mlruns`. Pots veure els experiments amb `mlflow ui`.

## Com ho veig

La separacio conceptual es bona: AQUAS per dades, TRANSFORMERS_PREDAP per models. El que faltava era una entrada unica i documentacio operacional. Encara hi ha deute tecnic en noms de carpetes (`architechture`, `poduction`) i alguns scripts amb valors hardcoded, pero ara el cami executable queda molt mes clar i es pot automatitzar des de Bash sense entrar a tocar fitxers interns.
