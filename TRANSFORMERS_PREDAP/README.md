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
  scripts/predap.py                    # Python wrapper
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
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
pip install -r AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/requirements.txt
```

Per defecte `requirements.txt` instal.la TensorFlow CPU (`tensorflow-cpu`). Aixo es el mes portable per Windows, macOS i servidors sense GPU.

Si vols instal.lar TensorFlow amb GPU via pip en Linux/WSL2:

```bash
pip install -r requirements-gpu.txt
pip install -r AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/requirements.txt
```

Per Docker NVIDIA no facis servir `requirements.txt`: la imatge base ja porta TensorFlow, CUDA i cuDNN. El Dockerfile utilitza `requirements-docker.txt`, que exclou TensorFlow per no substituir la build optimitzada d'NVIDIA.

## CLI ràpida

Les comandes s'executen directament amb Python:

```bash
python predap_cli.py --help
python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
python predap_cli.py aquas -- --sample --all
python predap_cli.py train --stage univariate --code TOTAL --lookbacks 7,14 --forecasts 7,14
python predap_cli.py reconstruct --code TOTAL --prediction-start 2025-12-23 --prediction-end 2025-12-31
```

Per comoditat també hi ha un wrapper Python:

```bash
python scripts/predap.py --help
```

## Docker NVIDIA

Aquest projecte ofereix un `docker-compose.yml` per executar la API amb una imatge basada en NVIDIA TensorFlow.

### 1. Preparar el directori del projecte

A la carpeta `TRANSFORMERS_PREDAP` crea o comprova que existeixin aquestes carpetes locals:

```bash
mkdir -p quantized_models data production_predictions
```

Si fas servir Windows sense WSL2, crea les mateixes carpetes amb l'Explorador o PowerShell.

### 2. Configurar variables d'entorn

Crea un fitxer `.env` a `TRANSFORMERS_PREDAP` amb els valors següents:

```env
MODELS_FOLDER_NAME=quantized_models
READ_DATA_FOLDER_NAME=data
SAVE_DATA_FOLDER_NAME=production_predictions

RELATIVE_MODELS_PATH=./quantized_models
RELATIVE_READ_DATA_PATH=./data
RELATIVE_SAVE_DATA_PATH=./production_predictions
```

Els valors `RELATIVE_*` indiquen quines carpetes del teu ordinador es connecten dins del container.

### 3. Construir i arrencar Docker

Des de `TRANSFORMERS_PREDAP` executa:

```bash
docker compose build
docker compose up -d
```

Després comprova l'estat del servei:

```bash
docker compose ps
docker compose logs -f mi-api-ia
```

### 4. Accedir a la API

La API queda exposada a:

```text
http://127.0.0.1:8000
```

Per provar si està viva:

```bash
curl http://127.0.0.1:8000
```

### 5. Verificar TensorFlow i GPU

Dins el container pots verificar la instal·lació amb:

```bash
docker compose exec mi-api-ia python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Si no surt cap GPU, comprova:

- Drivers NVIDIA instal·lats al host.
- Docker Desktop/Engine amb suport GPU activat.
- NVIDIA Container Toolkit instal·lat.
- Que el container utilitzi GPU, no només CPU.

### 6. Notes importants

- El `dockerfile` utilitza `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`, per tant no cal instal·lar TensorFlow dins el container.
- `requirements-docker.txt` només instal·la dependències addicionals, no `tensorflow` ni `keras`.
- Els volums es munten en aquestes rutes dins del container:
  - `/app/quantized_models`
  - `/app/data`
  - `/app/production_predictions`

### 7. Si tens errors d'UTF-8 en Docker

A vegades el problema no és Docker, sinó que el codi intenta llegir un fitxer `.parquet` amb `pd.read_csv()`. Aquest README assumeix que el dataset d'entrada és un `.parquet` correcte i existeix a:

```text
TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/sample/multiyear_output/finals/demand_diagnosis_joined.parquet
```

Si executant la comanda `python predap_cli.py reconstruct ...` tens un error UTF-8, comprova:

1. Que `--data-path` apunta a un `.parquet`, no a un `.csv` amb codificació errònia.
2. Que el fitxer existeix realment en la ruta local abans de muntar-lo a Docker.
3. Que dins el container la ruta coincideix amb `/app/data` si uses `READ_DATA_FOLDER_NAME=data`.

## 1. Obtenir dades amb AQUAS

Per executar el pipeline AQUAS integrat, fes servir el subcomandament `aquas`. Els arguments que venen despres de `aquas` es passen al runner intern `run_pipeline_optimized.py`.

```bash
python predap_cli.py aquas -- --all --start-date 2024-01-01 --end-date 2024-12-31
```

Mode sample sense base de dades:

```bash
python predap_cli.py aquas -- --sample --all
```

Generar un dataset sintetic multi-any:

```bash
python predap_cli.py sample-data \
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
python predap_cli.py train \
  --stage full \
  --codes TOTAL \
  --lookbacks 7,14 \
  --forecasts 7,14 \
  --data-path ../data/FINAL_DB/full_CAT1.parquet \
  --model-folder ../transformer_outputs/models_covid_token \
  --epochs 50 \
  --batch-size 32
```

Per entrenar nomes el model univariant:

```bash
python predap_cli.py train \
  --stage univariate \
  --code TOTAL \
  --lookbacks 7 \
  --forecasts 7
```

## 3. Quantitzar models

La quantitzacio carrega models des de MLflow i guarda pesos `float16` en una estructura apta per produccio.

```bash
python predap_cli.py quantize \
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
python predap_cli.py reconstruct \
  --code TOTAL \
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
