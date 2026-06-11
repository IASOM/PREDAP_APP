# PREDAP Forecasting Workspace

Aquest repositori agrupa tot el necessari per treballar amb el projecte PREDAP de predicció.

- `TRANSFORMERS_PREDAP/`: codi principal de models, entrenament, quantització, inferència, i documentació.
- `data/`: dades locals del projecte, incloent datasets finals i fitxers de característiques.
- `quantized_models/`: peses quantitzades preparades per reconstruir models en producció.

## Visió general

El flux principal és:

1. Recuperació i preparació de dades amb `AQUAS_DATA_RETRIEVAL`.
2. Entrenament de models transformer a `TRANSFORMERS_PREDAP`.
3. Quantització dels models i exportació de pesos float16.
4. Reconstrucció dels models quantitzats i generació de prediccions.

Així, AQUAS s'encarrega de les dades i `TRANSFORMERS_PREDAP` s'encarrega de la modelització.

## Com començar ràpid

### 1. Canvia al directori principal

```bash
cd TRANSFORMERS_PREDAP
```

### 2. Crea i activa un entorn virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Windows cmd:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Instal·la les dependències

```bash
python -m pip install -r requirements.txt
python -m pip install -r AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/requirements.txt
```

Per instal·lar només les dependències de documentació:

```bash
python -m pip install -r docs-requirements.txt
```

### 4. Verifica el CLI

```bash
python predap_cli.py --help
```

El CLI únic ofereix aquestes comandes principals:

- `aquas`: executa el pipeline AQUAS integrat.
- `sample-data`: genera dades sintètiques locals.
- `train`: entrena models univariants, residuals o el stack complet.
- `quantize`: converteix models entrenats en pesos quantitzats.
- `reconstruct`: reconstrueix models quantitzats i escriu prediccions.

Per veure les opcions específiques:

```bash
python predap_cli.py train --help
python predap_cli.py quantize --help
python predap_cli.py reconstruct --help
```

## Quantització: MLflow o local

La comanda `quantize` pot funcionar amb:

- `--experiments`: cercar models en experiments MLflow.
- `--trained-model-folder`: carregar models des de carpeta local si no hi ha MLflow.

Exemples:

```bash
python predap_cli.py quantize --experiments EXP1 --codes DEMAND_DEMANDA_TOTAL --lookbacks 7,14 --forecasts 7,14
```

```bash
python predap_cli.py quantize --trained-model-folder ../transformer_outputs/models_covid_token --codes DEMAND_DEMANDA_TOTAL --lookbacks 7,14 --forecasts 7,14
```

Això permet fer quantització local després d'un entrenament sense dependre d'un servidor MLflow.

## Exemples bàsics

Executar AQUAS en mode sample:

```bash
python predap_cli.py aquas -- --sample --all
```

Generar dades sintètiques multi-any:

```bash
python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
```

Entrenar un model univariant:

```bash
python predap_cli.py train --stage univariate --code TOTAL --lookback 7 --forecast 7
```

Reconstruir i predir amb models quantitzats:

```bash
python predap_cli.py reconstruct --code TOTAL --prediction-start 2025-12-23 --prediction-end 2025-12-31
```

## Bones pràctiques d'entorn

1. Utilitza la mateixa versió de Python a totes les màquines (3.11/3.12 recomanades).
2. No instal·lis dependències al Python global.
3. Activa sempre `.venv` abans d'executar `pip`, `pytest` o `predap_cli.py`.
4. Si no tens GPU, comprova la compatibilitat de `tensorflow` amb el `requirements.txt`.
5. Mantingues les dades grans fora del repositori: `data/`, `quantized_models/`, `mlruns/` i sortides de producció han de ser artefactes locals.

Congela l'entorn quan estigui validat:

```bash
python -m pip freeze > requirements-lock.txt
```

I replicas-lo a una altra màquina amb:

```bash
python -m pip install -r requirements-lock.txt
```

## Fitxers i rutes importants

- `TRANSFORMERS_PREDAP/README.md`: documentació específica del motor de models.
- `TRANSFORMERS_PREDAP/production/`: pipeline de producció, quantització i reconstrucció.
- `TRANSFORMERS_PREDAP/src/config/base_transformer_config.py`: configuració base de models i rutes.
- `TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/`: pipeline AQUAS integrat.

## Docker ràpid

Des de `TRANSFORMERS_PREDAP`:

```bash
cd TRANSFORMERS_PREDAP
docker compose build
docker compose up -d
docker compose logs -f mi-api-ia
```

API disponible a:

```text
http://127.0.0.1:8000
```

Abans d'aixecar Docker, comprova que les carpetes siguin presents a `PREDAP_APP`:

```text
data/
quantized_models/
production_predictions/
```

Per aturar:

```bash
docker compose down
```

## Verificació ràpida després d'instal·lar

```bash
python --version
python -m pip check
python predap_cli.py --help
python predap_cli.py sample-data --help
```

## Notes

Aquest README ofereix una visió general del repositori i la forma d'arrencar el projecte. Per detalls d'entrenament, quantització, reconstrucció i deploy, consulta `TRANSFORMERS_PREDAP/README.md`.

Notes:
- Ensure the container has the `data/` and `quantized_models/` mounts configured in `docker-compose.yml` so `/app/...` paths exist.
- Run the quick parquet sanity check shown earlier if the command fails to locate the file.
