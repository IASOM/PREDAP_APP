# PREDAP Forecasting Workspace

Aquesta supercarpeta agrupa tot el necessari per treballar amb el projecte PREDAP de prediccio:

- `TRANSFORMERS_PREDAP/`: codi principal de models, entrenament, quantitzacio, inferencia i documentacio tecnica.
- `data/`: dades locals utilitzades pel projecte, incloent datasets finals i fitxers de features.
- `quantized_models/`: pesos quantitzats preparats per reconstruir models en produccio.

## Idea general

El flux del projecte es:

1. Recuperar i preparar dades amb `AQUAS_DATA_RETRIEVAL`, que esta integrat dins `TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/`.
2. Entrenar models transformer amb `TRANSFORMERS_PREDAP`.
3. Quantitzar els models entrenats i guardar els pesos a `quantized_models/`.
4. Reconstruir els models quantitzats i generar prediccions de produccio.

En resum: AQUAS s'encarrega de les dades; TRANSFORMERS_PREDAP s'encarrega dels models.

## Comencar rapid

### 1. Entra al projecte principal

```bash
cd TRANSFORMERS_PREDAP
```

### 2. Crea un entorn virtual

Fes servir sempre un entorn virtual local. Aixi evitem barrejar dependencies del sistema amb les del projecte.

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

### 3. Instal.la dependencies

Instal.la primer el projecte de models i despres el pipeline AQUAS integrat:

```bash
python -m pip install -r requirements.txt
python -m pip install -r AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/requirements.txt
```

Si nomes vols generar documentacio:

```bash
python -m pip install -r docs-requirements.txt
```

### 4. Comprova que el CLI respon

```bash
python predap_cli.py --help
```

El CLI te aquests metodes principals:

- `aquas`: executa el pipeline AQUAS integrat.
- `sample-data`: genera dades sintetiques locals.
- `train`: entrena models univariants, residuals o el stack complet.
- `quantize`: converteix models entrenats en pesos quantitzats de produccio.
- `reconstruct`: reconstrueix models quantitzats i genera prediccions.

Per veure totes les opcions d'un metode concret:

```bash
python predap_cli.py train --help
python predap_cli.py reconstruct --help
python predap_cli.py quantize --help
```

## Exemples basics

Executar el pipeline AQUAS en mode sample:

```bash
python predap_cli.py aquas -- --sample --all
```

Generar dades sintetiques multi-any:

```bash
python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
```

Entrenar un model simple:

```bash
python predap_cli.py train --stage univariate --code J00 --lookback 7 --forecast 7
```

Reconstruir i predir amb models quantitzats:

```bash
python predap_cli.py reconstruct --code DEMAND_demanda_SERVEI_CODI_INF --prediction-start 2025-12-23 --prediction-end 2025-12-31
```

## Instal.lacio estable entre maquines

Per reduir problemes quan el projecte es mou entre portatils, servidors o contenidors:

1. Fes servir la mateixa versio minor de Python a totes les maquines. Per aquest projecte es recomanable treballar amb Python 3.11 o 3.12 si alguna dependencia de ML falla amb versions mes noves.
2. No instal.lis dependencies al Python global. Activa sempre `.venv` abans d'executar `pip`, `pytest`, `mlflow` o `predap_cli.py`.
3. Instal.la amb `python -m pip ...` en comptes de `pip ...`; aixi queda clar quin Python esta rebent les dependencies.
4. Si una maquina no te GPU/NVIDIA preparada, valida especialment la dependencia `tensorflow` del `requirements.txt`. En servidors de produccio, millor replicar la mateixa imatge Docker o el mateix lockfile.
5. Quan tinguis una maquina validada, congela l'entorn:

```bash
python -m pip freeze > requirements-lock.txt
```

I en una altra maquina replica exactament aquell entorn:

```bash
python -m pip install -r requirements-lock.txt
```

6. Mantingues les dades grans fora de git. Les carpetes `data/`, `quantized_models/`, `mlruns/` i sortides de produccio han de ser artefactes locals o de servidor.

El projecte separa tres escenaris:

- `TRANSFORMERS_PREDAP/requirements.txt`: instal.lacio local CPU amb `tensorflow-cpu`.
- `TRANSFORMERS_PREDAP/requirements-gpu.txt`: instal.lacio Linux/WSL2 amb GPU via pip.
- `TRANSFORMERS_PREDAP/requirements-docker.txt`: instal.lacio dins Docker NVIDIA; no inclou TensorFlow perque la imatge base ja el porta.

Per desplegar amb Docker i GPU, mira `TRANSFORMERS_PREDAP/docs/deployment/docker.md`.

## Docker rapid

Per llançar la API amb Docker NVIDIA/GPU des de la carpeta arrel `PREDAP_APP`:

```bash
cd TRANSFORMERS_PREDAP
docker compose build
docker compose up -d
docker compose logs -f mi-api-ia
```

La API queda disponible a:

```text
http://127.0.0.1:8000
```

Abans d'aixecar-la, comprova que existeixen aquestes carpetes al nivell de `PREDAP_APP`:

```text
data/
quantized_models/
production_predictions/
```

Per parar-ho:

```bash
docker compose down
```

Checklist de verificacio despres d'instal.lar:

```bash
python --version
python -m pip check
python predap_cli.py --help
python predap_cli.py sample-data --help
```

## On mirar els detalls

- Guia principal del projecte: `TRANSFORMERS_PREDAP/README.md`
- Pipeline de produccio: `TRANSFORMERS_PREDAP/production/`
- Configuracio base de models: `TRANSFORMERS_PREDAP/src/config/base_transformer_config.py`
- Projecte AQUAS integrat: `TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/`

## Notes

Aquest README fa de mapa general de la supercarpeta i dona la instal.lacio basica. Per entrenament complet, quantitzacio, reconstruccio i produccio, consulta el README de `TRANSFORMERS_PREDAP/`.
 
## Docker: Reconstruct inside container

If you want to run `reconstruct` inside the Docker container (recommended to match production deps):

1. Build and start the container from `TRANSFORMERS_PREDAP`:

```bash
cd TRANSFORMERS_PREDAP
docker compose build
docker compose up -d
```

2. Exec into the running service and run `reconstruct` (adjust service name if different):

```bash
docker compose exec mi-api-ia bash
python predap_cli.py reconstruct --code DEMAND_demanda_SERVEI_CODI_INF \
	--data-path /app/TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/sample/multiyear_output/finals/demand_diagnosis_joined.parquet \
	--prediction-start 2025-12-23 --prediction-end 2025-12-31
```

Notes:
- Ensure the container has the `data/` and `quantized_models/` mounts configured in `docker-compose.yml` so `/app/...` paths exist.
- Run the quick parquet sanity check shown earlier if the command fails to locate the file.
