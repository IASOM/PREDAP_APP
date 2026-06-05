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

Entra al projecte principal:

```bash
cd TRANSFORMERS_PREDAP
```

Veure ajuda del CLI:

```bash
python predap_cli.py --help
```

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

## On mirar els detalls

- Guia principal del projecte: `TRANSFORMERS_PREDAP/README.md`
- Pipeline de produccio: `TRANSFORMERS_PREDAP/production/`
- Configuracio base de models: `TRANSFORMERS_PREDAP/src/config/base_transformer_config.py`
- Projecte AQUAS integrat: `TRANSFORMERS_PREDAP/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/`

## Notes

Aquest README nomes fa de mapa general de la supercarpeta. Per instal.lacio, dependencies, entrenament complet, quantitzacio i produccio, consulta el README de `TRANSFORMERS_PREDAP/`.
