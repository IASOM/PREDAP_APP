#!/usr/bin/env bash
# PREDAP operational validation runbook.
#
# Usage from the repository root:
#   bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh
#
# Useful switches:
#   PYTHON=/path/to/python bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh
#   RUN_TRAIN=0 bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh
#   RUN_MULTIPLE=1 bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh
#   RUN_MLFLOW=1 bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh
#   RUN_HYDRA=1 bash TRANSFORMERS_PREDAP/scripts/validate_pipeline.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
CODE="${CODE:-TOTAL}"
LOOKBACK="${LOOKBACK:-7}"
FORECAST="${FORECAST:-30}"
VALID_FORECASTS="${VALID_FORECASTS:-30}"
INVALID_FORECASTS="${INVALID_FORECASTS:-35,42}"
PREDICTION_DATE="${PREDICTION_DATE:-2025-12-31}"

DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet}"
BEST_PREFIX="${BEST_PREFIX:-${APP_DIR}/data/best_features/BEST_features_NOSMOOTH_}"
TRAINED_DIR="${TRAINED_DIR:-${APP_DIR}/trained_models}"
QUANTIZED_DIR="${QUANTIZED_DIR:-${APP_DIR}/quantized_models}"
OUTPUT_DIR="${OUTPUT_DIR:-${APP_DIR}/production_predictions_validation}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_QUANTIZE="${RUN_QUANTIZE:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
RUN_RECONSTRUCT="${RUN_RECONSTRUCT:-1}"
RUN_MULTIPLE="${RUN_MULTIPLE:-0}"
RUN_MLFLOW="${RUN_MLFLOW:-0}"
RUN_HYDRA="${RUN_HYDRA:-0}"

# Small architecture for quick smoke tests. Override these for real training.
MODEL_ARGS=(
  --head-size "${HEAD_SIZE:-4}"
  --num-heads "${NUM_HEADS:-1}"
  --ff-dim "${FF_DIM:-8}"
  --num-transformer-blocks "${NUM_BLOCKS:-1}"
  --mlp-units "${MLP_UNITS:-8}"
)

TRAIN_ARGS=(
  --epochs "${EPOCHS:-1}"
  --batch-size "${BATCH_SIZE:-1024}"
  --no-evaluate-model
)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

ok() {
  printf '[OK] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

run() {
  printf '+ %q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

require_file() {
  local path="$1"
  local hint="${2:-}"
  if [[ ! -f "${path}" ]]; then
    fail "No trobo el fitxer: ${path}. ${hint}"
  fi
  ok "Fitxer trobat: ${path}"
}

require_glob() {
  local pattern="$1"
  local hint="${2:-}"
  if ! compgen -G "${pattern}" >/dev/null; then
    fail "No trobo cap fitxer amb patro: ${pattern}. ${hint}"
  fi
  ok "Patro trobat: ${pattern}"
}

module_status() {
  local module="$1"
  if "${PYTHON}" -c "import ${module}" >/dev/null 2>&1; then
    ok "Modul Python disponible: ${module}"
  else
    warn "Falta el modul Python '${module}'. Fallback: instala dependències amb '${PYTHON} -m pip install -r TRANSFORMERS_PREDAP/requirements.txt'."
  fi
}

require_module() {
  local module="$1"
  if "${PYTHON}" -c "import ${module}" >/dev/null 2>&1; then
    ok "Modul Python requerit disponible: ${module}"
  else
    fail "Falta el modul Python '${module}' al Python actiu (${PYTHON}). Fallback: executa amb PYTHON=/path/to/python_amb_tensorflow o instala 'python -m pip install -r TRANSFORMERS_PREDAP/requirements.txt'."
  fi
}

validate_runtime_for_models() {
  if [[ "${RUN_TRAIN}" == "1" || "${RUN_QUANTIZE}" == "1" || "${RUN_PREDICT}" == "1" || "${RUN_RECONSTRUCT}" == "1" || "${RUN_HYDRA}" == "1" ]]; then
    log "Validant runtime per TensorFlow/Keras"
    require_module tensorflow
    require_module keras
  fi
}

validate_data_and_best_features() {
  log "Validant dades reals i BEST_features"
  require_file "${DATA_PATH}" "Executa AQUAS o revisa DATA_PATH."
  if [[ -f "${BEST_PREFIX}${CODE}.xlsx" ]]; then
    ok "BEST_features directe trobat: ${BEST_PREFIX}${CODE}.xlsx"
  else
    warn "No hi ha BEST_features directe per CODE=${CODE}; provare aliases com DEMAND_DEMANDA_TOTAL o TOTAL."
  fi

  "${PYTHON}" - "${DATA_PATH}" "${BEST_PREFIX}" "${CODE}" "${FORECAST}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

data_path = Path(sys.argv[1])
prefix = Path(sys.argv[2])
code = sys.argv[3]
forecast = int(sys.argv[4])

df = pd.read_parquet(data_path, columns=["timestamp"])
print(f"[OK] Parquet llegible: {data_path}")
print(f"[OK] Rang temporal: {df['timestamp'].min()} -> {df['timestamp'].max()} ({len(df)} files)")

candidates = [
    Path(f"{prefix}{code}.xlsx"),
    Path(f"{prefix}DEMAND_DEMANDA_TOTAL.xlsx") if code.upper() == "TOTAL" else None,
    Path(f"{prefix}TOTAL.xlsx") if code.upper() in {"TOTAL", "DEMAND_DEMANDA_TOTAL"} else None,
]
candidates = [p for p in candidates if p is not None]
selected = next((p for p in candidates if p.exists()), None)
if selected is None:
    raise SystemExit(f"[ERROR] No trobo BEST_features per {code}. Provats: {candidates}")

features = pd.read_excel(selected, engine="openpyxl")
lags = sorted(features["LAG"].dropna().astype(int).unique().tolist())
print(f"[OK] BEST_features: {selected}")
print(f"[OK] LAG disponibles: {lags}")
if forecast not in lags:
    raise SystemExit(f"[ERROR] Forecast {forecast} no es un LAG valid. Tria un de: {lags}")

row = features.loc[features["LAG"].astype(int) == forecast].iloc[0]
predictors = [item.strip() for item in str(row.get("predictors", "")).split(",") if item.strip()]
print(f"[OK] Predictors per LAG={forecast}: {len(predictors)}")
if not predictors:
    raise SystemExit("[ERROR] El LAG existeix pero no te predictors.")
PY
}

expect_invalid_forecasts_to_fail() {
  log "Comprovant que forecasts sense BEST_features fallen abans de reconstruir"
  local tmp_log
  tmp_log="$(mktemp)"
  set +e
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" predict \
    --code "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecasts "${INVALID_FORECASTS}" \
    --data-path "${DATA_PATH}" \
    --model-folder "${QUANTIZED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}" \
    --output-dir "${OUTPUT_DIR}/invalid_forecasts" \
    --prediction-dates "${PREDICTION_DATE}" \
    --no-delete-old \
    "${MODEL_ARGS[@]}" >"${tmp_log}" 2>&1
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    cat "${tmp_log}" >&2
    rm -f "${tmp_log}"
    fail "Els forecasts invalids (${INVALID_FORECASTS}) han passat. S'esperava error."
  fi
  if grep -q "No requested forecasts" "${tmp_log}"; then
    ok "Missatge d'error esperat detectat: no hi ha forecasts validats per BEST_features."
  else
    warn "Els forecasts invalids han fallat, pero el missatge no era l'esperat. Ultimes linies:"
    tail -n 20 "${tmp_log}" >&2
  fi
  rm -f "${tmp_log}"
  ok "Forecasts invalids rebutjats correctament: ${INVALID_FORECASTS}"
}

train_valid_stack() {
  log "Entrenant stack valid: code=${CODE}, lookback=${LOOKBACK}, forecast=${FORECAST}"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" train \
    --stage full \
    --code "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecast "${FORECAST}" \
    --data-path "${DATA_PATH}" \
    --model-folder "${TRAINED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}" \
    "${MODEL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}"

  require_glob "${TRAINED_DIR}/DEMAND_DEMANDA_TOTAL/univariate_model/*_${FORECAST}fh_*_${LOOKBACK}lb_*.keras" "El model base no s'ha creat."
  require_glob "${TRAINED_DIR}/DEMAND_DEMANDA_TOTAL/diagnostics_model/*_${FORECAST}fh_*_${LOOKBACK}lb_*.keras" "El model diagnostics no s'ha creat."
  require_glob "${TRAINED_DIR}/DEMAND_DEMANDA_TOTAL/seasonal_model/*_${FORECAST}fh_*_${LOOKBACK}lb_*.keras" "El model seasonal no s'ha creat."
}

quantize_valid_stack() {
  log "Quantitzant stack valid"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" quantize \
    --codes "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecast "${FORECAST}" \
    --data-path "${DATA_PATH}" \
    --trained-model-folder "${TRAINED_DIR}" \
    --quantized-weights-folder "${QUANTIZED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}"

  require_glob "${QUANTIZED_DIR}/DEMAND_DEMANDA_TOTAL/univariate_model/*_${FORECAST}fh_${LOOKBACK}lb_f16.weights.h5" "Falten pesos univariate quantitzats."
  require_glob "${QUANTIZED_DIR}/DEMAND_DEMANDA_TOTAL/diagnostics_model/*_${FORECAST}fh_${LOOKBACK}lb_f16.weights.h5" "Falten pesos diagnostics quantitzats."
  require_glob "${QUANTIZED_DIR}/DEMAND_DEMANDA_TOTAL/seasonal_model/*_${FORECAST}fh_${LOOKBACK}lb_f16.weights.h5" "Falten pesos seasonal quantitzats."
}

predict_valid_stack() {
  local mode="$1"
  local output="$2"
  log "Executant ${mode} amb pesos quantitzats"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" "${mode}" \
    --code "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecast "${FORECAST}" \
    --data-path "${DATA_PATH}" \
    --model-folder "${QUANTIZED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}" \
    --output-dir "${output}" \
    --prediction-dates "${PREDICTION_DATE}" \
    --no-delete-old \
    "${MODEL_ARGS[@]}"

  "${PYTHON}" - "${output}" "${FORECAST}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

output = Path(sys.argv[1])
forecast = int(sys.argv[2])
if not output.exists():
    raise SystemExit(f"[ERROR] No existeix output-dir: {output}")
df = pd.read_parquet(output)
print(f"[OK] Prediccions llegibles: {output} shape={df.shape}")
required = {"target_date", "lookback", "forecast", "forecast_step", "predictions"}
missing = required.difference(df.columns)
if missing:
    raise SystemExit(f"[ERROR] Columnes absents a prediccions: {sorted(missing)}")
rows = len(df.loc[df["forecast"].astype(int) == forecast])
if rows != forecast:
    raise SystemExit(f"[ERROR] Files esperades={forecast}; files trobades={rows}")
print(df[["lookback", "forecast", "forecast_step", "target_date", "predictions"]].head(5).to_string(index=False))
PY
}

optional_multiple_forecasts() {
  if [[ "${RUN_MULTIPLE}" != "1" ]]; then
    warn "RUN_MULTIPLE=0. Saltant prova multi-forecast. Activa RUN_MULTIPLE=1 per provar VALID_FORECASTS=${VALID_FORECASTS}."
    return
  fi
  log "Prova multi-forecast amb forecasts valids: ${VALID_FORECASTS}"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" train \
    --stage full \
    --code "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecasts "${VALID_FORECASTS}" \
    --data-path "${DATA_PATH}" \
    --model-folder "${TRAINED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}" \
    "${MODEL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" quantize \
    --codes "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecasts "${VALID_FORECASTS}" \
    --data-path "${DATA_PATH}" \
    --trained-model-folder "${TRAINED_DIR}" \
    --quantized-weights-folder "${QUANTIZED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}"
  "${PYTHON}" "${PROJECT_DIR}/predap_cli.py" predict \
    --code "${CODE}" \
    --lookback "${LOOKBACK}" \
    --forecasts "${VALID_FORECASTS}" \
    --data-path "${DATA_PATH}" \
    --model-folder "${QUANTIZED_DIR}" \
    --diagnostic-covariates-prefix "${BEST_PREFIX}" \
    --output-dir "${OUTPUT_DIR}/predict_multi" \
    --prediction-dates "${PREDICTION_DATE}" \
    --no-delete-old \
    "${MODEL_ARGS[@]}"
  ok "Prova multi-forecast acabada."
}

optional_mlflow() {
  if [[ "${RUN_MLFLOW}" != "1" ]]; then
    warn "RUN_MLFLOW=0. Saltant MLflow. Activa RUN_MLFLOW=1 per validar el cami MLflow/Hydra."
    return
  fi
  log "Validant MLflow"
  module_status mlflow
  "${PYTHON}" -c "import mlflow; mlflow.set_tracking_uri('file:${APP_DIR}/mlruns'); print('[OK] MLflow tracking URI:', mlflow.get_tracking_uri())"
  warn "Per obrir la UI: cd ${PROJECT_DIR} && ${PYTHON} -m mlflow ui --backend-store-uri file:${APP_DIR}/mlruns --host 127.0.0.1 --port 5000"
}

optional_hydra() {
  if [[ "${RUN_HYDRA}" != "1" ]]; then
    warn "RUN_HYDRA=0. Saltant Hydra. Activa RUN_HYDRA=1 per executar main_experiments_hydra.py."
    return
  fi
  log "Executant Hydra smoke run amb overrides petits"
  module_status hydra
  (
    cd "${PROJECT_DIR}"
    "${PYTHON}" main_experiments_hydra.py \
      model.target_code="${CODE}" \
      model.lookback="${LOOKBACK}" \
      model.forecast="${FORECAST}" \
      model.head_size=4 \
      model.num_heads=1 \
      model.ff_dim=8 \
      model.mlp_units=8 \
      model.num_transformer_blocks=1 \
      model.dropout=0.0 \
      model.learning_rate=0.001 \
      training.evaluate_model=false \
      data.data_path="${DATA_PATH}" \
      mlflow.tracking_uri="file:${APP_DIR}/mlruns" \
      mlflow.experiment_name="PREDAP_HYDRA_SMOKE"
  )
}

main() {
  log "Inici validacio PREDAP"
  log "Python: ${PYTHON}"
  "${PYTHON}" --version
  module_status pandas
  module_status pyarrow
  module_status openpyxl
  module_status tensorflow
  validate_runtime_for_models

  mkdir -p "${TRAINED_DIR}" "${QUANTIZED_DIR}" "${OUTPUT_DIR}"

  validate_data_and_best_features
  expect_invalid_forecasts_to_fail

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    train_valid_stack
  else
    warn "RUN_TRAIN=0. Es comprovaran artefactes existents."
  fi

  if [[ "${RUN_QUANTIZE}" == "1" ]]; then
    quantize_valid_stack
  else
    warn "RUN_QUANTIZE=0. Es faran servir pesos quantitzats existents."
  fi

  if [[ "${RUN_PREDICT}" == "1" ]]; then
    predict_valid_stack predict "${OUTPUT_DIR}/predict_single"
  fi

  if [[ "${RUN_RECONSTRUCT}" == "1" ]]; then
    predict_valid_stack reconstruct "${OUTPUT_DIR}/reconstruct_single"
  fi

  optional_multiple_forecasts
  optional_mlflow
  optional_hydra

  log "Validacio completada"
  ok "Pipeline operativa per CODE=${CODE}, LOOKBACK=${LOOKBACK}, FORECAST=${FORECAST}"
}

main "$@"
