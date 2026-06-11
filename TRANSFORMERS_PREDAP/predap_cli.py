#!/usr/bin/env python
"""Command line entry point for the TRANSFORMERS_PREDAP project."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
AQUAS_ROOT = REPO_ROOT / "AQUAS_DATA_RETRIEVAL" / "AQUAS_DATA_RETRIEVAL-main"
DEFAULT_DATA_PATH = AQUAS_ROOT / "data" / "finals" / "demand_diagnosis_joined.parquet"
DEFAULT_TRAINED_MODELS_DIR = REPO_ROOT.parent / "trained_models"
DEFAULT_QUANTIZED_MODELS_DIR = REPO_ROOT.parent / "quantized_models"
DEFAULT_BEST_FEATURES_PREFIX = REPO_ROOT.parent / "data" / "best_features" / "BEST_features_NOSMOOTH_"


class PredapHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep multiline examples readable while showing defaults."""


def _ensure_local_imports() -> None:
    for path in (REPO_ROOT, SRC_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


# Ensure local imports early so we can safely override pandas CSV reader before other modules import it.
_ensure_local_imports()
try:
    # On Windows, importing pandas/pyarrow before TensorFlow can load DLLs that
    # make TensorFlow's native runtime fail later. Preload TensorFlow when present.
    import tensorflow  # noqa: F401
except Exception:
    pass
try:
    from src.utils.experiments_utils import smart_read
    import pandas as pd
    pd.read_csv = smart_read
except Exception:
    # If importing the smart reader fails (e.g., missing heavy deps in minimal environments),
    # silently continue — call-sites still may import and use `smart_read` explicitly.
    pass


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def _csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one value")
    return items


def _canonical_code_name(name: str) -> str:
    canonical = str(name).strip().replace("#", ":")
    if canonical.startswith("DEMAND_"):
        canonical = canonical[len("DEMAND_"):]
    canonical = canonical.replace("__", "_").upper()
    if canonical == "TOTAL":
        canonical = "DEMANDA_TOTAL"
    aliases = {
        "VISI_SITUACIO_VISITA_N": "VISI_SITUACIO_VISITA_NO_PROGRAMADA",
        "VISI_SITUACIO_VISITA_P": "VISI_SITUACIO_VISITA_PROGRAMADA",
        "VISI_SITUACIO_VISITA_R": "VISI_SITUACIO_VISITA_URGENT",
        "SERVEI_CODI_MF": "SERVEI_CODI_MEDFAM",
    }
    for legacy, current in aliases.items():
        canonical = re.sub(
            rf"(^|_){re.escape(legacy)}($|_)",
            lambda match: f"{match.group(1)}{current}{match.group(2)}",
            canonical,
        )
    return re.sub(r"[^A-Z0-9]+", "_", canonical).strip("_")


def _dataset_columns(data_path: Path) -> list[str]:
    if str(data_path).lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_schema(data_path).names
    import pandas as pd

    return pd.read_csv(data_path, nrows=0).columns.tolist()


def _filter_codes_in_dataset(codes: list[str], data_path: Path) -> list[str]:
    columns = _dataset_columns(data_path)
    lookup = {_canonical_code_name(column): column for column in columns if column != "timestamp"}
    resolved_codes = []
    skipped_codes = []
    for code in codes:
        resolved = lookup.get(_canonical_code_name(code))
        if resolved is None:
            skipped_codes.append(code)
        else:
            if resolved != code:
                print(f"-> INFO: Mapped requested code '{code}' to dataset column '{resolved}'.")
            resolved_codes.append(resolved)
    if skipped_codes:
        print(
            "-> WARNING: Skipping codes not found in input dataset: "
            + ", ".join(skipped_codes)
        )
    if not resolved_codes:
        raise ValueError("None of the requested codes were found in the input dataset.")
    return resolved_codes


def _best_feature_file_candidates(prefix: Path | str, code: str) -> list[Path]:
    prefix_path = Path(prefix)
    if prefix_path.suffix.lower() == ".xlsx":
        return [prefix_path]

    code_text = str(code).strip().replace("#", ":")
    variants = [
        code_text,
        code_text.replace("__", "_"),
        code_text.replace(":", "#"),
        code_text.upper(),
    ]
    if code_text.startswith("DEMAND_"):
        variants.append(code_text[len("DEMAND_"):])
    else:
        variants.append(f"DEMAND_{code_text}")

    ordered = []
    for variant in variants:
        for normalized in (variant, variant.replace("__", "_")):
            path = Path(f"{prefix_path}{normalized}.xlsx")
            if path not in ordered:
                ordered.append(path)
    return ordered


def _best_feature_lags(code: str, prefix: Path | str) -> set[int]:
    import pandas as pd

    candidates = _best_feature_file_candidates(prefix, code)
    selected_path = next((path for path in candidates if path.exists()), None)
    if selected_path is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No BEST_features file found for code '{code}'. Tried: {tried}"
        )

    df = pd.read_excel(selected_path, engine="openpyxl")
    if "LAG" not in df.columns:
        raise ValueError(f"BEST_features file has no LAG column: {selected_path}")

    lags = set(df["LAG"].dropna().astype(int).tolist())
    print(f"-> INFO: Available BEST_features LAGs for {code}: {sorted(lags)}")
    return lags


def _filter_temporal_pairs_by_best_features(
    code: str,
    temporal_pairs: list[tuple[int, int]],
    prefix: Path | str,
) -> list[tuple[int, int]]:
    valid_lags = _best_feature_lags(code, prefix)
    filtered_pairs = [
        (lookback, forecast)
        for lookback, forecast in temporal_pairs
        if forecast in valid_lags
    ]
    skipped = [
        (lookback, forecast)
        for lookback, forecast in temporal_pairs
        if forecast not in valid_lags
    ]
    if skipped:
        skipped_text = ", ".join(f"{lookback}/{forecast}" for lookback, forecast in skipped)
        print(
            f"-> WARNING: Skipping unsupported lookback/forecast pairs for {code}: "
            f"{skipped_text}. Forecast must exist as LAG in data/best_features."
        )
    if not filtered_pairs:
        raise ValueError(
            f"No requested forecasts for {code} exist in data/best_features. "
            f"Requested {[forecast for _, forecast in temporal_pairs]}, "
            f"available {sorted(valid_lags)}."
        )
    return filtered_pairs


def _temporal_pairs(
    *,
    lookback: int,
    forecast: int,
    lookbacks: list[int] | None = None,
    forecasts: list[int] | None = None,
    default_lookbacks: list[int] | None = None,
    default_forecasts: list[int] | None = None,
) -> list[tuple[int, int]]:
    if (
        lookback is None
        and forecast is None
        and lookbacks is None
        and forecasts is None
        and default_lookbacks is not None
    ):
        lookbacks = default_lookbacks
        forecasts = default_forecasts
    else:
        lookbacks = lookbacks or [lookback]
        forecasts = forecasts or [forecast]

    if forecasts is None:
        forecasts = [forecast]

    if len(lookbacks) == 1 and len(forecasts) > 1:
        lookbacks = lookbacks * len(forecasts)
    elif len(forecasts) == 1 and len(lookbacks) > 1:
        forecasts = forecasts * len(lookbacks)

    if len(lookbacks) != len(forecasts):
        raise ValueError(
            "--lookbacks and --forecasts must have the same length, "
            "unless one side contains a single value to reuse"
        )

    return list(zip(lookbacks, forecasts))


def _subprocess(command: Iterable[str], cwd: Path) -> int:
    print("+ " + " ".join(str(part) for part in command))
    return subprocess.call(list(command), cwd=str(cwd))


def cmd_aquas(args: argparse.Namespace) -> int:
    script = AQUAS_ROOT / "run_pipeline_optimized.py"
    if not script.exists():
        raise FileNotFoundError(f"AQUAS runner not found: {script}")

    forwarded = args.aquas_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        forwarded = ["--all"]

    return _subprocess([sys.executable, str(script), *forwarded], cwd=AQUAS_ROOT)


def cmd_sample_data(args: argparse.Namespace) -> int:
    script = AQUAS_ROOT / "scripts" / "create_multiyear_sample.py"
    if not script.exists():
        raise FileNotFoundError(f"AQUAS sample script not found: {script}")

    command = [
        sys.executable,
        str(script),
        "--start",
        args.start,
        "--end",
        args.end,
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    return _subprocess(command, cwd=AQUAS_ROOT)


def _common_config_kwargs(args: argparse.Namespace) -> dict:
    return {
        "code": args.code,
        "lookback": args.lookback,
        "forecast": args.forecast,
        "cutoff_date": args.cutoff_date,
        "covid_token": args.covid_token,
        "positional_encoding": args.positional_encoding,
        "evaluate_model": args.evaluate_model,
        "data_path": str(args.data_path),
        "model_folder": str(args.model_folder),
        "head_size": args.head_size,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "num_transformer_blocks": args.num_transformer_blocks,
        "mlp_units": args.mlp_units,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "activation_function": args.activation_function,
        "final_cutoff_date": args.max_date,
        "max_date": args.max_date,
    }


def _run_one_training(args: argparse.Namespace, code: str, lookback: int, forecast: int) -> None:
    _ensure_local_imports()

    from src.main_train_diagnostic_residual_transformer_class import (
        DiagnosticResidualTransformerConfig,
        DiagnosticResidualTransformerPipeline,
    )
    from src.main_train_seasonal_residual_transformer_class import (
        SeasonalResidualTransformerConfig,
        SeasonalResidualTransformerPipeline,
    )
    from src.main_train_univ_transformer_class import (
        TransformerUnivConfig,
        UnivariateTransformerPipeline,
    )

    base_kwargs = _common_config_kwargs(args)
    base_kwargs.update({"code": code, "lookback": lookback, "forecast": forecast})

    print(f"\n=== Training {args.stage}: code={code}, lookback={lookback}, forecast={forecast} ===")

    univ_outputs = None
    diag_outputs = None

    if args.stage in {"univariate", "full"}:
        univ_config = TransformerUnivConfig(**base_kwargs)
        univ_outputs = UnivariateTransformerPipeline(univ_config).run_complete_pipeline()

    if args.stage in {"diagnostic", "full"}:
        diag_kwargs = dict(base_kwargs)
        if args.diagnostic_covariates_prefix:
            diag_kwargs["diagnostic_covariates_path"] = str(args.diagnostic_covariates_prefix)
        diag_config = DiagnosticResidualTransformerConfig(
            **diag_kwargs,
            predictions_train_corrected=(
                univ_outputs.train_predictions if univ_outputs is not None else None
            ),
            predictions_test_corrected=(
                univ_outputs.test_predictions if univ_outputs is not None else None
            ),
        )
        diag_outputs = DiagnosticResidualTransformerPipeline(diag_config).run_complete_pipeline()

    if args.stage in {"seasonal", "full"}:
        seasonal_config = SeasonalResidualTransformerConfig(
            **base_kwargs,
            predictions_train_corrected=(
                diag_outputs.predictions_train_corrected if diag_outputs is not None else None
            ),
            predictions_test_corrected=(
                diag_outputs.predictions_test_corrected if diag_outputs is not None else None
            ),
        )
        SeasonalResidualTransformerPipeline(seasonal_config).run_complete_pipeline()


def cmd_train(args: argparse.Namespace) -> int:
    if args.stage == "diagnostics":
        args.stage = "diagnostic"
    codes = args.codes or [args.code]
    codes = _filter_codes_in_dataset(codes, args.data_path)
    temporal_pairs = _temporal_pairs(
        lookback=args.lookback,
        forecast=args.forecast,
        lookbacks=args.lookbacks,
        forecasts=args.forecasts,
    )

    for code in codes:
        code_temporal_pairs = _filter_temporal_pairs_by_best_features(
            code,
            temporal_pairs,
            args.diagnostic_covariates_prefix,
        )
        for lookback, forecast in code_temporal_pairs:
            _run_one_training(args, code=code, lookback=lookback, forecast=forecast)
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    _ensure_local_imports()

    import pandas as pd
    from sklearn.preprocessing import FunctionTransformer

    from production.model_reconstruction_pipeline import ModelPredictionPipeline
    from src.config.base_transformer_config import BaseTransformerConfig
    temporal_pairs = _temporal_pairs(
        lookback=args.lookback,
        forecast=args.forecast,
        lookbacks=args.lookbacks,
        forecasts=args.forecasts,
        default_lookbacks=[7, 14, 60, 60, 182, 182],
        default_forecasts=[7, 14, 30, 60, 182, 365],
    )

    if args.codes:
        codes = args.codes
    elif args.all_codes:
        codes = [
            column
            for column in _dataset_columns(args.data_path)
            if column != "timestamp" and not str(column).startswith("__index")
        ]
    else:
        codes = [args.code]
    codes = _filter_codes_in_dataset(codes, args.data_path)

    if args.prediction_dates:
        prediction_dates = args.prediction_dates
    elif args.prediction_start or args.prediction_end:
        if not args.prediction_start or not args.prediction_end:
            raise ValueError("Use both --prediction-start and --prediction-end")
        prediction_dates = [
            date.strftime("%Y-%m-%d")
            for date in pd.date_range(args.prediction_start, args.prediction_end, freq="D")
        ]
    else:
        prediction_dates = None

    final_output_df = pd.DataFrame()
    pipeline = None
    for code in codes:
        code_temporal_pairs = _filter_temporal_pairs_by_best_features(
            code,
            temporal_pairs,
            args.diagnostic_covariates_prefix,
        )
        config_kwargs = {
            "code": code,
            "data_path": str(args.data_path),
            "model_folder": str(args.model_folder),
            "lookback": code_temporal_pairs[0][0],
            "forecast": code_temporal_pairs[0][1],
            "cutoff_date": args.cutoff_date,
            "covid_token": args.covid_token,
            "positional_encoding": args.positional_encoding,
            "evaluate_model": args.evaluate_model,
            "head_size": args.head_size,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "num_transformer_blocks": args.num_transformer_blocks,
            "mlp_units": args.mlp_units,
            "activation_function": args.activation_function,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_date": args.max_date,
            "final_cutoff_date": args.max_date,
        }
        if args.diagnostic_covariates_prefix:
            config_kwargs["diagnostic_covariates_path"] = str(args.diagnostic_covariates_prefix)

        config = BaseTransformerConfig(
            **config_kwargs,
        )
        config.scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, check_inverse=False)
        pipeline = ModelPredictionPipeline(config=config)
        final_output_df = pipeline.run_reconstruct_save_results_pipeline(
            input_directory=str(args.data_path),
            old_input_directory=str(args.old_data_path or args.data_path),
            code=code,
            LOOKBACK_LIST=[lookback for lookback, _ in code_temporal_pairs],
            FORECAST_LIST=[forecast for _, forecast in code_temporal_pairs],
            final_output_predictions=None,
            final_output_df=final_output_df,
            prediction_dates=prediction_dates,
        )
        pipeline.save_final_output_predictions(final_output_df, output_path=str(args.output_dir))

    if args.delete_old and pipeline is not None:
        pipeline.delete_old_data(
            predictions_dataset_path=str(args.output_dir),
            real_data_dataset_path=str(args.data_path),
            metrics_df_path=str(args.metrics_path),
        )
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    _ensure_local_imports()

    from sklearn.preprocessing import FunctionTransformer

    from production.model_quantization_pipeline import ModelQuantizationPipeline
    from src.config.base_transformer_config import BaseTransformerConfig

    pipeline = ModelQuantizationPipeline(config=BaseTransformerConfig())
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, check_inverse=False)

    temporal_pairs = _temporal_pairs(
        lookback=args.lookback,
        forecast=args.forecast,
        lookbacks=args.lookbacks,
        forecasts=args.forecasts,
    )

    for code in args.codes:
        code_temporal_pairs = _filter_temporal_pairs_by_best_features(
            code,
            temporal_pairs,
            args.diagnostic_covariates_prefix,
        )
        for lookback, forecast in code_temporal_pairs:
            models = pipeline.run_quantization_pipeline(
                exp_names=args.experiments,
                input_directory=str(args.data_path),
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=args.cutoff_date,
                max_date=args.max_date,
                scaler=scaler,
                eliminate_covid_data=args.eliminate_covid_data,
                covid_dates=None,
                trained_model_folder=str(args.trained_model_folder),
                quantized_weights_folder=str(args.quantized_weights_folder),
                model_path=str(args.model_path) if args.model_path else None,
            )
            if args.evaluate and all(model is not None for model in models):
                pipeline.eval_quantization_impact(
                    input_directory=str(args.data_path),
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=args.cutoff_date,
                    max_date=args.max_date,
                    scaler=scaler,
                    univ_model=models[0],
                    diagnostics_model=models[1],
                    seasonal_model=models[2],
                    quant_univ_model=models[3],
                    quant_diagnostics_model=models[4],
                    quant_seasonal_model=models[5],
                    eliminate_covid_data=args.eliminate_covid_data,
                    covid_dates=None,
                )
            elif args.evaluate:
                print(
                    "-> WARNING: Skipping quantization evaluation because a complete "
                    "univariate+diagnostics+seasonal stack was not loaded."
                )
    return 0


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    model_folder_default: Path,
    model_folder_help: str,
    lookback_default: int | None = 7,
    forecast_default: int | None = 7,
) -> None:
    parser.add_argument("--code", default="DEMAND_DEMANDA_TOTAL", help="Single demand or diagnosis code to process.")
    parser.add_argument("--codes", type=_csv_strings, help="Comma-separated list of codes. Overrides --code.")
    parser.add_argument("--lookback", type=int, default=lookback_default, help="Single lookback window, in days.")
    parser.add_argument("--forecast", type=int, default=forecast_default, help="Single forecast horizon, in days.")
    parser.add_argument("--lookbacks", type=_csv_ints, help="Comma-separated lookback windows. A single --forecast value is reused when --forecasts is omitted.")
    parser.add_argument("--forecasts", type=_csv_ints, help="Comma-separated forecast horizons. A single --lookback value is reused when --lookbacks is omitted.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Input dataset used for training or reconstruction.")
    parser.add_argument("--model-folder", type=Path, default=model_folder_default, help=model_folder_help)
    parser.add_argument("--cutoff-date", default="2008-01-01", help="First date kept for training/evaluation.")
    parser.add_argument("--max-date", default="2025-12-31", help="Last date kept for training/evaluation.")
    parser.add_argument("--covid-token", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable the COVID indicator feature.")
    parser.add_argument("--positional-encoding", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable positional encoding in transformer models.")
    parser.add_argument("--evaluate-model", action=argparse.BooleanOptionalAction, default=True, help="Run model evaluation when the pipeline supports it.")
    parser.add_argument("--head-size", type=int, default=32, help="Transformer attention head size.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--ff-dim", type=int, default=512, help="Feed-forward layer dimension.")
    parser.add_argument("--num-transformer-blocks", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--mlp-units", type=_csv_ints, default=[512, 256], help="Comma-separated MLP layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--activation-function", default="gelu", help="Activation function used by transformer blocks.")


def add_reconstruct_args(parser: argparse.ArgumentParser) -> None:
    add_model_args(
        parser,
        model_folder_default=DEFAULT_QUANTIZED_MODELS_DIR,
        model_folder_help="Directory containing quantized production weights.",
        lookback_default=None,
        forecast_default=None,
    )
    parser.add_argument("--old-data-path", type=Path, help="Previous production dataset used for comparison/cleanup. Defaults to --data-path.")
    parser.add_argument("--all-codes", action="store_true", help="Read all available codes from --data-path instead of using --code/--codes.")
    parser.add_argument("--prediction-dates", type=_csv_strings, help="Comma-separated prediction dates formatted YYYY-MM-DD.")
    parser.add_argument("--prediction-start", help="Start date for a daily prediction range, formatted YYYY-MM-DD.")
    parser.add_argument("--prediction-end", help="End date for a daily prediction range, formatted YYYY-MM-DD.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "production_predictions" / "final_output_predictions", help="Directory where prediction outputs are written.")
    parser.add_argument("--metrics-path", type=Path, default=REPO_ROOT.parent / "production_predictions" / "production_evaluation_metrics.parquet", help="Metrics parquet used when deleting old production rows.")
    parser.add_argument("--diagnostic-covariates-prefix", type=Path, default=DEFAULT_BEST_FEATURES_PREFIX, help="Prefix/path for BEST_features_NOSMOOTH diagnostic covariate files.")
    parser.add_argument("--delete-old", action=argparse.BooleanOptionalAction, default=True, help="Delete/update old prediction, real-data, and metrics rows after reconstruction.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python predap_cli.py",
        formatter_class=PredapHelpFormatter,
        description=(
            "Unified CLI for the PREDAP workflow: AQUAS data retrieval, sample data, "
            "training, quantization, and production reconstruction."
        ),
        epilog="""Available methods:
  aquas         Run the bundled AQUAS data retrieval pipeline.
  sample-data   Generate local synthetic AQUAS sample data.
  train         Train univariate, diagnostic residual, seasonal residual, or full model stacks.
  quantize      Convert trained MLflow models into production weight files.
  reconstruct   Rebuild quantized models and write production predictions.
  predict       Alias of reconstruct.

Common examples:
  python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
  python predap_cli.py aquas -- --sample --all
  python predap_cli.py train --stage univariate --code TOTAL --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py quantize --experiments EXP1 --codes DEMAND_DEMANDA_TOTAL --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py predict --code TOTAL --prediction-start 2025-12-23 --prediction-end 2025-12-31

Use "python predap_cli.py <method> --help" to see all options for a method.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    aquas = subparsers.add_parser(
        "aquas",
        formatter_class=PredapHelpFormatter,
        help="Run the bundled AQUAS data retrieval pipeline.",
        description=(
            "Forward arguments to "
            "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/run_pipeline_optimized.py."
        ),
        epilog="""Examples:
  python predap_cli.py aquas -- --sample --all
  python predap_cli.py aquas -- --all --start-date 2024-01-01 --end-date 2024-12-31

Arguments after "--" are passed unchanged to AQUAS.
""",
    )
    aquas.add_argument("aquas_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the AQUAS runner.")
    aquas.set_defaults(func=cmd_aquas)

    sample = subparsers.add_parser(
        "sample-data",
        formatter_class=PredapHelpFormatter,
        help="Generate local synthetic AQUAS sample data.",
        description="Create a multi-year synthetic dataset for local development without database access.",
        epilog="""Example:
  python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
""",
    )
    sample.add_argument("--start", default="2008-01-01", help="First date to generate, formatted YYYY-MM-DD.")
    sample.add_argument("--end", default="2025-12-31", help="Last date to generate, formatted YYYY-MM-DD.")
    sample.add_argument("--input-dir", type=Path, default=AQUAS_ROOT / "data" / "sample" / "multiyear_input", help="Temporary sample input directory.")
    sample.add_argument("--output-dir", type=Path, default=AQUAS_ROOT / "data" / "sample" / "multiyear_output", help="Directory where generated sample outputs are written.")
    sample.set_defaults(func=cmd_sample_data)

    train = subparsers.add_parser(
        "train",
        formatter_class=PredapHelpFormatter,
        help="Train univariate, residual, or full model stack.",
        description=(
            "Train one or more PREDAP model stages. Use --codes for multiple series and "
            "--lookbacks/--forecasts for multiple temporal configurations."
        ),
        epilog="""Stages:
  univariate   Base transformer for the target series.
  diagnostic   Residual transformer with diagnostic covariates.
  seasonal     Residual transformer with calendar/seasonal covariates.
  full         Run univariate, diagnostic, and seasonal in sequence.

Examples:
  python predap_cli.py train --stage univariate --code TOTAL --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py train --stage full --codes TOTAL --epochs 50 --batch-size 32
""",
    )
    add_model_args(
        train,
        model_folder_default=DEFAULT_TRAINED_MODELS_DIR,
        model_folder_help="Directory where trained Keras models are saved.",
    )
    train.add_argument(
        "--stage",
        choices=["univariate", "diagnostic", "diagnostics", "seasonal", "full"],
        default="full",
        help="Training stage or complete stack to run. 'diagnostics' is accepted as an alias for 'diagnostic'.",
    )
    train.add_argument("--diagnostic-covariates-prefix", type=Path, default=DEFAULT_BEST_FEATURES_PREFIX, help="Prefix/path for BEST_features_NOSMOOTH diagnostic covariate files.")
    train.set_defaults(func=cmd_train)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        formatter_class=PredapHelpFormatter,
        help="Reconstruct quantized models and write predictions.",
        description=(
            "Load quantized model weights, reconstruct the model stack, and write "
            "production predictions for selected codes and dates."
        ),
        epilog="""Date selection:
  --prediction-dates 2025-12-23,2025-12-24
  --prediction-start 2025-12-23 --prediction-end 2025-12-31

Examples:
  python predap_cli.py reconstruct --code TOTAL --prediction-start 2025-12-23 --prediction-end 2025-12-31
  python predap_cli.py reconstruct --all-codes --lookbacks 7,14,60 --forecasts 7,14,30 --no-delete-old
""",
    )
    add_reconstruct_args(reconstruct)
    reconstruct.set_defaults(func=cmd_reconstruct)

    predict = subparsers.add_parser(
        "predict",
        formatter_class=PredapHelpFormatter,
        help="Alias of reconstruct: rebuild quantized models and write predictions.",
        description=(
            "Alias of reconstruct. Load quantized weights, rebuild the model stack, "
            "and write production predictions for selected codes and dates."
        ),
        epilog="""Date selection:
  --prediction-dates 2025-12-23,2025-12-24
  --prediction-start 2025-12-23 --prediction-end 2025-12-31

Examples:
  python predap_cli.py predict --code TOTAL --prediction-start 2025-12-23 --prediction-end 2025-12-31
  python predap_cli.py predict --all-codes --lookbacks 7,14,60 --forecasts 7,14,30 --no-delete-old
""",
    )
    add_reconstruct_args(predict)
    predict.set_defaults(func=cmd_reconstruct)

    quantize = subparsers.add_parser(
        "quantize",
        formatter_class=PredapHelpFormatter,
        help="Quantize MLflow or local models into production weight files.",
        description="Load trained MLflow runs or local model outputs and save float16 production weights.",
        epilog="""Examples:
  python predap_cli.py quantize --trained-model-folder ../trained_models --codes DEMAND_DEMANDA_TOTAL --lookbacks 7,14 --forecasts 7,14 --evaluate
  python predap_cli.py quantize --experiments EXP1 --codes DEMAND_DEMANDA_TOTAL --lookbacks 7,14 --forecasts 7,14 --evaluate
""",
    )
    quantize.add_argument("--experiments", type=_csv_strings, help="Comma-separated MLflow experiment names to search. If omitted, local models are loaded from --trained-model-folder.")
    quantize.add_argument("--trained-model-folder", type=Path, default=DEFAULT_TRAINED_MODELS_DIR, help="Local base folder containing trained Keras models organized by code.")
    quantize.add_argument("--model-path", type=Path, help="Direct path to a .keras model file or a code folder containing model subfolders to quantize.")
    quantize.add_argument("--quantized-weights-folder", type=Path, default=DEFAULT_QUANTIZED_MODELS_DIR, help="Directory where quantized weight files are written.")
    quantize.add_argument("--codes", type=_csv_strings, required=True, help="Comma-separated codes to quantize.")
    quantize.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Input dataset used for quantization checks.")
    quantize.add_argument("--lookback", type=int, default=7, help="Single lookback window, in days.")
    quantize.add_argument("--forecast", type=int, default=7, help="Single forecast horizon, in days.")
    quantize.add_argument("--lookbacks", type=_csv_ints, help="Comma-separated lookback windows. Must match --forecasts length.")
    quantize.add_argument("--forecasts", type=_csv_ints, help="Comma-separated forecast horizons. Must match --lookbacks length.")
    quantize.add_argument("--cutoff-date", default="2008-01-01", help="First date kept for quantization evaluation.")
    quantize.add_argument("--max-date", default="2025-12-31", help="Last date kept for quantization evaluation.")
    quantize.add_argument("--diagnostic-covariates-prefix", type=Path, default=DEFAULT_BEST_FEATURES_PREFIX, help="Prefix/path for BEST_features_NOSMOOTH diagnostic covariate files.")
    quantize.add_argument("--eliminate-covid-data", action=argparse.BooleanOptionalAction, default=False, help="Exclude COVID-period rows during quantization evaluation.")
    quantize.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=False, help="Evaluate original versus quantized model outputs.")
    quantize.set_defaults(func=cmd_quantize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
