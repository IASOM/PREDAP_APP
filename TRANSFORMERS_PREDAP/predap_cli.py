#!/usr/bin/env python
"""Command line entry point for the TRANSFORMERS_PREDAP project."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
AQUAS_ROOT = REPO_ROOT / "AQUAS_DATA_RETRIEVAL" / "AQUAS_DATA_RETRIEVAL-main"
DEFAULT_DATA_PATH = AQUAS_ROOT / "data" / "sample" / "multiyear_output" / "finals" / "demand_diagnosis_joined.parquet"


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
            diag_kwargs["diagnostic_covariates_path"] = args.diagnostic_covariates_prefix
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
    codes = args.codes or [args.code]
    lookbacks = args.lookbacks or [args.lookback]
    forecasts = args.forecasts or [args.forecast]
    if len(lookbacks) != len(forecasts):
        raise ValueError("--lookbacks and --forecasts must have the same length")

    for code in codes:
        for lookback, forecast in zip(lookbacks, forecasts):
            _run_one_training(args, code=code, lookback=lookback, forecast=forecast)
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    _ensure_local_imports()

    import pandas as pd
    from sklearn.preprocessing import FunctionTransformer

    from production.model_reconstruction_pipeline import ModelPredictionPipeline
    from src.config.base_transformer_config import BaseTransformerConfig
    from src.utils.experiments_utils import get_codes_list

    lookbacks = args.lookbacks or [7, 14, 60, 60, 182, 182]
    forecasts = args.forecasts or [7, 14, 30, 60, 182, 365]
    if len(lookbacks) != len(forecasts):
        raise ValueError("--lookbacks and --forecasts must have the same length")

    if args.codes:
        codes = args.codes
    elif args.all_codes:
        codes = get_codes_list(str(args.data_path))
    else:
        codes = [args.code]

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
        config = BaseTransformerConfig(
            code=code,
            data_path=str(args.data_path),
            model_folder=str(args.model_folder),
            cutoff_date=args.cutoff_date,
            covid_token=args.covid_token,
            positional_encoding=args.positional_encoding,
            evaluate_model=args.evaluate_model,
            head_size=args.head_size,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            mlp_units=args.mlp_units,
            activation_function=args.activation_function,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        config.scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
        pipeline = ModelPredictionPipeline(config=config)
        final_output_df = pipeline.run_reconstruct_save_results_pipeline(
            input_directory=str(args.data_path),
            old_input_directory=str(args.old_data_path or args.data_path),
            code=code,
            LOOKBACK_LIST=lookbacks,
            FORECAST_LIST=forecasts,
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

    lookbacks = args.lookbacks or [args.lookback]
    forecasts = args.forecasts or [args.forecast]
    if len(lookbacks) != len(forecasts):
        raise ValueError("--lookbacks and --forecasts must have the same length")

    pipeline = ModelQuantizationPipeline(config=BaseTransformerConfig())
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    for code in args.codes:
        for lookback, forecast in zip(lookbacks, forecasts):
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
            )
            if args.evaluate:
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
                )
    return 0


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code", default="DEMAND_demanda_SERVEI_CODI_INF", help="Single demand or diagnosis code to process.")
    parser.add_argument("--codes", type=_csv_strings, help="Comma-separated list of codes. Overrides --code.")
    parser.add_argument("--lookback", type=int, default=7, help="Single lookback window, in days.")
    parser.add_argument("--forecast", type=int, default=7, help="Single forecast horizon, in days.")
    parser.add_argument("--lookbacks", type=_csv_ints, help="Comma-separated lookback windows. Must match --forecasts length.")
    parser.add_argument("--forecasts", type=_csv_ints, help="Comma-separated forecast horizons. Must match --lookbacks length.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Input dataset used for training or reconstruction.")
    parser.add_argument("--model-folder", type=Path, default=REPO_ROOT.parent / "quantized_models", help="Folder containing model outputs or quantized weights.")
    parser.add_argument("--cutoff-date", default="2008-01-01", help="First date kept for training/evaluation.")
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

Common examples:
  python predap_cli.py sample-data --start 2010-01-01 --end 2023-10-31
  python predap_cli.py aquas -- --sample --all
  python predap_cli.py train --stage univariate --code J00 --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py quantize --experiments EXP1 --codes J00,I10 --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py reconstruct --code J00 --prediction-start 2025-12-23 --prediction-end 2025-12-31

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
    sample.add_argument("--start", default="2010-01-01", help="First date to generate, formatted YYYY-MM-DD.")
    sample.add_argument("--end", default="2023-10-31", help="Last date to generate, formatted YYYY-MM-DD.")
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
  python predap_cli.py train --stage univariate --code J00 --lookbacks 7,14 --forecasts 7,14
  python predap_cli.py train --stage full --codes J00,I10,M54 --epochs 50 --batch-size 32
""",
    )
    add_model_args(train)
    train.add_argument("--stage", choices=["univariate", "diagnostic", "seasonal", "full"], default="full", help="Training stage or complete stack to run.")
    train.add_argument("--diagnostic-covariates-prefix", help="Prefix/path for BEST_features_NOSMOOTH diagnostic covariate files.")
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
  python predap_cli.py reconstruct --code J00 --prediction-start 2025-12-23 --prediction-end 2025-12-31
  python predap_cli.py reconstruct --all-codes --lookbacks 7,14,60 --forecasts 7,14,30 --no-delete-old
""",
    )
    add_model_args(reconstruct)
    reconstruct.add_argument("--old-data-path", type=Path, help="Previous production dataset used for comparison/cleanup. Defaults to --data-path.")
    reconstruct.add_argument("--all-codes", action="store_true", help="Read all available codes from --data-path instead of using --code/--codes.")
    reconstruct.add_argument("--prediction-dates", type=_csv_strings, help="Comma-separated prediction dates formatted YYYY-MM-DD.")
    reconstruct.add_argument("--prediction-start", help="Start date for a daily prediction range, formatted YYYY-MM-DD.")
    reconstruct.add_argument("--prediction-end", help="End date for a daily prediction range, formatted YYYY-MM-DD.")
    reconstruct.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "production_predictions" / "final_output_predictions", help="Directory where prediction outputs are written.")
    reconstruct.add_argument("--metrics-path", type=Path, default=REPO_ROOT.parent / "production_predictions" / "production_evaluation_metrics.parquet", help="Metrics parquet used when deleting old production rows.")
    reconstruct.add_argument("--delete-old", action=argparse.BooleanOptionalAction, default=True, help="Delete/update old prediction, real-data, and metrics rows after reconstruction.")
    reconstruct.set_defaults(func=cmd_reconstruct)

    quantize = subparsers.add_parser(
        "quantize",
        formatter_class=PredapHelpFormatter,
        help="Quantize MLflow models into production weight files.",
        description="Load trained MLflow models and save float16 production weights.",
        epilog="""Example:
  python predap_cli.py quantize --experiments EXP1 --codes J00,I10 --lookbacks 7,14 --forecasts 7,14 --evaluate
""",
    )
    quantize.add_argument("--experiments", type=_csv_strings, required=True, help="Comma-separated MLflow experiment names to search.")
    quantize.add_argument("--codes", type=_csv_strings, required=True, help="Comma-separated codes to quantize.")
    quantize.add_argument("--data-path", type=Path, default=REPO_ROOT.parent / "data" / "FINAL_DB" / "finals_combined.csv", help="Input dataset used for quantization checks.")
    quantize.add_argument("--lookback", type=int, default=7, help="Single lookback window, in days.")
    quantize.add_argument("--forecast", type=int, default=7, help="Single forecast horizon, in days.")
    quantize.add_argument("--lookbacks", type=_csv_ints, help="Comma-separated lookback windows. Must match --forecasts length.")
    quantize.add_argument("--forecasts", type=_csv_ints, help="Comma-separated forecast horizons. Must match --lookbacks length.")
    quantize.add_argument("--cutoff-date", default="2008-01-01", help="First date kept for quantization evaluation.")
    quantize.add_argument("--max-date", default="2027-09-30", help="Last date kept for quantization evaluation.")
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
