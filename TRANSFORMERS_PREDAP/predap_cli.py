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
    parser.add_argument("--code", default="DEMAND_demanda_SERVEI_CODI_INF")
    parser.add_argument("--codes", type=_csv_strings)
    parser.add_argument("--lookback", type=int, default=7)
    parser.add_argument("--forecast", type=int, default=7)
    parser.add_argument("--lookbacks", type=_csv_ints)
    parser.add_argument("--forecasts", type=_csv_ints)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT.parent / "data" / "FINAL_DB" / "full_CAT1.parquet")
    parser.add_argument("--model-folder", type=Path, default=REPO_ROOT.parent / "quantized_models")
    parser.add_argument("--cutoff-date", default="2008-01-01")
    parser.add_argument("--covid-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positional-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-transformer-blocks", type=int, default=2)
    parser.add_argument("--mlp-units", type=_csv_ints, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--activation-function", default="gelu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predap",
        description="Unified CLI for AQUAS data retrieval and TRANSFORMERS_PREDAP model workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    aquas = subparsers.add_parser(
        "aquas",
        help="Run the bundled AQUAS data retrieval pipeline.",
        description="Forward arguments to AQUAS_DATA_RETRIEVAL/run_pipeline_optimized.py.",
    )
    aquas.add_argument("aquas_args", nargs=argparse.REMAINDER)
    aquas.set_defaults(func=cmd_aquas)

    sample = subparsers.add_parser("sample-data", help="Generate local synthetic AQUAS sample data.")
    sample.add_argument("--start", default="2010-01-01")
    sample.add_argument("--end", default="2023-10-31")
    sample.add_argument("--input-dir", type=Path, default=AQUAS_ROOT / "data" / "sample" / "multiyear_input")
    sample.add_argument("--output-dir", type=Path, default=AQUAS_ROOT / "data" / "sample" / "multiyear_output")
    sample.set_defaults(func=cmd_sample_data)

    train = subparsers.add_parser("train", help="Train univariate, residual, or full model stack.")
    add_model_args(train)
    train.add_argument("--stage", choices=["univariate", "diagnostic", "seasonal", "full"], default="full")
    train.add_argument("--diagnostic-covariates-prefix")
    train.set_defaults(func=cmd_train)

    reconstruct = subparsers.add_parser("reconstruct", help="Reconstruct quantized models and write predictions.")
    add_model_args(reconstruct)
    reconstruct.add_argument("--old-data-path", type=Path)
    reconstruct.add_argument("--all-codes", action="store_true")
    reconstruct.add_argument("--prediction-dates", type=_csv_strings)
    reconstruct.add_argument("--prediction-start")
    reconstruct.add_argument("--prediction-end")
    reconstruct.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "production_predictions" / "final_output_predictions")
    reconstruct.add_argument("--metrics-path", type=Path, default=REPO_ROOT.parent / "production_predictions" / "production_evaluation_metrics.parquet")
    reconstruct.add_argument("--delete-old", action=argparse.BooleanOptionalAction, default=True)
    reconstruct.set_defaults(func=cmd_reconstruct)

    quantize = subparsers.add_parser("quantize", help="Quantize MLflow models into production weight files.")
    quantize.add_argument("--experiments", type=_csv_strings, required=True)
    quantize.add_argument("--codes", type=_csv_strings, required=True)
    quantize.add_argument("--data-path", type=Path, default=REPO_ROOT.parent / "data" / "FINAL_DB" / "finals_combined.csv")
    quantize.add_argument("--lookback", type=int, default=7)
    quantize.add_argument("--forecast", type=int, default=7)
    quantize.add_argument("--lookbacks", type=_csv_ints)
    quantize.add_argument("--forecasts", type=_csv_ints)
    quantize.add_argument("--cutoff-date", default="2008-01-01")
    quantize.add_argument("--max-date", default="2027-09-30")
    quantize.add_argument("--eliminate-covid-data", action=argparse.BooleanOptionalAction, default=False)
    quantize.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=False)
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
