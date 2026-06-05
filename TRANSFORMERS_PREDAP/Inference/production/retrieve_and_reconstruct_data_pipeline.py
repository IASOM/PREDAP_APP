"""Standalone inference entrypoint for the reconstruction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from config.base_transformer_config import BaseTransformerConfig
from production.model_reconstruction_pipeline import ModelPredictionPipeline, get_codes_list
from utils.experiments_utils import load_json_codes_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the retrieval and reconstruction inference pipeline.")
    parser.add_argument("--input-directory", required=True, help="Path to the input CSV or Parquet file.")
    parser.add_argument("--old-input-directory", required=True, help="Path to the historical input CSV used by the diagnostics branch.")
    parser.add_argument("--model-folder", required=True, help="Folder containing the saved model weights.")
    parser.add_argument("--output-path", required=True, help="Directory where reconstructed predictions will be written.")
    parser.add_argument("--metrics-df-path", required=True, help="Path to the metrics parquet file.")
    parser.add_argument("--diagnostic-covariates-path", required=True, help="Prefix path for the diagnostic covariate XLSX files.")
    parser.add_argument("--code", action="append", dest="codes", help="Target code to run. Repeat to run multiple codes. Defaults to codes discovered in the input file.")
    parser.add_argument("--lookback-list", default="7,14,60,60,182,182", help="Comma-separated lookback values.")
    parser.add_argument("--forecast-list", default="7,14,30,60,182,365", help="Comma-separated forecast values.")
    parser.add_argument("--cutoff-date", default="2008-01-01", help="Lower date cutoff used during preparation.")
    parser.add_argument("--final-cutoff-date", default="2027-09-30", help="Upper date cutoff used during preparation.")
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-transformer-blocks", type=int, default=2)
    parser.add_argument("--mlp-units", default="512,256", help="Comma-separated MLP units.")
    parser.add_argument("--activation-function", default="gelu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--covid-token", action="store_true", default=True)
    parser.add_argument("--no-covid-token", action="store_false", dest="covid_token")
    parser.add_argument("--positional-encoding", action="store_true", default=True)
    parser.add_argument("--no-positional-encoding", action="store_false", dest="positional_encoding")
    return parser


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    args = build_parser().parse_args()

    lookback_list = parse_int_list(args.lookback_list)
    forecast_list = parse_int_list(args.forecast_list)
    mlp_units = parse_int_list(args.mlp_units)

    codes = args.codes if args.codes else get_codes_list(args.input_directory)
    if not codes:
        raise ValueError("No codes were provided or discovered in the input file.")

    config = BaseTransformerConfig(
        code=codes[0],
        head_size=args.head_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        mlp_units=mlp_units,
        activation_function=args.activation_function,
        dropout=args.dropout,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        cutoff_date=args.cutoff_date,
        final_cutoff_date=args.final_cutoff_date,
        covid_token=args.covid_token,
        positional_encoding=args.positional_encoding,
        evaluate_model=True,
        data_path=args.input_directory,
        model_folder=args.model_folder,
        production_predictions_dir=args.output_path,
        production_metrics_file=args.metrics_df_path,
        diagnostic_covariates_path=args.diagnostic_covariates_path,
    )

    pipeline = ModelPredictionPipeline(config=config)
    final_output_df = pd.DataFrame()
    for code in codes:
        final_output_df = pipeline.run_reconstruct_save_results_pipeline(
            input_directory=args.input_directory,
            old_input_directory=args.old_input_directory,
            code=code,
            LOOKBACK_LIST=lookback_list,
            FORECAST_LIST=forecast_list,
            final_output_predictions=None,
            final_output_df=final_output_df,
        )

    pipeline.save_final_output_predictions(final_output_df, output_path=args.output_path)
    pipeline.delete_old_data(
        predictions_dataset_path=args.output_path,
        real_data_dataset_path=args.input_directory,
        metrics_df_path=args.metrics_df_path,
    )
    print(f"Saved {len(final_output_df)} prediction rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
