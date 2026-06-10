import sys
import os
import importlib
import importlib.util
from pathlib import Path


# Ensure repository root and src are on sys.path so local packages resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.config.base_transformer_config import BaseTransformerConfig
from production.model_reconstruction_pipeline import ModelPredictionPipeline
from src.utils.experiments_utils import get_codes_list

# We do not import the wrapper or optimized runner here because those modules
# perform package-level imports (e.g. `from config.config import ...`) that
# can fail depending on how this script is executed. Only import the specific
# helper we need below (`create_multiyear_sample`).

# Import script helper (create_multiyear_sample)
try:
    cms_mod = importlib.import_module("AQUAS_DATA_RETRIEVAL.scripts.create_multiyear_sample")
    create_multiyear_sample = cms_mod
except Exception:
    # As a fallback import by path
    cms_path = REPO_ROOT / "AQUAS_DATA_RETRIEVAL" / "scripts" / "create_multiyear_sample.py"
    spec = importlib.util.spec_from_file_location("create_multiyear_sample", cms_path)
    cms_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cms_mod)
    create_multiyear_sample = cms_mod

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.experiments_utils import load_json_codes_list
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)
config_name = "config_production.yaml" 

def load_hydra_config():
    @hydra.main(version_base=None, config_path="conf", config_name=config_name)
    def main(cfg: DictConfig):
        print("Configuration loaded successfully:")
        print(OmegaConf.to_yaml(cfg))
        return cfg
    
    return main()

if __name__ == "__main__":
    DEFAULT_CODES_LIST = ["DEMAND_demanda_SERVEI_CODI_INF"]
    
    LOOKBACK_LIST = [7,14, 60, 60, 182,182]
    FORECAST_LIST = [7,14, 30, 60, 182,365]
    FINAL_LOOKBACK = 182
    FINAL_FORECAST = 365

    #input_directory = '../data/FINAL_DB/full_CAT1.parquet'
    #input_directory = '../data/FINAL_DB/finals_combined.csv'
    # Point to the expected sample output file (do not read it yet — it may
    # be created by `create_multiyear_sample` below). Use a path string so
    # downstream code can decide when to read it.
    input_directory = 'AQUAS_DATA_RETRIEVAL/data/sample/multilayer_output/finals/demanda_diagnostics_joined.parquet'
    old_input_directory = '../data/FINAL_DB/finals_combined.csv'

    #model_folder = '../transformer_outputs/models_covid_token'
    output_path = f"../production_predictions/final_output_predictions"
    metrics_df_path = "../production_predictions/production_evaluation_metrics.parquet"
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, check_inverse=False)
    max_date = '2027-09-30'
    cutoff_date = '2008-01-01'
    eliminate_covid_data = False
    covid_dates = None
    model_folder = "../quantized_models"
    head_size = 32
    num_heads = 8
    ff_dim = 512
    num_transformer_blocks = 2
    mlp_units = [512,256]
    activation_function = "gelu"

    simulation_dates = pd.date_range(start='2023-10-01', end='2023-10-31', freq='D')
    for date in simulation_dates:
        input_dir = f"AQUAS_DATA_RETRIEVAL/data/sample/multilayer_input/"
        out_dir = f"AQUAS_DATA_RETRIEVAL/data/sample/multilayer_output/"
        input_directory = f"AQUAS_DATA_RETRIEVAL/data/sample/multilayer_output/finals/demand_diagnosis_joined.parquet"
        str_date = date.strftime("%Y-%m-%d")
        custom_args = [
        "--start", "2010-01-01",
        "--end", str_date,
        "--input-dir", input_dir,
        "--output-dir", out_dir
        ]

        #create_multiyear_sample.main(custom_args)

        # `create_multiyear_sample.main` expects no arguments and parses
        # `sys.argv` via argparse. Temporarily set `sys.argv` and call it.
        old_argv = sys.argv
        sys.argv = [old_argv[0]] + custom_args
        try:
            create_multiyear_sample.main()
        finally:
            sys.argv = old_argv
        CODES_LIST = get_codes_list(input_directory)
        # rEMOVE THE first "DEMAND_" charachers in front of every code to match the format expected by the reconstruction pipeline
        #CODES_LIST = [code[7:] for code in CODES_LIST ]
        CODES_LIST = [code for code in CODES_LIST if code in DEFAULT_CODES_LIST]
        final_output_df = pd.DataFrame()
        for code in DEFAULT_CODES_LIST:
            final_output_predictions = None
            
            #final_output_df = generate_future_dates_df(input_directory, num_days=FORECAST_LIST[-1])
            
            base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
                    code=code,
                    head_size=head_size,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    num_transformer_blocks=num_transformer_blocks,
                    mlp_units=mlp_units,
                    activation_function=activation_function,
                    dropout=0,
                    learning_rate=0.001,
                    epochs=50,
                    batch_size=32,
                    cutoff_date=cutoff_date,
                    covid_token=True,
                    positional_encoding=True,
                    evaluate_model=True, 
                    data_path=input_directory, 
                    model_folder=model_folder,
                ))
            final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(input_directory,old_input_directory, code, LOOKBACK_LIST, FORECAST_LIST, final_output_predictions, final_output_df)
            base_pipeline.save_final_output_predictions(final_output_df)
        base_pipeline.delete_old_data(predictions_dataset_path=output_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
        print(f"\nFinal output predictions for code {code}:\n")
    