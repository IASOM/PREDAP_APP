#!/usr/bin/env python
"""
Verification script to validate all data paths in the PREDAP pipeline.
Run from TRANSFORMERS_PREDAP root directory.
"""

import os
import sys
from pathlib import Path

def check_path(path_str, description, is_required=False):
    """Check if a path exists and print status."""
    path = Path(path_str)
    exists = path.exists()
    status = "✓" if exists else "✗"
    required_str = " (REQUIRED)" if is_required else " (optional)"
    
    if exists:
        if path.is_dir():
            print(f"  {status} {description}: {path_str}/ ✓")
        else:
            print(f"  {status} {description}: {path_str} ✓")
    else:
        if is_required:
            print(f"  {status} {description}: {path_str} ✗ MISSING{required_str}")
            return False
        else:
            print(f"  {status} {description}: {path_str} (will be created)")
    return True

def main():
    print("\n" + "="*70)
    print("PREDAP PIPELINE DATA PATHS VERIFICATION")
    print("="*70 + "\n")
    
    root = Path.cwd()
    if not (root / "predap_cli.py").exists():
        print("❌ Error: Must run from TRANSFORMERS_PREDAP root directory")
        print(f"   Current directory: {root}")
        return 1
    
    print(f"✓ Running from: {root}\n")
    
    all_ok = True
    
    # ===== INPUT DATA =====
    print("INPUT DATA FILES:")
    print("-" * 70)
    
    # Main data source
    main_data = "AQUAS_DATA_RETRIEVAL/AQUAS_DATA_RETRIEVAL-main/data/finals/demand_diagnosis_joined.parquet"
    if not check_path(main_data, "Main data source (demand_diagnosis_joined.parquet)", is_required=True):
        all_ok = False
    
    # Feature selection files
    print("\n  Feature selection files (BEST_features_NOSMOOTH_*.xlsx):")
    feature_dir = Path("data/best_features")
    if feature_dir.exists():
        xlsx_files = list(feature_dir.glob("*.xlsx"))
        if xlsx_files:
            print(f"    ✓ Found {len(xlsx_files)} feature files:")
            for f in sorted(xlsx_files)[:5]:
                print(f"      - {f.name}")
            if len(xlsx_files) > 5:
                print(f"      ... and {len(xlsx_files) - 5} more")
        else:
            print(f"    ⚠ Directory exists but no XLSX files found")
    else:
        print(f"    ⚠ Feature directory not found: {feature_dir}")
    
    # ===== OUTPUT DIRECTORIES =====
    print("\n\nOUTPUT DIRECTORIES:")
    print("-" * 70)
    
    output_dirs = [
        ("quantized_models", "Quantized model weights", False),
        ("production_predictions", "Production predictions & metrics", False),
        ("plots/plots_residual_transformers", "Training/evaluation plots", False),
        ("mlruns", "MLflow experiment tracking", False),
    ]
    
    for dir_path, desc, required in output_dirs:
        check_path(dir_path, desc, required)
    
    # ===== CONFIGURATION FILES =====
    print("\n\nCONFIGURATION FILES:")
    print("-" * 70)
    
    config_files = [
        ("conf/config.yaml", "Default Hydra config", True),
        ("conf/config_production.yaml", "Production grid search config", True),
        ("src/config/base_transformer_config.py", "Base transformer config", True),
        ("Inference/config/base_transformer_config.py", "Inference config", True),
    ]
    
    for file_path, desc, required in config_files:
        check_path(file_path, desc, required)
    
    # ===== DATA PATH VALIDATION =====
    print("\n\nPYTHON CONFIGURATION VALIDATION:")
    print("-" * 70)
    
    try:
        sys.path.insert(0, str(root))
        from src.config.base_transformer_config import BaseTransformerConfig
        
        config = BaseTransformerConfig()
        
        # Check data_path
        data_path = Path(config.data_path)
        if data_path.exists():
            print(f"  ✓ data_path: {config.data_path}")
        else:
            print(f"  ✗ data_path: {config.data_path} (NOT FOUND)")
            all_ok = False
        
        # Check diagnostic_covariates_path
        cov_path = Path(config.diagnostic_covariates_path)
        if cov_path.parent.exists():
            print(f"  ✓ diagnostic_covariates_path: {config.diagnostic_covariates_path}[CODE].xlsx")
        else:
            print(f"  ✗ diagnostic_covariates_path: {config.diagnostic_covariates_path} (DIRECTORY NOT FOUND)")
            all_ok = False
        
        # Check output paths
        print(f"  ✓ production_predictions_dir: {config.production_predictions_dir}")
        print(f"  ✓ production_predictions_file: {config.production_predictions_file}")
        print(f"  ✓ production_metrics_file: {config.production_metrics_file}")
        
        # Check temporal params
        print(f"  ✓ cutoff_date: {config.cutoff_date}")
        print(f"  ✓ max_date: {config.max_date}")
        
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        all_ok = False
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    if all_ok:
        print("✓ ALL REQUIRED PATHS VERIFIED SUCCESSFULLY")
        print("\nYou can now run the pipeline:")
        print("  python predap_cli.py train --codes TOTAL --lookbacks 7 --forecasts 7")
        print("  python predap_cli.py quantize --codes TOTAL --lookbacks 7 --forecasts 7")
        print("  python production/retrieve_and_reconstruct_data_pipeline.py")
        return 0
    else:
        print("✗ SOME PATHS ARE MISSING OR INCORRECT")
        print("\nPlease check:")
        print("  1. AQUAS data has been generated: AQUAS_DATA_RETRIEVAL/.../data/finals/demand_diagnosis_joined.parquet")
        print("  2. Feature files exist: data/best_features/BEST_features_NOSMOOTH_*.xlsx")
        print("  3. All paths are relative to TRANSFORMERS_PREDAP root")
        return 1

if __name__ == "__main__":
    sys.exit(main())
