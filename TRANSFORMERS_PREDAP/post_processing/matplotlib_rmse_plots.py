"""
Matplotlib RMSE Plotting for Transformer Model Performance Analysis

This module creates matplotlib plots showing RMSE (corrected) vs Forecast Horizon
with separate lines for each lookback window, extracting data from JSON result files.
"""

import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime

def create_matplotlib_rmse_plot(results_directory: str = "c:/Users/Sira/Escritorio/predapProject/results"):
    """
    Create matplotlib plot showing RMSE (corrected) vs Forecast Horizon 
    with separate lines for each lookback window.
    
    Args:
        results_directory: Path to the results folder containing JSON files
    """
    # Set up the plot style (use matplotlib's built-in styles)
    plt.style.use('default')
    
    # Find all JSON files in results directory
    json_files = glob.glob(os.path.join(results_directory, "*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {results_directory}")
    
    data_list = []
    
    print(f"📁 Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result_data = json.load(f)
            
            # Extract model type from filename (DIAGNOSTIC or SEASONAL)
            filename = os.path.basename(json_file)
            if 'DIAGNOSTIC' in filename:
                model_type = 'DIAGNOSTIC'
            elif 'SEASONAL' in filename:
                model_type = 'SEASONAL'
            else:
                continue  # Skip if not DIAGNOSTIC or SEASONAL
            
            # Extract information from JSON data
            model_info = result_data.get('model_info', {})
            target_code = model_info.get('target_code', 'Unknown')
            forecast_horizon = model_info.get('forecast_horizon', 0)
            lookback_window = model_info.get('lookback_window', 0)
            
            # Get corrected RMSE value
            corrected_performance = result_data.get('corrected_model_performance', {})
            rmse_corrected = corrected_performance.get('RMSE', 0)
            
            # Only include if we have valid data
            if forecast_horizon > 0 and lookback_window > 0 and rmse_corrected > 0:
                data_list.append({
                    'target_code': target_code,
                    'model_type': model_type,
                    'forecast_horizon': forecast_horizon,
                    'lookback_window': lookback_window,
                    'rmse_corrected': rmse_corrected,
                    'filename': filename
                })
                
        except Exception as e:
            print(f"⚠️ Error processing {json_file}: {e}")
            continue
    
    if not data_list:
        raise ValueError("No valid data found in JSON files")
    
    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    print(f"✅ Loaded {len(df)} model results")
    print(f"📊 Target codes: {sorted(df['target_code'].unique())}")
    print(f"🔧 Model types: {sorted(df['model_type'].unique())}")
    print(f"📈 Forecast horizons: {sorted(df['forecast_horizon'].unique())}")
    print(f"📉 Lookback windows: {sorted(df['lookback_window'].unique())}")
    
    # Create separate plots for each target code and model type
    target_codes = sorted(df['target_code'].unique())
    model_types = sorted(df['model_type'].unique())
    
    for target_code in target_codes:
        for model_type in model_types:
            # Filter data for current target code and model type
            subset = df[(df['target_code'] == target_code) & (df['model_type'] == model_type)]
            
            if subset.empty:
                continue
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Get unique lookback windows for this subset
            lookback_windows = sorted(subset['lookback_window'].unique())
            
            # Create a color palette using matplotlib
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(lookback_windows), 10)))
            
            # Plot line for each lookback window
            for i, lookback in enumerate(lookback_windows):
                lookback_data = subset[subset['lookback_window'] == lookback]
                
                # Sort by forecast horizon for proper line plotting
                lookback_data = lookback_data.sort_values('forecast_horizon')
                
                plt.plot(lookback_data['forecast_horizon'], 
                        lookback_data['rmse_corrected'],
                        marker='o', 
                        linewidth=2.5,
                        markersize=8,
                        color=colors[i],
                        label=f'Lookback {lookback} days')
            
            # Customize the plot
            plt.xlabel('Forecast Horizon (days)', fontsize=14, fontweight='bold')
            plt.ylabel('RMSE (Corrected)', fontsize=14, fontweight='bold')
            plt.title(f'RMSE Performance Analysis\n{target_code} - {model_type} Models', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Add grid
            plt.grid(True, linestyle='--')
            
            # Add legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            # Add some styling
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Set x-axis to show all forecast horizons
            forecast_horizons = sorted(subset['forecast_horizon'].unique())
            plt.xticks(forecast_horizons)
            
            # Add minor ticks
            plt.minorticks_on()
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            # Save the plot
            filename = f"RMSE_vs_Forecast_{target_code}_{model_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Saved plot: {filename}")
            
            # Show the plot
            plt.show()
    
    print(f"✅ Generated RMSE plots for all available target codes and model types")


def create_combined_rmse_plot(results_directory: str = "c:/Users/Sira/Escritorio/predapProject/results"):
    """
    Create a single matplotlib plot showing RMSE (corrected) vs Forecast Horizon 
    with separate subplots for DIAGNOSTIC and SEASONAL models.
    
    Args:
        results_directory: Path to the results folder containing JSON files
    """
    # Set up the plot style
    plt.style.use('default')
    
    # Load data using similar logic as above
    json_files = glob.glob(os.path.join(results_directory, "*.json"))
    data_list = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result_data = json.load(f)
            
            filename = os.path.basename(json_file)
            if 'DIAGNOSTIC' in filename:
                model_type = 'DIAGNOSTIC'
            elif 'SEASONAL' in filename:
                model_type = 'SEASONAL'
            else:
                continue
            
            model_info = result_data.get('model_info', {})
            target_code = model_info.get('target_code', 'Unknown')
            forecast_horizon = model_info.get('forecast_horizon', 0)
            lookback_window = model_info.get('lookback_window', 0)
            
            corrected_performance = result_data.get('corrected_model_performance', {})
            rmse_corrected = corrected_performance.get('RMSE', 0)
            
            if forecast_horizon > 0 and lookback_window > 0 and rmse_corrected > 0:
                data_list.append({
                    'target_code': target_code,
                    'model_type': model_type,
                    'forecast_horizon': forecast_horizon,
                    'lookback_window': lookback_window,
                    'rmse_corrected': rmse_corrected
                })
                
        except Exception as e:
            continue
    
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("❌ No data found for plotting")
        return
    
    # Create subplots for DIAGNOSTIC and SEASONAL
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    model_types = ['DIAGNOSTIC', 'SEASONAL']
    
    for idx, model_type in enumerate(model_types):
        subset = df[df['model_type'] == model_type]
        
        if subset.empty:
            continue
        
        ax = axes[idx]
        
        # Get unique lookback windows
        lookback_windows = sorted(subset['lookback_window'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(lookback_windows), 10)))
        
        # Plot each target code with different styles
        target_codes = sorted(subset['target_code'].unique())
        linestyles = ['-', '--', '-.', ':']
        
        for tc_idx, target_code in enumerate(target_codes):
            tc_data = subset[subset['target_code'] == target_code]
            
            for lb_idx, lookback in enumerate(lookback_windows):
                lookback_data = tc_data[tc_data['lookback_window'] == lookback]
                
                if lookback_data.empty:
                    continue
                
                lookback_data = lookback_data.sort_values('forecast_horizon')
                
                ax.plot(lookback_data['forecast_horizon'], 
                       lookback_data['rmse_corrected'],
                       marker='o', 
                       linewidth=2.5,
                       markersize=6,
                       color=colors[lb_idx],
                       linestyle=linestyles[tc_idx % len(linestyles)],
                       label=f'{target_code} - LB{lookback}')
        
        # Customize subplot
        ax.set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Corrected)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_type} Models', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Style the subplot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Set x-axis ticks
        forecast_horizons = sorted(subset['forecast_horizon'].unique())
        ax.set_xticks(forecast_horizons)
    
    plt.suptitle('RMSE Performance Comparison: DIAGNOSTIC vs SEASONAL Models', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the combined plot
    filename = "RMSE_Combined_Comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved combined plot: {filename}")
    plt.show()


def main():
    """
    Main function to run the matplotlib RMSE plotting analysis
    """
    print("📈 Starting Matplotlib RMSE Analysis")
    print("=" * 60)
    
    try:
        # Create individual plots for each target code and model type
        print("📊 Creating individual RMSE plots...")
        create_matplotlib_rmse_plot()
        
        print("\n📈 Creating combined comparison plot...")
        create_combined_rmse_plot()
        
        print("\n✅ Analysis complete!")
        print("📁 All plots have been saved as PNG files")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()