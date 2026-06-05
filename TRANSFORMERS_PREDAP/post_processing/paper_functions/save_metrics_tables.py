from http import client
import mlflow
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def dict_to_df(data):
    rows = []
    # If 'data' is a dictionary of DataFrames (from the new save_metrics_tables)
    # we don't need to loop manually, just concatenate them.
    if isinstance(next(iter(data.values())), pd.DataFrame):
        return pd.concat(data.values(), ignore_index=True)

    # If it's the old nested dictionary structure:
    for model, phases in data.items():
        for phase, steps in phases.items():
            for step, metrics in steps.items():
                row = {
                    'Model': model,
                    'Phase': phase,
                    'Step': int(step)
                }
                
                # FIX: Check if metrics is actually a dictionary before updating
                if isinstance(metrics, dict):
                    row.update(metrics) 
                else:
                    # If metrics is a single value (like just a float), 
                    # we give it a default key or skip
                    row['metric_value'] = metrics
                    
                rows.append(row)
    return pd.DataFrame(rows)
# --- TASK 1: WAPE vs Steps per Phase ---
def plot_wape_comparison(df):
    phases = df['Phase'].unique()
    # sharey=True ensures all plots are on the same scale for direct comparison
    fig, axes = plt.subplots(1, len(phases), figsize=(18, 5), sharey=True)
    
    if len(phases) == 1:
        axes = [axes]

    for i, phase in enumerate(phases):
        phase_df = df[df['Phase'] == phase]
        sns.lineplot(ax=axes[i], data=phase_df, x='Step', y='wape', hue='Model', marker='o')
        
        # Re-enable the Y-axis labels which are hidden by default when sharey=True
        axes[i].tick_params(labelleft=True)
        
        axes[i].set_title(f'Phase: {phase.capitalize()}')
        axes[i].set_ylabel('WAPE')
        axes[i].set_xlabel('Forecast Step')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('wape_comparison.png')
    plt.show()


def plot_model_comparison_by_phase(df):
    # Now we unique-ify 'Model' instead of 'Phase' for the subplots
    models = df['Model'].unique()
    
    # Create subplots based on the number of models
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5), sharey=True)
    
    # Ensure axes is iterable even if there is only one model
    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        # Filter for the specific model
        model_df = df[df['Model'] == model]
        
        # Plot all phases within this model's subplot
        # We use 'Phase' as the hue to see them side-by-side
        sns.lineplot(ax=axes[i], data=model_df, x='Step', y='wape', hue='Phase', marker='o')
        
        # Consistent with your previous request: keep Y-axis labels on all plots
        axes[i].tick_params(labelleft=True)
        
        axes[i].set_title(f'Model: {model}')
        axes[i].set_ylabel('WAPE')
        axes[i].set_xlabel('Forecast Step')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
        # Move legend to a consistent spot (optional: best, upper right, etc.)
        axes[i].legend(title='Phase')

    plt.tight_layout()
    plt.savefig('model_comparison_by_phase.png')
    plt.show()
    
def plot_wape_unified(df):
    plt.figure(figsize=(12, 7))
    
    # style='Phase' gives different line patterns for each phase
    # hue='Model' gives different colors for each model
    # markers=True adds distinct points for clarity
    ax = sns.lineplot(
        data=df, 
        x='Step', 
        y='wape', 
        hue='Model', 
        style='Phase', 
        markers=True, 
        dashes=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title('WAPE Comparison: Models vs. Phases', fontsize=15)
    plt.ylabel('WAPE (Error)', fontsize=12)
    plt.xlabel('Forecast Horizon (Steps)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Move legend outside the plot for better visibility
    plt.legend(title='Model & Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('wape_unified_comparison.png')  # Save the figure for later reference
    plt.show()

# --- TASK 2: Metrics Table ---
# We pivot this to make it readable: Model and Step as index, Metrics as columns
def display_metrics_table(df):
    # Grouping by Phase first to keep it organized
    for phase in df['Phase'].unique():
        print(f"\n--- Metrics Table for Phase: {phase.upper()} ---")
        
        # Filter for the phase
        phase_df = df[df['Phase'] == phase]
        
        # Use pivot_table instead of pivot
        # aggfunc='mean' averages the values if duplicates exist
        phase_table = phase_df.pivot_table(
            index=['Step'], 
            columns='Model', 
            values=['mae', 'wape'],
            aggfunc='mean'
        )
        
        # Displaying just MAE and WAPE for brevity in console
        print(phase_table.round(4))

# --- TASK 3: Heatmap of RMSE ---
# Heatmaps are excellent for seeing exactly where a model starts to "break" (long-term vs short-term)
def plot_rmse_heatmap(df):
    plt.figure(figsize=(10, 6))
    # Averaging across phases for a global model health check
    heatmap_data = df.pivot_table(index='Model', columns='Step', values='rmse', aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt=".3f")
    plt.title('Global RMSE Heatmap (Model vs. Horizon)')
    plt.savefig('rmse_heatmap.png')  # Save the figure for later reference
    plt.show()




def extract_experiment_type(experiment_name):
    """
    Extract the experiment type (e.g., 'informer', 'lstnet', 'univ_transformer') from the experiment name.
    Assumes the experiment name follows a pattern like 'full_INFORMER_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260205'.
    """
    match = re.search(r'fh(\d+)_', experiment_name)

    if match:
        fh_number = match.group(1)
    else:
        fh_number = "unknown_fh"

    exp_type = None
    if "INFORMER" in experiment_name:
        exp_type = "informer"
    elif "LSTNET" in experiment_name:
        exp_type = "lstnet"
    elif "TRANSFORMER3" in experiment_name or "TRANSFORMERS3" in experiment_name:
        exp_type = "our_transformer"

    elif "LOG_TRANSFORMER" in experiment_name:
        exp_type = "log_transformer"
    else:
        exp_type = "unknown_model"

    return exp_type, fh_number


'''def save_metrics_to_csv(df, codes, prefix="experiment"):
    """
    Filters the main DataFrame by Phase and saves a pivoted table 
    for each phase into separate CSV files.
    """
    phases = df['Phase'].unique()
    saved_files = []

    for phase in phases:
        # 1. Filter data for the specific phase
        phase_df = df[df['Phase'] == phase]
        
        # 2. Pivot the table so Steps are rows and Models/Metrics are columns
        # This creates a structured table: Rows = Steps, Columns = (Metric, Model)
        pivoted_table = phase_df.pivot(index='Step', columns='Model', values=['mae', 'mse', 'rmse', 'wape'])
        
        # 3. Generate a filename
        filename = f"{prefix}_{phase}_metrics.csv"
        
        # 4. Save to CSV
        pivoted_table.to_csv(filename)
        saved_files.append(filename)
        print(f"Successfully saved: {filename}")
    
    return saved_files
'''

def save_metrics_tables(codes, metrics_dict, experiment_names):
    """
    Groups MLflow runs by 'codes', processes metrics into a clean DataFrame,
    and saves separate CSVs for each code.
    """
    # 1. Fetch all runs from the specified experiments
    experiments = [mlflow.get_experiment_by_name(exp) for exp in experiment_names]
    experiment_ids = [e.experiment_id for e in experiments if e is not None]
    
    if not experiment_ids:
        print("No valid experiments found.")
        return None

    # Load all runs into one master DataFrame
    master_df = mlflow.search_runs(experiment_ids=experiment_ids)
    model_names_col = 'tags.mlflow.runName'
    
    # Dictionary to store a DataFrame for each code
    results_by_code = {}

    for code in codes:
        print(f"Processing group: {code}")
        
        # 2. Filter runs that contain the specific code
        code_mask = master_df[model_names_col].str.contains(code, na=False)
        code_df = master_df[code_mask].copy()
        
        if code_df.empty:
            print(f"No runs found for code: {code}")
            continue

        # 3. Extract metadata and build the long-form records
        rows = []
        model_phases = ['univariate', 'diagnostics', 'seasonal']

        for _, run in code_df.iterrows():
            run_name = run[model_names_col]
            model_type, fh_number = extract_experiment_type(run_name)
            
            for phase in model_phases:
                # Get the list of metrics we expect for this phase
                expected_metrics = metrics_dict.get(phase, [])
                
                # Build metric dictionary for this specific run and phase
                current_metrics = {}
                for metric_col in expected_metrics:
                    # MLflow metrics columns are usually prefixed with 'metrics.'
                    # or they might be direct column names in the search_runs df
                    val = run.get(metric_col)
                    metric_key = metric_col.split('_')[-1] # e.g. 'val_mae' -> 'mae'
                    current_metrics[metric_key] = val

                # Append to our rows list (The "Pandas-like" structure)
                rows.append({
                    'Code': code,
                    'Model': model_type,
                    'Phase': phase,
                    'Step': int(fh_number) if str(fh_number).isdigit() else fh_number,
                    **current_metrics
                })

        # 4. Convert to DataFrame and store
        final_df = pd.DataFrame(rows)
        results_by_code[code] = final_df

        # 5. Save to CSV using your existing logic (one file per phase per code)
        save_metrics_to_csv(final_df, code, prefix=code)

    return results_by_code

def save_metrics_to_csv(df, code, prefix="experiment"):
    """
    Saves pivoted tables for each phase within a code group, 
    aggregating duplicates if they exist.
    """
    phases = df['Phase'].unique()
    for phase in phases:
        phase_df = df[df['Phase'] == phase]
        if phase == 'univariate':
            df_univ = phase_df[phase_df['Phase'] == 'univariate']

            pivoted = phase_df.pivot_table(
            index='Step', 
            columns='Model', 
            values=['mae', 'mse', 'rmse', 'wape'],
            aggfunc='mean' 
            )
            pivoted = phase_df.pivot_table(
            index='Step', 
            columns='Model', 
            values=['mae', 'mse', 'rmse', 'wape'],
            aggfunc='mean' 
            )

            # Use pivot_table instead of pivot to handle duplicates
            # aggfunc='mean' will average metrics if a model/step combo appears twice
            
            filename = f"{prefix}_{phase}_metrics.csv"
            pivoted.to_csv(filename)
            print(f"Saved: {filename}")

        else:
            mae_improve = (df_univ['mae'].values - phase_df['mae'].values) / df_univ['mae'].values
            name = f"{phase}_mae_improve"
            phase_df[name] = mae_improve


            pivoted = phase_df.pivot_table(
            index='Step', 
            columns='Model', 
            values=['mae', 'mse', 'rmse', 'wape', f'{phase}_mae_improve'],
            aggfunc='mean' 
            )

            # Use pivot_table instead of pivot to handle duplicates
            # aggfunc='mean' will average metrics if a model/step combo appears twice
            
            filename = f"{prefix}_{phase}_metrics.csv"
            pivoted.to_csv(filename)
            print(f"Saved: {filename}")




if __name__ == "__main__":
    # Example usage
    metrics_dict = {
        'train_mae': 0.1,
        'val_mae': 0.2,
        'train_mse': 0.01,
        'val_mse': 0.04
    }

    codes = ["J00", "demanda__TOTAL", "demanda__SERVEI_CODI__URG", "B34", "I10", "M54", "Ch01#subch01#A00-A09"]

    metrics_dict = {
                    'univariate':
                    ['metrics.eval/univ_transformer_mae',
                    'metrics.eval/univ_transformer_mse',
                    'metrics.eval/univ_transformer_rmse',
                    'metrics.eval/univ_transformer_wape',],

                    'diagnostics':
                    ['metrics.eval/residual_diagnostics_model_mae',
                    'metrics.eval/residual_diagnostics_model_mse',
                    'metrics.eval/residual_diagnostics_model_rmse',
                    'metrics.eval/residual_diagnostics_model_wape'],
                    
                    'seasonal':
                    ['metrics.final/seasonal_mae',
                    'metrics.final/seasonal_mse',
                    'metrics.final/seasonal_rmse',
                    'metrics.final/seasonal_wape']

                 }
    experiment_names =["full_TRANSFORMER3_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210",
                       "full_TRANSFORMER3_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212",
                    "full_INFORMER1_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260209",
                    "full_INFORMER1_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210",
                    "full_LOG_TRANSFORMER1_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260209",
                    "full_LOG_TRANSFORMER1_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212",
                    "full_LOG_TRANSFORMER2_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260209",
                    "full_LOG_TRANSFORMER2_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210",
                    "full_LSTNET3_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210",
                    "full_LSTNET3_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260211",
                    "full_LSTNET3_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212",
                    #"full_LOG_TRANSFORMER1_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212",

                        ] 
    # ['full_NO_DIAGNOSTICS1_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260211']
                        

    models_dict = save_metrics_tables(codes,  metrics_dict, experiment_names=experiment_names)
    # Initialize and Process
    # (Using your provided models_dict structure)
    # models_dict = { ... } 
    
    df = dict_to_df(models_dict)

    plot_wape_comparison(df)
    plot_wape_unified(df)
    display_metrics_table(df)
    plot_rmse_heatmap(df)
    plot_model_comparison_by_phase(df)
    save_metrics_to_csv(df, code=codes, prefix="final_experiment")
    print(models_dict)



    