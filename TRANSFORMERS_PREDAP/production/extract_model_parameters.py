import mlflow
import mlflow.pyfunc
import pandas as pd
import os

def load_mlflow_model_parameters(exp_names, model_names, output_filename="../models_parameters/model_parameters.csv"):
    new_runs_data = []

    for model_name in model_names:
        for exp_name in exp_names:
            filter_string = f"attributes.run_name LIKE '{model_name}%'"
            runs = mlflow.search_runs(
                experiment_names=[exp_name],
                filter_string=filter_string,
                order_by=["start_time DESC"],
                max_results=1,
            )

            if not runs.empty:
                # 1. Extract parameters into a clean dictionary
                param_cols = [col for col in runs.columns if col.startswith("params.")]
                model_params = {
                    col.replace("params.", ""): runs.iloc[0][col] for col in param_cols
                }

                # 2. Add metadata directly to the dictionary *before* turning it into a DataFrame row
                model_params['experiment_name'] = exp_name
                model_params['run_id'] = runs.iloc[0]['run_id']
                model_params['model_name'] = model_name
                
                new_runs_data.append(model_params)
            else:
                print(f"No matching runs found for model '{model_name}' in experiment '{exp_name}'.")

    # If we didn't find any runs across all loops, we can exit early
    if not new_runs_data:
        print("No new runs found to add.")
        if os.path.exists(output_filename):
            return pd.read_csv(output_filename)
        return pd.DataFrame()

    # 3. Create DataFrame from all newly found runs
    df_new = pd.DataFrame(new_runs_data)

    # 4. Handle appending to the existing file out here
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename)
            updated_df = pd.concat([existing_df, df_new], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing file, creating new one instead. Error: {e}")
            updated_df = df_new
    else:
        updated_df = df_new

    # 5. Drop duplicates strictly based on unique run IDs so histories don't collide
    updated_df = updated_df.drop_duplicates(subset=['run_id'], keep='last', inplace=False)
    
    # 6. Save once
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    updated_df.to_csv(output_filename, index=False)
    
    print(f"\nParameters successfully updated/saved to {output_filename}")
    print(updated_df)
    
    return updated_df


    


if __name__ == "__main__":
    model_uri = "models:/your_model_name/production"  # Replace with your model's URI
    exp_names = ['1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260520', '1.0_Production_TRANSFORMER_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260518']

    #model_name = '1.0_Production_TRANSFORMER_demanda__SERVEI_CODI__INF_lb182_fh365_092911'
    model_names = ['1.0_Production_TRANSFORMER_demanda__SERVEI_CODI__INF_lb182_fh365_092911','1.0_Production_TRANSFORMER_CHAPTER_chapter_01_lb60_fh30_075400']
    parameters = load_mlflow_model_parameters(exp_names, model_names)
    print("Extracted Model Parameters:")
    print(parameters)