# This file is an script to find those more representative diagnostic codes in different precentiles of the data 
import pandas as pd 
import numpy as np

def find_code_percentile_diagnostics(data, percentile=95, top_n= 1):
    """
    Find diagnostic codes that fall within the specified percentile of total counts.

    Parameters:
    - data (pd.DataFrame): DataFrame containing diagnostic codes and their counts.
    - percentile (float): Percentile threshold to filter diagnostic codes.

    Returns:
    - List[str]: List of diagnostic codes within the specified percentile.
    """
    import pandas as pd

    sorted_data = data.iloc[0].sort_values(ascending=False)
    threshold = np.percentile(sorted_data.values, percentile)
    representative_codes = sorted_data[sorted_data <= threshold].index.tolist()
    most_representative_code = representative_codes[:top_n]
    return most_representative_code

def find_more_prevalent_diagnostics(data, top_n=1):
    """
    Find the top N most prevalent diagnostic codes based on their counts.

    Parameters:
    - data (pd.DataFrame): DataFrame containing diagnostic codes and their counts.
    - top_n (int): Number of top prevalent diagnostic codes to return.

    Returns:
    - List[str]: List of the top N most prevalent diagnostic codes.
    """
    import pandas as pd

    # Sort the data by count in descending order and select the top N codes
    prevalent_codes = data.sort_values(by='count', ascending=False).head(top_n)['diagnostic_code'].tolist()

    return prevalent_codes

def aggregate_dataframe(df):
    """
    Aggregate the DataFrame by diagnostic code, summing the counts.

    Parameters:
    - df (pd.DataFrame): DataFrame containing diagnostic codes and their counts.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with summed counts per diagnostic code.
    """
    import pandas as pd

    agg_df = pd.DataFrame()
    df = df.drop(columns=['timestamp'])

    for code in df.columns:
        agg_df[code] = [np.sum(df[code].values)]
    
    return agg_df

if __name__ == "__main__":
    df = pd.read_parquet('../data/final_data/FINAL_diagnostics_CAT1.parquet')  # Example CSV file
    aggregated_df = aggregate_dataframe(df)
    representative_codes = find_code_percentile_diagnostics(aggregated_df, percentile=95)
    
    
    print("Aggregated DataFrame:")
    print(aggregated_df)

    percentiles = [10, 25, 50, 75, 90]

    for perc in percentiles:
        codes_in_percentile = find_code_percentile_diagnostics(aggregated_df, percentile=perc, top_n=5)
        print(f"Diagnostic codes in the {perc}th percentile: {codes_in_percentile}")