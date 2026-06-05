import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_data_derivatives(df):
    """
    Compute the mean and standard deviation of the time series data for each diagnostic code.

    Parameters:
    - df (pd.DataFrame): DataFrame containing diagnostic codes as columns and time series data as rows.

    Returns:
    - pd.DataFrame: DataFrame with original data and computed derivatives.
    """
    window = 7 

    # 2. Calculate Rolling Mean and Rolling Std Dev
    df['rolling_mean'] = df['J00'].rolling(window=window).mean()
    df['rolling_std'] = df['J00'].rolling(window=window).std()

    # 3. Calculate Upper and Lower bounds (1.96 for 95% confidence)
    df['upper_95'] = df['rolling_mean'] + (1.96 * df['rolling_std'])
    df['lower_95'] = df['rolling_mean'] - (1.96 * df['rolling_std'])

    plot_derivatives(df, code='code')  # Replace 'code' with the actual column name for the diagnostic code
    
    return df

def plot_derivatives(df, code):
    """
    Plot the original time series data along with its rolling mean and confidence intervals.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the original data and computed derivatives.
    - code (str): The diagnostic code to plot.
    """
    # Assuming df['code'] is your data and you've calculated these columns:
    # df['rolling_mean'], df['upper_95'], df['lower_95']

    fig = go.Figure()

    # 1. Add the Confidence Interval (Lower Bound)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['lower_95'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Lower Bound'
    ))

    # 2. Add the Confidence Interval (Upper Bound) and Fill to Lower
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['upper_95'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty', # This creates the shaded area
        fillcolor='rgba(213, 52, 235, 0.2)',
        name='95% Confidence Interval'
    ))

    # 3. Add the Actual Data (The "Spiky" Trajectory)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['J00'],  # Replace 'J00' with the actual column name for the diagnostic code
        mode='lines',
        line=dict(color='#1f77b4', width=1.5),
        name='Actual Value'
    ))

    # 4. Add the Rolling Mean (The "Smooth" Trend)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rolling_mean'],
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        name='Trend (Rolling Mean)'
    ))

    # 5. Styling for "Useful Data for Tendency Analysis"
    fig.update_layout(
        title='Time Series Trajectory & Tendency Analysis',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified', # Shows all values for a single point in time
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()

if __name__ == "__main__":
    df = pd.read_parquet('../data/final_data/FINAL_diagnostics_CAT1.parquet')  # Example CSV file
    derivatives_df = compute_data_derivatives(df)