from typing import Tuple, Optional
import pandas as pd

MAX_DATE = '2027-09-30'

def split_train_test(df: pd.DataFrame,
                     split_ratio: float = 0.8,
                     cutoff_date: Optional[str] = None,
                     max_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified train/test split utility.
    Filters by cutoff_date/max_date if provided and returns train/test split.
    """
    df = df.copy()
    if cutoff_date:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'] >= pd.to_datetime(cutoff_date)]
    if max_date:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'] <= pd.to_datetime(max_date)]
    n = int(len(df) * split_ratio)
    train = df.iloc[:n].reset_index(drop=True)
    test = df.iloc[n:].reset_index(drop=True)
    return train, test
