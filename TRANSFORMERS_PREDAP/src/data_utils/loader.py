import pandas as pd
from src.utils.experiments_utils import smart_read

def read(path, **kwargs):
    """Smart read that supports parquet and csv via experiments_utils.smart_read."""
    try:
        return smart_read(path, **kwargs)
    except Exception:
        # Fallback to pandas
        if str(path).lower().endswith('.parquet'):
            return pd.read_parquet(path, **kwargs)
        return pd.read_csv(path, **kwargs)
