import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric sensor data for model input.

    Args:
        df (pd.DataFrame): Raw sensor dataframe with columns
            ['temperature', 'vibration', 'pressure'].

    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    if df.empty:
        return df

    numeric_cols = ["temperature", "vibration", "pressure"]

    # Remove invalid or NaN rows
    df = df[numeric_cols].dropna()

    # Normalize using simple min-max scaling
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-6)

    return df_norm
