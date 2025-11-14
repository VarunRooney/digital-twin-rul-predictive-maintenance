import pandas as pd

def load_engine_data(path: str) -> pd.DataFrame:
    """
    Load engine sensor data from CSV file.
    
    Args:
        path (str): File path to the CSV dataset.
    
    Returns:
        pd.DataFrame: Sensor readings with temperature, vibration, and pressure.
    """
    try:
        df = pd.read_csv(path)
        if not {"temperature", "vibration", "pressure"}.issubset(df.columns):
            raise ValueError("CSV file missing required columns.")
        return df
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        return pd.DataFrame(columns=["temperature", "vibration", "pressure"])
