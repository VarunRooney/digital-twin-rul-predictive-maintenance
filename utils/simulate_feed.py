import pandas as pd
import numpy as np

def simulate_sensor_data(n_points: int = 50) -> pd.DataFrame:
    """
    Simulate live sensor readings for an engine system.
    
    Args:
        n_points (int): Number of data points to simulate.
        
    Returns:
        pd.DataFrame: Simulated sensor data (temperature, vibration, pressure).
    """
    # Random walk simulation for realistic fluctuations
    temp_base = 80 + np.random.randn(n_points).cumsum() * 0.1
    vib_base = 3 + np.random.randn(n_points).cumsum() * 0.05
    press_base = 20 + np.random.randn(n_points).cumsum() * 0.08

    # Add smooth trend (slight increase over time)
    trend = np.linspace(0, 1, n_points)
    temp = temp_base + trend * 2
    vib = vib_base + trend * 0.2
    press = press_base + trend * 0.5

    df = pd.DataFrame({
        "temperature": temp,
        "vibration": vib,
        "pressure": press
    })

    return df
