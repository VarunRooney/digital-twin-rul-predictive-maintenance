import torch
import numpy as np
import pandas as pd
from torch import nn
from utils.preprocess import preprocess

# --- Define LSTM model (same architecture as training) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def predict_rul(df: pd.DataFrame) -> float:
    """
    Predict Remaining Useful Life (RUL) using the trained LSTM model.

    Args:
        df (pd.DataFrame): Sensor dataframe (temperature, vibration, pressure)

    Returns:
        float: Estimated RUL in hours (0–100 scale)
    """
    if df.empty:
        return 0.0

    # --- Preprocess input data ---
    df_norm = preprocess(df)
    input_data = df_norm.tail(3).values  # use last 3 readings
    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    # --- Load trained model ---
    model = LSTMModel()
    try:
        model.load_state_dict(torch.load("models/lstm_rul_model.pt", map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"⚠️ Model not loaded properly: {e}")
        return 0.0

    # --- Make prediction ---
    with torch.no_grad():
        pred = model(x).item()

    # --- Convert to 0–100 scale ---
    rul = max(0, min(100, 100 - abs(pred - df["temperature"].mean()) * 5))
    return float(rul)
