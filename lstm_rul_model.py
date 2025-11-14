# train_model.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Load the dataset ---
df = pd.read_csv("data/engine_sensor_stream.csv")
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)

# --- Prepare sequences ---
X, y = [], []
window = 3
for i in range(len(data) - window):
    X.append(data[i:i+window])
    y.append(data[i+window, 0])  # using temperature as RUL trend

X, y = np.array(X), np.array(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- Define LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Train Model ---
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training LSTM model...")

for epoch in range(150):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch+1}/150], Loss: {loss.item():.6f}")

# --- Save trained model ---
torch.save(model.state_dict(), "models/lstm_rul_model.pt")
print("✅ Model trained and saved as models/lstm_rul_model.pt")
