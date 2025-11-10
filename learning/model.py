import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from learning.dataloader import load_all_motion_data


def create_sequences(df: pd.DataFrame, input_steps=20, pred_steps=10):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "source"]]

    data = df[feature_cols].values.astype(np.float32)
    X, y = [], []

    for i in range(len(data) - input_steps - pred_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+pred_steps, 0:2])

    return np.array(X), np.array(y)


class PathPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, pred_steps=10):
        super(PathPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, pred_steps * 2)
        self.pred_steps = pred_steps

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.view(-1, self.pred_steps, 2)


def train_path_model(input_steps=20, pred_steps=10, epochs=40, batch_size=32, lr=1e-3):
    df = load_all_motion_data()
    df = df.dropna().reset_index(drop=True)
    print(f"[INFO] Using full dataset with {len(df)} rows.")

    X, y = create_sequences(df, input_steps, pred_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = PathPredictor(input_dim=X.shape[2], pred_steps=pred_steps)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_loss = criterion(test_preds, y_test_t).item()
            print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {loss.item():.6f}  Test Loss: {test_loss:.6f}")

    torch.save(model.state_dict(), "path_model.pt")
    print("[INFO] Path prediction model trained and saved → path_model.pt")

    return model

def predict_path(model, recent_sequence: np.ndarray):
    model.eval()
    x = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).numpy().squeeze()
    return pred