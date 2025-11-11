import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal, MotionEvent

# Setup retraining log
LOG_DIR = "learning/logs"
os.makedirs(LOG_DIR, exist_ok=True)
RETRAIN_LOG_FILE = os.path.join(LOG_DIR, "retraining_log.txt")

def log_retrain_event(message: str):
    """Log retraining events to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(RETRAIN_LOG_FILE, "a") as f:
        f.write(log_message + "\n")


def load_supervised_data(limit: int = 5000):
    """Load motion data with actual and predicted values from database."""
    session = SessionLocal()
    try:
        query = session.query(MotionEvent).order_by(MotionEvent.timestamp)
        rows = query.all()
        
        data = []
        for r in rows:
            # Only include rows that have both actual values and predictions
            if r.cx is not None and r.cx_pred is not None:
                data.append({
                    "timestamp": r.timestamp,
                    "cx": r.cx,
                    "cy": r.cy,
                    "vx": r.vx,
                    "vy": r.vy,
                    "area": r.area,
                    "cx_pred": r.cx_pred,
                    "cy_pred": r.cy_pred,
                    "vx_pred": r.vx_pred,
                    "vy_pred": r.vy_pred,
                    "area_pred": r.area_pred,
                    "source": r.source,
                })
        
        df = pd.DataFrame(data)
        print(f"[INFO] Loaded {len(df)} supervised motion events from database.")
        return df
    finally:
        session.close()


def create_prediction_data_for_cells():
    """
    Create prediction data for cells that don't have it yet.
    Uses actual values as predictions for initial training data.
    Only updates rows where ANY prediction column is NULL.
    """
    from sqlalchemy import or_
    
    session = SessionLocal()
    try:
        # Find rows missing ANY prediction data (if any pred column is NULL)
        rows_updated = session.query(MotionEvent).filter(
            or_(
                MotionEvent.cx_pred.is_(None),
                MotionEvent.cy_pred.is_(None),
                MotionEvent.vx_pred.is_(None),
                MotionEvent.vy_pred.is_(None),
                MotionEvent.area_pred.is_(None)
            )
        ).update({
            MotionEvent.cx_pred: MotionEvent.cx,
            MotionEvent.cy_pred: MotionEvent.cy,
            MotionEvent.vx_pred: MotionEvent.vx,
            MotionEvent.vy_pred: MotionEvent.vy,
            MotionEvent.area_pred: MotionEvent.area,
        })
        session.commit()
        
        if rows_updated > 0:
            log_retrain_event(f"Created/updated prediction data for {rows_updated} cells")
        else:
            log_retrain_event("All cells already have complete prediction data")
        
    except Exception as e:
        session.rollback()
        log_retrain_event(f"Error creating prediction data: {e}")
    finally:
        session.close()


def clear_database_predictions():
    """Clear all prediction columns in the database."""
    session = SessionLocal()
    try:
        session.query(MotionEvent).update({
            MotionEvent.cx_pred: None,
            MotionEvent.cy_pred: None,
            MotionEvent.vx_pred: None,
            MotionEvent.vy_pred: None,
            MotionEvent.area_pred: None,
        })
        session.commit()
        log_retrain_event("Database predictions cleared")
        return True
    except Exception as e:
        session.rollback()
        log_retrain_event(f"Error clearing predictions: {e}")
        return False
    finally:
        session.close()


def detect_and_split_jumps(df: pd.DataFrame, position_threshold=50.0):
    """
    Detect big jumps in position that indicate separate objects.
    Splits sequences at jump points to treat them as separate motion tracks.
    
    Args:
        df: DataFrame with motion data
        position_threshold: distance threshold for detecting jumps
        
    Returns:
        Modified DataFrame with jump points marked as sequence boundaries
    """
    if df.empty or len(df) < 2:
        return df
    
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    # Calculate position deltas between consecutive frames
    df['cx_delta'] = df['cx'].diff().abs()
    df['cy_delta'] = df['cy'].diff().abs()
    df['position_jump'] = np.sqrt(df['cx_delta']**2 + df['cy_delta']**2)
    
    # Detect jumps
    jump_mask = df['position_jump'] > position_threshold
    num_jumps = jump_mask.sum()
    
    if num_jumps > 0:
        print(f"[INFO] Detected {num_jumps} position jumps (threshold={position_threshold})")
        jump_indices = df[jump_mask].index.tolist()
        print(f"[INFO] Jump points at indices: {jump_indices[:10]}")  # Show first 10
        
        # Mark jump points for sequence splitting
        df['is_jump'] = jump_mask.astype(int)
        
        # Return dataframe with jump information
        return df, jump_indices
    
    df['is_jump'] = 0
    return df, []


def clean_supervised_data(df: pd.DataFrame, detect_jumps=True, position_threshold=50.0):
    """Clean data for supervised learning: actual vs predicted."""
    if df.empty:
        print("[WARN] No data loaded from database.")
        return None, None, None
    
    print(f"[INFO] Initial dataset size: {len(df)} rows")
    
    # Detect big jumps (separate objects) BEFORE cleaning
    if detect_jumps:
        df, jump_indices = detect_and_split_jumps(df, position_threshold=position_threshold)
    else:
        df['is_jump'] = 0
        jump_indices = []
    
    # Remove duplicates and NaNs
    df = df.dropna(how="any").drop_duplicates(subset=["timestamp", "source"], keep="last")
    print(f"[INFO] After removing NaNs/duplicates: {len(df)} rows")
    
    # Sort by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Define feature columns
    actual_cols = ["cx", "cy", "vx", "vy", "area"]
    pred_cols = ["cx_pred", "cy_pred", "vx_pred", "vy_pred", "area_pred"]
    
    # Handle any remaining NaNs with interpolation
    df[actual_cols + pred_cols] = df[actual_cols + pred_cols].interpolate(
        method="linear", limit_direction="both"
    )
    
    # Impute remaining NaNs with median
    imputer = SimpleImputer(strategy="median")
    df[actual_cols + pred_cols] = imputer.fit_transform(df[actual_cols + pred_cols])
    
    # Standardize both actual and predicted features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[actual_cols + pred_cols])
    
    df_scaled = pd.DataFrame(
        scaled_data,
        columns=actual_cols + pred_cols
    )
    df_scaled["timestamp"] = df["timestamp"].values
    df_scaled["source"] = df["source"].values
    
    print(f"[INFO] Data cleaning complete. Final size: {len(df_scaled)} rows")
    
    return df_scaled, scaler, (actual_cols, pred_cols)


def create_supervised_sequences(df: pd.DataFrame, actual_cols, pred_cols, input_steps=20):
    """
    Create sequences for supervised learning.
    X: historical actual values (input_steps history of actual cx, cy, vx, vy, area)
    y: predicted values at current timestep (cx_pred, cy_pred, vx_pred, vy_pred, area_pred)
    
    Respects jump boundaries (separate objects) - doesn't create sequences across jumps.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    X, y = [], []
    
    # If no jump information, use standard sequence creation
    if 'is_jump' not in df.columns:
        for i in range(len(df) - input_steps):
            X.append(df[actual_cols].iloc[i:i+input_steps].values)
            y.append(df[pred_cols].iloc[i+input_steps].values)
    else:
        # Create sequences respecting jump boundaries
        # Find sequences where no jump occurs
        for i in range(len(df) - input_steps):
            window = df.iloc[i:i+input_steps+1]
            
            # Skip if any jump point is in the window (except at the start)
            if window['is_jump'].iloc[1:].sum() > 0:
                continue
            
            # This is a valid sequence
            X.append(df[actual_cols].iloc[i:i+input_steps].values)
            y.append(df[pred_cols].iloc[i+input_steps].values)
    
    if len(X) == 0:
        print("[WARN] No valid sequences created after filtering jumps")
        return np.array([]), np.array([])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class PathPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=5):
        super(PathPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


def count_database_entries():
    """Count total motion events in database."""
    session = SessionLocal()
    try:
        count = session.query(MotionEvent).count()
        return count
    finally:
        session.close()


def train_path_model(input_steps=20, epochs=40, batch_size=32, lr=1e-3, data_limit=5000, 
                     auto_retrain=True, retrain_threshold=3000, detect_jumps=True, 
                     position_threshold=50.0):
    """
    Train supervised LSTM model to predict motion from historical motion data.
    
    Args:
        input_steps: number of historical timesteps to use as input
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        data_limit: max rows to load from database
        auto_retrain: if True, automatically retrain when threshold is reached
        retrain_threshold: number of entries before auto-retraining (default 3000)
        detect_jumps: if True, detect big position jumps as separate objects
        position_threshold: distance threshold for detecting jumps (default 50.0)
    """
    print("[INFO] Starting training pipeline...")
    log_retrain_event("=" * 80)
    log_retrain_event("RETRAINING SESSION STARTED")
    log_retrain_event("=" * 80)
    
    # Check if auto-retrain is needed
    if auto_retrain:
        db_count = count_database_entries()
        log_retrain_event(f"Database has {db_count} entries (retrain threshold: {retrain_threshold})")
        
        if db_count >= retrain_threshold:
            log_retrain_event(f"\n*** AUTO-RETRAIN TRIGGERED ***")
            log_retrain_event(f"Database has reached {db_count} entries (>= {retrain_threshold})")
            log_retrain_event(f"Clearing prediction history...")
            clear_database_predictions()
            log_retrain_event(f"Ready for next prediction cycle with fresh model\n")
            return None, None  # Exit after clearing
    
    # Create prediction data for cells that don't have it
    log_retrain_event(f"\nChecking for cells without prediction data...")
    create_prediction_data_for_cells()
    
    # Step 1: Load data from database
    log_retrain_event(f"\n[STEP 1] Loading supervised data from database...")
    df = load_supervised_data(limit=data_limit)
    
    if df.empty or len(df) < input_steps + 1:
        log_retrain_event("[ERROR] Not enough data for training.")
        return None, None
    
    # Step 2: Clean data (with jump detection)
    log_retrain_event(f"[STEP 2] Cleaning data and detecting jumps...")
    df_clean, scaler, (actual_cols, pred_cols) = clean_supervised_data(
        df, detect_jumps=detect_jumps, position_threshold=position_threshold
    )
    
    if df_clean is None or len(df_clean) < input_steps + 1:
        log_retrain_event("[ERROR] Not enough clean data for training.")
        return None
    
    # Step 3: Create sequences
    log_retrain_event(f"[STEP 3] Creating sequences (input_steps={input_steps})...")
    X, y = create_supervised_sequences(df_clean, actual_cols, pred_cols, input_steps=input_steps)
    log_retrain_event(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) < batch_size * 2:
        new_batch_size = max(1, len(X) // 4)
        log_retrain_event(f"[WARN] Limited data ({len(X)} samples). Adjusting batch_size from {batch_size} to {new_batch_size}")
        batch_size = new_batch_size
    
    # Step 4: Train/test split
    log_retrain_event(f"[STEP 4] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    log_retrain_event(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 5: Initialize model
    log_retrain_event(f"[STEP 5] Initializing model...")
    input_dim = X.shape[2]  # Number of features (5: cx, cy, vx, vy, area)
    output_dim = y.shape[1]  # Number of prediction targets (5)
    model = PathPredictor(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    log_retrain_event(f"[INFO] Model: input_dim={input_dim}, output_dim={output_dim}")
    
    # Step 6: Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Step 7: Train model
    log_retrain_event(f"[STEP 6] Training for {epochs} epochs...")
    best_test_loss = float('inf')
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
            
            status = f"[EPOCH {epoch+1}/{epochs}] Train Loss: {loss.item():.6f}  Test Loss: {test_loss:.6f}"
            print(status)
            
            # Track best test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss
    
    # Step 8: Save model
    log_retrain_event(f"\n[STEP 7] Saving model...")
    torch.save(model.state_dict(), "learning/path_model.pt")
    log_retrain_event(f"[INFO] ✓ Model trained and saved → learning/path_model.pt")
    log_retrain_event(f"[INFO] Final Test Loss: {best_test_loss:.6f}")
    log_retrain_event(f"\nRETRAINING SESSION COMPLETED SUCCESSFULLY")
    log_retrain_event("=" * 80)
    
    return model, scaler


def predict_path(model, recent_sequence: np.ndarray):
    """
    Predict next motion values given recent motion history.
    
    Args:
        model: trained PathPredictor model
        recent_sequence: numpy array of shape (input_steps, 5) with recent motion data
        
    Returns:
        numpy array of shape (5,) with predicted [cx, cy, vx, vy, area]
    """
    model.eval()
    x = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze()
    return pred