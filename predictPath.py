import torch
import numpy as np
from learning.model import PathPredictor
from db import SessionLocal, MotionEvent
from datetime import datetime


def load_latest_motion_data(limit=30):
    session = SessionLocal()
    try:
        rows = (
            session.query(MotionEvent)
            .order_by(MotionEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()
        data = np.array([[r.cx, r.cy, r.vx, r.vy, r.area] for r in rows])
        return data
    finally:
        session.close()


def predict_future_path(model_path="learning/path_model.pt", input_steps=20, pred_steps=10, device="cpu"):
    import os
    # NEW MODEL: output_dim=5 (single prediction), not pred_steps*5
    model = PathPredictor(input_dim=5, output_dim=5).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"[WARN] Checkpoint size mismatch: {e}")
                print("[INFO] Loading LSTM weights only; fc layer will be randomly initialized.")
                checkpoint = torch.load(model_path, map_location=device)
                # Load only LSTM parameters, skip fc layer
                lstm_state = {k: v for k, v in checkpoint.items() if k.startswith("lstm")}
                model.load_state_dict(lstm_state, strict=False)
                print("[INFO] LSTM weights loaded successfully.")
            else:
                print(f"[WARN] Could not load model checkpoint: {e}")
                print("[WARN] Using untrained model for predictions")
        except Exception as e:
            print(f"[WARN] Could not load model checkpoint: {e}")
            print("[WARN] Using untrained model for predictions")
    else:
        print(f"[WARN] Model checkpoint not found at {model_path}")
        print("[WARN] Using untrained model for predictions")
    
    model.eval()

    data = load_latest_motion_data(limit=input_steps)
    if len(data) < input_steps:
        print("[WARN] Not enough data points for prediction window.")
        return []

    X = torch.tensor(data[-input_steps:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        # NEW MODEL: Output is (1, 5) -> squeeze to (5,)
        pred = model(X).cpu().numpy().squeeze()
        # Expand to (pred_steps, 5) for backward compatibility with video.py
        # Repeat the single prediction pred_steps times
        pred = np.tile(pred, (pred_steps, 1))

    return pred


def save_predictions(event_id, preds):
    print("inside save_predictions call")
    session = SessionLocal()
    try:
        event = session.query(MotionEvent).filter_by(id=event_id).first()
        if event:
            # preds is a (pred_steps, 5) array with [cx, cy, vx, vy, area] predictions
            # Save the final predicted values
            if len(preds) > 0:
                final_pred = preds[-1]
                event.cx_pred = float(final_pred[0])
                event.cy_pred = float(final_pred[1])
                event.vx_pred = float(final_pred[2])
                event.vy_pred = float(final_pred[3])
                event.area_pred = float(final_pred[4])
                session.commit()
                print(f"[DB] Updated predictions for event id={event_id}: cx_pred={final_pred[0]:.2f}, cy_pred={final_pred[1]:.2f}, vx_pred={final_pred[2]:.2f}, vy_pred={final_pred[3]:.2f}, area_pred={final_pred[4]:.2f}")
            else:
                print(f"[DB WARN] Empty predictions for id={event_id}")
        else:
            print(f"[DB WARN] No event found for id={event_id}")
    except Exception as e:
        print(f"[DB ERROR] {e}")
        session.rollback()
    finally:
        session.close()