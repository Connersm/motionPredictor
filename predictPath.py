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
    model = PathPredictor(input_dim=5, pred_steps=pred_steps).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = load_latest_motion_data(limit=input_steps)
    if len(data) < input_steps:
        print("[WARN] Not enough data points for prediction window.")
        return []

    X = torch.tensor(data[-input_steps:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(X).cpu().numpy().reshape(pred_steps, 2)

    return pred


def save_predictions(event_id, preds):
    print("inside save_predictions call")
    session = SessionLocal()
    try:
        event = session.query(MotionEvent).filter_by(id=event_id).first()
        if event:
            (
                event.cx_pred,
                event.cy_pred,
                event.vx_pred,
                event.vy_pred,
                event.area_pred,
            ) = preds
            session.commit()
            print(f"[DB] Updated predictions for event id={event_id}")
        else:
            print(f"[DB WARN] No event found for id={event_id}")
    except Exception as e:
        print(f"[DB ERROR] {e}")
        session.rollback()
    finally:
        session.close()