import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal, MotionEvent
import pandas as pd

def load_all_motion_data():
    session = SessionLocal()
    try:
        query = session.query(MotionEvent).order_by(MotionEvent.timestamp)
        rows = query.all()
        data = [
            {
                "timestamp": r.timestamp,
                "cx": r.cx,
                "cy": r.cy,
                "vx": r.vx,
                "vy": r.vy,
                "area": r.area,
                "source": r.source,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        print(f"[INFO] Loaded {len(df)} motion events from database.")
        return df
    finally:
        session.close()