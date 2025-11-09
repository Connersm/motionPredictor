import pandas as pd
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import SessionLocal, MotionEvent
from dataengineering import engineer_features


def load_motion_data(limit: int = 1000):
    session: Session = SessionLocal()
    try:
        query = session.query(MotionEvent).order_by(MotionEvent.timestamp.desc()).limit(limit)
        data = pd.read_sql(query.statement, session.bind)
        return data
    finally:
        session.close()


def clean_motion_data(df: pd.DataFrame):
    if df.empty:
        print("[WARN] No data loaded from database.")
        return df, None

    df = df.dropna(how="all").drop_duplicates(subset=["timestamp", "source"], keep="last")

    df = df.sort_values(by="timestamp").reset_index(drop=True)

    numeric_cols = ["cx", "cy", "vx", "vy", "area"]

    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )

    df_scaled["timestamp"] = df["timestamp"].values
    df_scaled["source"] = df["source"].values

    if df_scaled[numeric_cols].isna().sum().sum() > 0:
        print("[WARN] Remaining NaNs detected after imputation.")

    return df_scaled, scaler




def get_prepared_data(limit: int = 1000):
    raw_df = load_motion_data(limit)
    cleaned_df, scaler = clean_motion_data(raw_df)

    engineered_df = engineer_features(cleaned_df)

    return engineered_df, scaler
