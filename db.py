# db.py
# Author: Conner Santa Monica
# Description: PostgreSQL ORM layer for motion data persistence on AWS RDS.

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?sslmode={DB_SSLMODE}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class MotionEvent(Base):
    __tablename__ = "motion_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    cx = Column(Float)
    cy = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    area = Column(Float)
    source = Column(String(32))


Base.metadata.create_all(bind=engine)


def save_motion_event(timestamp, cx, cy, vx, vy, area, source):
    db = SessionLocal()
    try:
        event = MotionEvent(
            timestamp=datetime.fromtimestamp(timestamp),
            cx=cx,
            cy=cy,
            vx=vx,
            vy=vy,
            area=area,
            source=source,
        )
        db.add(event)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] {e}")
    finally:
        db.close()
