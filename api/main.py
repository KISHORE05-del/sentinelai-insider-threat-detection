"""
SentinelAI - API Layer (FastAPI)
---------------------------------
Serves risk scores and alerts computed by the trained models.
Reads from the SQLite database (populated by db/load_to_db.py).

Run with:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "sentinelai.db")

app = FastAPI(
    title="SentinelAI API",
    description="AI-based insider threat detection - risk scores & alerts",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskRecord(BaseModel):
    employee_id: int
    department: str
    date: str
    risk_score: float
    is_odd_hour_login: int
    high_usb_activity: int
    large_upload: int
    bulk_file_access: int
    label_is_anomaly: Optional[int] = None


def get_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=500,
            detail="Database not found. Run db/load_to_db.py first to populate it.",
        )
    return sqlite3.connect(DB_PATH)


@app.get("/")
def root():
    return {"status": "ok", "service": "SentinelAI API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/alerts", response_model=list[RiskRecord])
def get_alerts(
    threshold: float = Query(50.0, description="Minimum risk score to be considered an alert"),
    limit: int = Query(50, le=500),
):
    """Return the highest-risk records above a given threshold, most recent/riskiest first."""
    conn = get_connection()
    query = """
        SELECT employee_id, department, date, risk_score,
               is_odd_hour_login, high_usb_activity, large_upload, bulk_file_access,
               label_is_anomaly
        FROM risk_scores
        WHERE risk_score >= ?
        ORDER BY risk_score DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(threshold, limit))
    conn.close()
    return df.to_dict(orient="records")


@app.get("/employee/{employee_id}/history", response_model=list[RiskRecord])
def get_employee_history(employee_id: int, limit: int = Query(30, le=200)):
    """Return an employee's recent activity + risk scores, most recent first."""
    conn = get_connection()
    query = """
        SELECT employee_id, department, date, risk_score,
               is_odd_hour_login, high_usb_activity, large_upload, bulk_file_access,
               label_is_anomaly
        FROM risk_scores
        WHERE employee_id = ?
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(employee_id, limit))
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No records found for employee_id={employee_id}")
    return df.to_dict(orient="records")


@app.get("/employees/summary")
def get_employee_summary():
    """Return the current max risk score per employee, for a dashboard overview table."""
    conn = get_connection()
    query = """
        SELECT employee_id, department,
               MAX(risk_score) AS max_risk_score,
               AVG(risk_score) AS avg_risk_score,
               COUNT(*) AS records_count
        FROM risk_scores
        GROUP BY employee_id, department
        ORDER BY max_risk_score DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
