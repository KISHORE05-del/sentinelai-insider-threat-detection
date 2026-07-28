"""
SentinelAI - Load risk scores into SQLite
--------------------------------------------
Takes the output of models/train_model.py (artifacts/risk_scores.csv)
and loads it into a SQLite database that the API/dashboard read from.

Run this after every retraining.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sentinelai.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")
RISK_SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "risk_scores.csv")

DB_COLUMNS = [
    "employee_id", "department", "date", "login_hour", "logout_hour",
    "session_length_hrs", "files_accessed", "usb_events", "upload_mb",
    "apps_used", "after_hours_access", "is_odd_hour_login", "high_usb_activity",
    "large_upload", "bulk_file_access", "isolation_forest_risk", "lof_risk",
    "knn_risk", "risk_score", "label_is_anomaly",
]


def main():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    df = pd.read_csv(RISK_SCORES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Populate employees table
    employees = df[["employee_id", "department"]].drop_duplicates()
    employees.to_sql("employees", conn, if_exists="replace", index=False)

    # Populate risk_scores table (replace each run so retraining stays consistent)
    conn.execute("DELETE FROM risk_scores")
    df[DB_COLUMNS].to_sql("risk_scores", conn, if_exists="append", index=False)

    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
    print(f"Loaded {count:,} rows into {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
