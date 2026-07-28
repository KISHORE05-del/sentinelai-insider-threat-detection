-- SentinelAI Database Schema (SQLite)

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    department TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    department TEXT NOT NULL,
    date TEXT NOT NULL,
    login_hour REAL,
    logout_hour REAL,
    session_length_hrs REAL,
    files_accessed INTEGER,
    usb_events INTEGER,
    upload_mb REAL,
    apps_used INTEGER,
    after_hours_access INTEGER,
    is_odd_hour_login INTEGER,
    high_usb_activity INTEGER,
    large_upload INTEGER,
    bulk_file_access INTEGER,
    isolation_forest_risk REAL,
    lof_risk REAL,
    knn_risk REAL,
    risk_score REAL,
    label_is_anomaly INTEGER,
    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
);

CREATE INDEX IF NOT EXISTS idx_risk_scores_employee ON risk_scores (employee_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_risk ON risk_scores (risk_score DESC);
