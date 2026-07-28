"""
SentinelAI - Synthetic Data Generator
--------------------------------------
Generates realistic employee activity logs (login times, file access,
USB usage, application usage, data uploads) for N employees over a
period of days, with a controlled set of injected anomalies so we
have ground truth to evaluate our models against.

Output: data/raw/employee_logs.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
NUM_EMPLOYEES = 60
NUM_DAYS = 90
ANOMALY_EMPLOYEE_FRACTION = 0.10   # ~10% of employees get anomalous days injected
ANOMALY_DAY_FRACTION = 0.05        # of an anomalous employee's days, 5% are anomalous
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

DEPARTMENTS = ["Engineering", "Finance", "HR", "Sales", "IT", "Legal"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "employee_logs.csv")


def make_employee_profiles(n_employees: int) -> pd.DataFrame:
    """Give each employee a stable 'normal behavior' baseline."""
    profiles = []
    for emp_id in range(1, n_employees + 1):
        profiles.append({
            "employee_id": emp_id,
            "department": np.random.choice(DEPARTMENTS),
            "avg_login_hour": np.random.normal(9, 0.6),       # typical login time
            "avg_logout_hour": np.random.normal(18, 0.7),
            "avg_files_accessed": max(5, np.random.normal(25, 8)),
            "avg_usb_events": max(0, np.random.normal(0.3, 0.3)),
            "avg_upload_mb": max(1, np.random.normal(15, 6)),
            "avg_apps_used": max(1, np.random.normal(4, 1.2)),
            "is_flagged_for_anomalies": False,
        })
    df = pd.DataFrame(profiles)

    # Randomly select a subset of employees who will have anomalous days
    n_anomalous_emps = max(1, int(n_employees * ANOMALY_EMPLOYEE_FRACTION))
    anomalous_ids = np.random.choice(df["employee_id"], size=n_anomalous_emps, replace=False)
    df.loc[df["employee_id"].isin(anomalous_ids), "is_flagged_for_anomalies"] = True
    return df


def generate_day_record(profile: pd.Series, date: datetime, is_anomalous: bool) -> dict:
    """Generate one day's activity log for one employee."""

    if not is_anomalous:
        login_hour = np.clip(np.random.normal(profile["avg_login_hour"], 0.5), 5, 12)
        logout_hour = np.clip(np.random.normal(profile["avg_logout_hour"], 0.6), login_hour + 1, 23)
        files_accessed = max(0, int(np.random.normal(profile["avg_files_accessed"], 5)))
        usb_events = max(0, int(np.random.poisson(profile["avg_usb_events"])))
        upload_mb = max(0, np.random.normal(profile["avg_upload_mb"], 4))
        apps_used = max(1, int(np.random.normal(profile["avg_apps_used"], 1)))
        after_hours_access = 1 if (login_hour < 6 or logout_hour > 21) else 0
        label = 0
    else:
        # Inject one of several insider-threat-like patterns
        pattern = np.random.choice(
            ["odd_hours_bulk_download", "usb_exfiltration", "mass_upload", "off_hours_access"]
        )
        login_hour = profile["avg_login_hour"]
        logout_hour = profile["avg_logout_hour"]
        files_accessed = int(profile["avg_files_accessed"])
        usb_events = int(profile["avg_usb_events"])
        upload_mb = profile["avg_upload_mb"]
        apps_used = int(profile["avg_apps_used"])
        after_hours_access = 0

        if pattern == "odd_hours_bulk_download":
            login_hour = np.random.choice([1, 2, 3, 23])
            logout_hour = login_hour + np.random.uniform(1, 3)
            files_accessed = int(profile["avg_files_accessed"] * np.random.uniform(6, 12))
            after_hours_access = 1
        elif pattern == "usb_exfiltration":
            usb_events = int(np.random.uniform(5, 15))
            files_accessed = int(profile["avg_files_accessed"] * np.random.uniform(2, 4))
        elif pattern == "mass_upload":
            upload_mb = profile["avg_upload_mb"] * np.random.uniform(15, 40)
            files_accessed = int(profile["avg_files_accessed"] * np.random.uniform(3, 6))
        elif pattern == "off_hours_access":
            login_hour = np.random.choice([0, 1, 4, 22, 23])
            logout_hour = login_hour + np.random.uniform(0.5, 2)
            after_hours_access = 1

        label = 1

    return {
        "employee_id": profile["employee_id"],
        "department": profile["department"],
        "date": date.strftime("%Y-%m-%d"),
        "login_hour": round(login_hour, 2),
        "logout_hour": round(logout_hour, 2),
        "session_length_hrs": round(max(0.1, logout_hour - login_hour), 2),
        "files_accessed": files_accessed,
        "usb_events": usb_events,
        "upload_mb": round(upload_mb, 2),
        "apps_used": apps_used,
        "after_hours_access": after_hours_access,
        "is_weekend": 1 if date.weekday() >= 5 else 0,
        "label_is_anomaly": label,  # ground truth, NOT used as a model feature
    }


def generate_dataset() -> pd.DataFrame:
    profiles = make_employee_profiles(NUM_EMPLOYEES)
    start_date = datetime.today() - timedelta(days=NUM_DAYS)

    records = []
    for _, profile in profiles.iterrows():
        for day_offset in range(NUM_DAYS):
            date = start_date + timedelta(days=day_offset)

            is_anomalous_day = False
            if profile["is_flagged_for_anomalies"]:
                if np.random.rand() < ANOMALY_DAY_FRACTION:
                    is_anomalous_day = True

            # Skip some weekends to keep data realistic (not everyone works weekends)
            if date.weekday() >= 5 and np.random.rand() > 0.15:
                continue

            record = generate_day_record(profile, date, is_anomalous_day)
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} activity records for {NUM_EMPLOYEES} employees over {NUM_DAYS} days.")
    print(f"Anomalous records (ground truth): {df['label_is_anomaly'].sum()} "
          f"({100 * df['label_is_anomaly'].mean():.2f}%)")
    print(f"Saved to: {OUTPUT_PATH}")
