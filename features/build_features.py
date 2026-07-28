"""
SentinelAI - Feature Engineering
---------------------------------
Turns raw daily activity logs into behavioral-deviation features:
for each day, how far does this employee's activity deviate from
THEIR OWN rolling personal baseline (not a global average).

This is the core idea behind insider threat detection: a login at
3 AM might be totally normal for a night-shift sysadmin, but highly
anomalous for a 9-to-5 accountant. So we compare each person to
their own history, not to everyone else.

Output: features/processed/employee_features.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "employee_logs.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
OUT_PATH = os.path.join(OUT_DIR, "employee_features.csv")

ROLLING_WINDOW = 14  # days of history used to build each employee's personal baseline

RAW_NUMERIC_COLS = [
    "login_hour", "logout_hour", "session_length_hrs",
    "files_accessed", "usb_events", "upload_mb", "apps_used",
]


def add_rolling_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each employee, compute a rolling mean/std of their own past
    behavior (shifted by 1 day so we never leak "today" into its own
    baseline), then express today's values as a z-score deviation
    from that personal baseline.
    """
    df = df.sort_values(["employee_id", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    feature_frames = []

    for emp_id, group in df.groupby("employee_id"):
        group = group.sort_values("date").reset_index(drop=True)

        for col in RAW_NUMERIC_COLS:
            roll_mean = group[col].shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
            roll_std = group[col].shift(1).rolling(ROLLING_WINDOW, min_periods=3).std().replace(0, np.nan)

            group[f"{col}_baseline_mean"] = roll_mean
            group[f"{col}_zscore"] = (group[col] - roll_mean) / roll_std

        feature_frames.append(group)

    out = pd.concat(feature_frames, ignore_index=True)

    # Drop the earliest rows per employee where we don't have enough history yet
    out = out.dropna(subset=[f"{c}_zscore" for c in RAW_NUMERIC_COLS])

    # Fill any remaining edge-case NaNs (e.g. zero-variance windows) with 0 (no deviation)
    zscore_cols = [f"{c}_zscore" for c in RAW_NUMERIC_COLS]
    out[zscore_cols] = out[zscore_cols].fillna(0)

    return out


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra engineered signals that are useful even without personal history."""
    df["is_odd_hour_login"] = ((df["login_hour"] < 6) | (df["login_hour"] > 21)).astype(int)
    df["high_usb_activity"] = (df["usb_events"] >= 3).astype(int)
    df["large_upload"] = (df["upload_mb"] > df["upload_mb"].quantile(0.95)).astype(int)
    df["bulk_file_access"] = (df["files_accessed"] > df["files_accessed"].quantile(0.95)).astype(int)
    return df


def build_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = add_contextual_features(df)
    df = add_rolling_baseline_features(df)
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    features_df = build_features()
    features_df.to_csv(OUT_PATH, index=False)
    print(f"Built features for {len(features_df):,} rows.")
    print(f"Anomalies retained after windowing: {features_df['label_is_anomaly'].sum()}")
    print(f"Saved to: {OUT_PATH}")
    print("\nFeature columns created:")
    new_cols = [c for c in features_df.columns if "zscore" in c or "baseline" in c or c.startswith(("is_odd", "high_usb", "large_upload", "bulk_file"))]
    for c in new_cols:
        print(f"  - {c}")
