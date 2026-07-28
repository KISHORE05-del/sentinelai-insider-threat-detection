"""
SentinelAI - Model Training
-----------------------------
Trains multiple unsupervised anomaly detectors on the engineered
behavioral features, combines them into a single 0-100 risk score,
and evaluates against the ground-truth anomaly labels (which exist
only because this is synthetic data -- a real deployment would be
fully unsupervised).

Models used:
  - Isolation Forest (sklearn)
  - Local Outlier Factor (PyOD)
  - K-Nearest Neighbors outlier detector (PyOD)

Output:
  - models/artifacts/*.joblib      (trained models + scaler)
  - artifacts/risk_scores.csv      (per-record risk scores for the dashboard/API)
  - Printed evaluation report
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pyod.models.lof import LOF
from pyod.models.knn import KNN

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "features", "processed", "employee_features.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RISK_SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "risk_scores.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RISK_SCORES_PATH), exist_ok=True)

FEATURE_COLUMNS = [
    "login_hour_zscore", "logout_hour_zscore", "session_length_hrs_zscore",
    "files_accessed_zscore", "usb_events_zscore", "upload_mb_zscore", "apps_used_zscore",
    "is_odd_hour_login", "high_usb_activity", "large_upload", "bulk_file_access",
]

CONTAMINATION = 0.02  # expected fraction of anomalies -- tune based on your risk tolerance


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLUMNS].values
    y = df["label_is_anomaly"].values  # only used for evaluation, never for training
    return df, X, y


def train_models(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200, contamination=CONTAMINATION, random_state=42
    )
    iso_forest.fit(X_scaled)

    lof = LOF(contamination=CONTAMINATION)
    lof.fit(X_scaled)

    knn = KNN(contamination=CONTAMINATION)
    knn.fit(X_scaled)

    return scaler, X_scaled, {"isolation_forest": iso_forest, "lof": lof, "knn": knn}


def score_to_risk(raw_scores: np.ndarray) -> np.ndarray:
    """Min-max normalize any model's raw anomaly score to a 0-100 risk score."""
    lo, hi = raw_scores.min(), raw_scores.max()
    if hi - lo == 0:
        return np.zeros_like(raw_scores)
    return 100 * (raw_scores - lo) / (hi - lo)


def compute_ensemble_risk(models: dict, X_scaled: np.ndarray) -> pd.DataFrame:
    scores = {}

    # Isolation Forest: more negative score_samples = more anomalous, so we flip sign
    iso_raw = -models["isolation_forest"].score_samples(X_scaled)
    scores["isolation_forest_risk"] = score_to_risk(iso_raw)

    # PyOD models expose decision_scores_ directly (higher = more anomalous)
    scores["lof_risk"] = score_to_risk(models["lof"].decision_scores_)
    scores["knn_risk"] = score_to_risk(models["knn"].decision_scores_)

    scores_df = pd.DataFrame(scores)
    # Ensemble risk score = average across the three detectors
    scores_df["risk_score"] = scores_df.mean(axis=1).round(1)
    return scores_df


def evaluate(df: pd.DataFrame, y_true: np.ndarray, risk_scores: np.ndarray, threshold_percentile=98):
    threshold = np.percentile(risk_scores, threshold_percentile)
    y_pred = (risk_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, risk_scores)
    except ValueError:
        auc = float("nan")

    print("\n--- Evaluation (against synthetic ground truth) ---")
    print(f"Risk threshold (P{threshold_percentile}): {threshold:.2f}")
    print(f"Flagged as high-risk: {y_pred.sum()} / {len(y_pred)} records")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


if __name__ == "__main__":
    df, X, y = load_data()
    scaler, X_scaled, models = train_models(X)
    scores_df = compute_ensemble_risk(models, X_scaled)

    result_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    result_df.to_csv(RISK_SCORES_PATH, index=False)

    metrics = evaluate(df, y, scores_df["risk_score"].values)

    # Persist everything the API/dashboard will need at inference time
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    joblib.dump(models["isolation_forest"], os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    joblib.dump(models["lof"], os.path.join(ARTIFACTS_DIR, "lof.joblib"))
    joblib.dump(models["knn"], os.path.join(ARTIFACTS_DIR, "knn.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))

    print(f"\nSaved risk scores to: {RISK_SCORES_PATH}")
    print(f"Saved trained models to: {ARTIFACTS_DIR}")

    print("\nTop 10 highest-risk records:")
    top10 = result_df.sort_values("risk_score", ascending=False).head(10)
    print(top10[["employee_id", "department", "date", "risk_score", "label_is_anomaly"]].to_string(index=False))
