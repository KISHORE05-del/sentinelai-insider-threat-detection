"""
SentinelAI - Streamlit Dashboard
-----------------------------------
Interactive dashboard for security admins to review risk scores,
drill into individual employees, and see alert timelines.

Run with:
    streamlit run dashboard/app.py

Reads directly from the SQLite DB (same source the API serves from),
so it works even if you haven't started the FastAPI server.
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "sentinelai.db")

st.set_page_config(page_title="SentinelAI - Insider Threat Dashboard", layout="wide")


@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM risk_scores", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    st.title("🛡️ SentinelAI — Insider Threat Detection Dashboard")
    st.caption("AI-driven behavioral anomaly detection for employee activity monitoring")

    if not os.path.exists(DB_PATH):
        st.error("Database not found. Run `python db/load_to_db.py` first (after training the model).")
        st.stop()

    df = load_data()

    # ---------------- Sidebar filters ----------------
    st.sidebar.header("Filters")
    departments = ["All"] + sorted(df["department"].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    risk_threshold = st.sidebar.slider("Minimum risk score to flag as alert", 0, 100, 50)

    filtered = df.copy()
    if selected_dept != "All":
        filtered = filtered[filtered["department"] == selected_dept]

    # ---------------- Top KPIs ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", f"{len(filtered):,}")
    col2.metric("Employees monitored", filtered["employee_id"].nunique())
    alerts = filtered[filtered["risk_score"] >= risk_threshold]
    col3.metric(f"Alerts (risk ≥ {risk_threshold})", len(alerts))
    col4.metric("Avg risk score", f"{filtered['risk_score'].mean():.1f}")

    st.divider()

    # ---------------- Risk score distribution ----------------
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Risk score distribution")
        fig = px.histogram(filtered, x="risk_score", nbins=40,
                            color_discrete_sequence=["#4C78A8"])
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                      annotation_text="Alert threshold")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Alerts by department")
        dept_counts = alerts["department"].value_counts().reset_index()
        dept_counts.columns = ["department", "alert_count"]
        fig2 = px.pie(dept_counts, names="department", values="alert_count", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---------------- Alert timeline ----------------
    st.subheader("Risk score over time (top 10 riskiest employees)")
    top_employees = (
        filtered.groupby("employee_id")["risk_score"].max().sort_values(ascending=False).head(10).index
    )
    timeline_df = filtered[filtered["employee_id"].isin(top_employees)]
    fig3 = px.line(
        timeline_df.sort_values("date"), x="date", y="risk_score",
        color="employee_id", markers=True,
    )
    fig3.add_hline(y=risk_threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ---------------- Alert table ----------------
    st.subheader(f"🚨 Active alerts (risk ≥ {risk_threshold})")
    if alerts.empty:
        st.info("No alerts at this threshold. Try lowering the slider.")
    else:
        display_cols = [
            "employee_id", "department", "date", "risk_score",
            "is_odd_hour_login", "high_usb_activity", "large_upload", "bulk_file_access",
        ]
        st.dataframe(
            alerts[display_cols].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ---------------- Employee drill-down ----------------
    st.subheader("🔍 Employee drill-down")
    emp_id = st.selectbox("Select employee ID", sorted(df["employee_id"].unique()))
    emp_df = df[df["employee_id"] == emp_id].sort_values("date")

    st.write(f"**Department:** {emp_df['department'].iloc[0]}  |  "
             f"**Max risk score:** {emp_df['risk_score'].max():.1f}  |  "
             f"**Records:** {len(emp_df)}")

    fig4 = px.line(emp_df, x="date", y="risk_score", markers=True,
                   title=f"Risk score history — Employee {emp_id}")
    fig4.add_hline(y=risk_threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Raw activity log for this employee"):
        st.dataframe(emp_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
