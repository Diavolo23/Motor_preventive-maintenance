import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Motor Preventive Maintenance Dashboard",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

master = pd.read_csv("motor_master.csv")
condition = pd.read_csv("motor_condition_history.csv")
maintenance = pd.read_csv("motor_maintenance.csv")

condition["Date"] = pd.to_datetime(condition["Date"])
maintenance["Next_Service"] = pd.to_datetime(
    maintenance["Next_Service"]
)

# ==========================================
# GET LATEST READING PER MOTOR
# ==========================================

latest = (
    condition.sort_values("Date")
    .groupby("Motor_ID")
    .tail(1)
)

df = latest.merge(master, on="Motor_ID")
df = df.merge(maintenance, on="Motor_ID")

# ==========================================
# HEALTH SCORE
# ==========================================

def calculate_health(row):

    temp_score = max(
        0,
        100 - row["Bearing_Temp_C"]
    )

    vib_score = max(
        0,
        100 - row["Vibration_mm_s"] * 15
    )

    days_left = (
        row["Next_Service"] - pd.Timestamp.today()
    ).days

    if days_left < 0:
        service_score = 50
    elif days_left < 30:
        service_score = 75
    else:
        service_score = 100

    health = (
        temp_score * 0.4
        + vib_score * 0.4
        + service_score * 0.2
    )

    return round(health, 1)

df["Health_Score"] = df.apply(
    calculate_health,
    axis=1
)

# ==========================================
# HEALTH STATUS
# ==========================================

def health_status(score):

    if score >= 80:
        return "Healthy"

    elif score >= 60:
        return "Attention"

    return "Critical"

df["Health_Status"] = df["Health_Score"].apply(
    health_status
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Filters")

area = st.sidebar.multiselect(
    "Location",
    df["Location"].unique(),
    default=df["Location"].unique()
)

filtered = df[df["Location"].isin(area)]

# ==========================================
# TITLE
# ==========================================

st.title("⚙️ Motor Preventive Maintenance Dashboard")

# ==========================================
# KPI CARDS
# ==========================================

total_motors = len(filtered)

healthy = len(
    filtered[
        filtered["Health_Status"] == "Healthy"
    ]
)

attention = len(
    filtered[
        filtered["Health_Status"] == "Attention"
    ]
)

critical = len(
    filtered[
        filtered["Health_Status"] == "Critical"
    ]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Motors", total_motors)
c2.metric("Healthy", healthy)
c3.metric("Attention", attention)
c4.metric("Critical", critical)

st.divider()

# ==========================================
# HEALTH DISTRIBUTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        filtered,
        names="Health_Status",
        title="Motor Health Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    avg_health = (
        filtered
        .groupby("Location")["Health_Score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        avg_health,
        x="Location",
        y="Health_Score",
        title="Average Health by Area"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================
# HEALTH RANKING
# ==========================================

st.subheader("Worst Motors")

worst = filtered.sort_values(
    "Health_Score"
).head(10)

fig = px.bar(
    worst,
    x="Motor_ID",
    y="Health_Score",
    color="Health_Status",
    title="Top 10 Motors Requiring Attention"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# MOTOR ANALYSIS
# ==========================================

st.subheader("Motor Analysis")

selected_motor = st.selectbox(
    "Select Motor",
    master["Motor_ID"].unique()
)

motor_history = condition[
    condition["Motor_ID"] == selected_motor
]

col1, col2 = st.columns(2)

with col1:

    fig = px.line(
        motor_history,
        x="Date",
        y="Bearing_Temp_C",
        title="Bearing Temperature Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.line(
        motor_history,
        x="Date",
        y="Vibration_mm_s",
        title="Vibration Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# CURRENT MOTOR DETAILS
# ==========================================

st.subheader("Current Motor Condition")

motor_row = filtered[
    filtered["Motor_ID"] == selected_motor
]

st.dataframe(
    motor_row[
        [
            "Motor_ID",
            "Location",
            "Bearing_Temp_C",
            "Winding_Temp_C",
            "Vibration_mm_s",
            "Current_A",
            "Running_Hours",
            "Health_Score",
            "Health_Status"
        ]
    ]
)

# ==========================================
# RECOMMENDATIONS
# ==========================================

st.subheader("Maintenance Recommendations")

recommendations = []

for _, row in filtered.iterrows():

    if row["Vibration_mm_s"] > 4:
        rec = "Inspect Bearings"

    elif row["Bearing_Temp_C"] > 85:
        rec = "Check Cooling System"

    elif row["Current_A"] > 180:
        rec = "Investigate Overload"

    else:
        rec = "Normal Operation"

    recommendations.append(
        [row["Motor_ID"], rec]
    )

recommend_df = pd.DataFrame(
    recommendations,
    columns=[
        "Motor_ID",
        "Recommendation"
    ]
)

st.dataframe(recommend_df)

# ==========================================
# MAINTENANCE DUE
# ==========================================

st.subheader("Maintenance Due")

maintenance["Days_Left"] = (
    maintenance["Next_Service"]
    - pd.Timestamp.today()
).dt.days

due = maintenance[
    maintenance["Days_Left"] <= 30
]

st.dataframe(due)

st.success(
    " Good to GO"
)