import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Motor Health Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }

    /* KPI card */
    .kpi-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid;
        margin-bottom: 6px;
    }
    .kpi-card .label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-card .value {
        font-size: 36px;
        font-weight: 800;
        line-height: 1;
    }
    .kpi-card .sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    .kpi-healthy  { border-color: #22c55e; }
    .kpi-healthy  .value { color: #22c55e; }
    .kpi-attention { border-color: #f59e0b; }
    .kpi-attention .value { color: #f59e0b; }
    .kpi-critical  { border-color: #ef4444; }
    .kpi-critical  .value { color: #ef4444; }
    .kpi-total     { border-color: #3b82f6; }
    .kpi-total     .value { color: #3b82f6; }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
    }
    .pill-healthy   { background: #14532d; color: #86efac; }
    .pill-attention { background: #78350f; color: #fcd34d; }
    .pill-critical  { background: #7f1d1d; color: #fca5a5; }
    .pill-due       { background: #1e3a5f; color: #93c5fd; }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        padding-bottom: 6px;
        border-bottom: 2px solid #334155;
        margin-bottom: 16px;
    }

    /* Motor detail card */
    .detail-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid #334155;
    }
    .detail-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .detail-value { font-size: 20px; font-weight: 700; color: #f1f5f9; }
    .detail-unit  { font-size: 11px; color: #94a3b8; }

    /* Gauge text */
    .gauge-wrap { text-align: center; }

    /* Alert box */
    .alert-box {
        background: #292524;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #fca5a5;
    }
    .warn-box {
        background: #292524;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #fcd34d;
    }
    .ok-box {
        background: #1a2e1a;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #86efac;
    }

    /* Hide default streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD + CACHE DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    master      = pd.read_csv("motor_master.csv")
    condition   = pd.read_csv("motor_condition_history.csv")
    maintenance = pd.read_csv("motor_maintenance.csv")

    condition["Date"]             = pd.to_datetime(condition["Date"])
    maintenance["Next_Service"]   = pd.to_datetime(maintenance["Next_Service"])
    maintenance["Last_Service"]   = pd.to_datetime(maintenance["Last_Service"])
    maintenance["Lubrication_Date"] = pd.to_datetime(maintenance["Lubrication_Date"])
    master["Install_Date"]        = pd.to_datetime(master["Install_Date"])

    return master, condition, maintenance

master, condition, maintenance = load_data()

# ─────────────────────────────────────────
# LATEST READING PER MOTOR
# ─────────────────────────────────────────
latest = (
    condition.sort_values("Date")
    .groupby("Motor_ID")
    .tail(1)
    .reset_index(drop=True)
)

df = latest.merge(master, on="Motor_ID", suffixes=("_cond", "_master"))
df = df.merge(maintenance, on="Motor_ID")
# Resolve Voltage_V collision: use nameplate voltage from master
df["Voltage_V"] = df["Voltage_V_master"] if "Voltage_V_master" in df.columns else df.get("Voltage_V_cond", df.get("Voltage_V", 0))

# ─────────────────────────────────────────
# HEALTH SCORING (improved)
# ─────────────────────────────────────────
def calculate_health(row):
    # Bearing temp: warning >70, critical >80
    if   row["Bearing_Temp_C"] < 60:  temp_score = 100
    elif row["Bearing_Temp_C"] < 70:  temp_score = 80
    elif row["Bearing_Temp_C"] < 80:  temp_score = 55
    else:                              temp_score = 30

    # Vibration: ISO 10816-3 bands
    v = row["Vibration_mm_s"]
    if   v < 1.8: vib_score = 100
    elif v < 2.8: vib_score = 80
    elif v < 4.5: vib_score = 55
    elif v < 7.1: vib_score = 30
    else:          vib_score = 0

    # Insulation resistance: IEEE 43-2013
    ir = row["Insulation_Resistance_MOhm"]
    if   ir >= 100: ir_score = 100
    elif ir >= 50:  ir_score = 80
    elif ir >= 10:  ir_score = 55
    elif ir >= 1:   ir_score = 30
    else:            ir_score = 0

    # Days to service
    days_left = (row["Next_Service"] - pd.Timestamp.today()).days
    if   days_left > 60: svc_score = 100
    elif days_left > 30: svc_score = 80
    elif days_left > 0:  svc_score = 60
    else:                 svc_score = 30

    return round(
        temp_score * 0.30 +
        vib_score  * 0.30 +
        ir_score   * 0.25 +
        svc_score  * 0.15,
        1
    )

df["Health_Score"]  = df.apply(calculate_health, axis=1)
df["Days_To_Service"] = (df["Next_Service"] - pd.Timestamp.today()).dt.days

def health_status(score):
    if score >= 75: return "Healthy"
    if score >= 50: return "Attention"
    return "Critical"

df["Health_Status"] = df["Health_Score"].apply(health_status)

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e293b",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    title_font=dict(size=14, color="#e2e8f0"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    xaxis=dict(gridcolor="#334155", linecolor="#475569", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#334155", linecolor="#475569", tickfont=dict(color="#94a3b8")),
)

STATUS_COLORS = {
    "Healthy":   "#22c55e",
    "Attention": "#f59e0b",
    "Critical":  "#ef4444",
}

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px'>
            <span style='font-size:32px'>⚙️</span><br>
            <span style='font-size:18px; font-weight:800; color:#f1f5f9'>Motor Health</span><br>
            <span style='font-size:11px; color:#64748b; letter-spacing:2px'>DASHBOARD</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#94a3b8; font-size:12px; font-weight:700; letter-spacing:1px; margin-bottom:8px'>FILTERS</div>", unsafe_allow_html=True)

    locations = st.multiselect(
        "Location",
        options=df["Location"].unique().tolist(),
        default=df["Location"].unique().tolist(),
        key="loc_filter"
    )

    manufacturers = st.multiselect(
        "Manufacturer",
        options=df["Manufacturer"].unique().tolist(),
        default=df["Manufacturer"].unique().tolist(),
        key="mfr_filter"
    )

    status_filter = st.multiselect(
        "Health Status",
        options=["Healthy", "Attention", "Critical"],
        default=["Healthy", "Attention", "Critical"],
        key="sts_filter"
    )

    st.markdown("---")
    st.markdown("<div style='color:#94a3b8; font-size:12px; font-weight:700; letter-spacing:1px; margin-bottom:8px'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio(
        "View",
        ["📊 Fleet Overview", "🔍 Motor Analysis", "🔧 Maintenance Planner"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"<div style='color:#475569; font-size:11px; text-align:center'>Data updated: {condition['Date'].max().strftime('%d %b %Y')}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────
filtered = df[
    df["Location"].isin(locations) &
    df["Manufacturer"].isin(manufacturers) &
    df["Health_Status"].isin(status_filter)
].copy()

# ─────────────────────────────────────────
# ════════════════════════════════════════
#   PAGE 1 — FLEET OVERVIEW
# ════════════════════════════════════════
# ─────────────────────────────────────────
if page == "📊 Fleet Overview":

    st.markdown("<h1 style='color:#f1f5f9; font-size:26px; font-weight:800; margin-bottom:4px'>Fleet Overview</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b; font-size:14px; margin-bottom:24px'>{len(filtered)} motors shown · {len(df)} total fleet</p>", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total     = len(filtered)
    n_healthy = len(filtered[filtered["Health_Status"] == "Healthy"])
    n_attn    = len(filtered[filtered["Health_Status"] == "Attention"])
    n_crit    = len(filtered[filtered["Health_Status"] == "Critical"])

    c1.markdown(f"""<div class='kpi-card kpi-total'>
        <div class='label'>Total Motors</div>
        <div class='value'>{total}</div>
        <div class='sub'>in selection</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""<div class='kpi-card kpi-healthy'>
        <div class='label'>Healthy</div>
        <div class='value'>{n_healthy}</div>
        <div class='sub'>{100*n_healthy//max(total,1)}% of fleet</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""<div class='kpi-card kpi-attention'>
        <div class='label'>Attention</div>
        <div class='value'>{n_attn}</div>
        <div class='sub'>monitor closely</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""<div class='kpi-card kpi-critical'>
        <div class='label'>Critical</div>
        <div class='value'>{n_crit}</div>
        <div class='sub'>action required</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: pie + avg health bar ──────
    col_a, col_b = st.columns([1, 1.6])

    with col_a:
        st.markdown("<div class='section-header'>Health Distribution</div>", unsafe_allow_html=True)
        counts = filtered["Health_Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=counts["Status"],
            values=counts["Count"],
            hole=0.55,
            marker_colors=[STATUS_COLORS.get(s, "#94a3b8") for s in counts["Status"]],
            textinfo="percent+label",
            textfont=dict(size=12, color="#f1f5f9"),
            hovertemplate="<b>%{label}</b><br>%{value} motors<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{total}</b><br>Motors",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e2e8f0")
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown("<div class='section-header'>Average Health Score by Location</div>", unsafe_allow_html=True)
        avg_loc = (
            filtered.groupby("Location")["Health_Score"]
            .mean()
            .reset_index()
            .sort_values("Health_Score")
        )
        avg_loc["Color"] = avg_loc["Health_Score"].apply(
            lambda s: "#22c55e" if s >= 75 else ("#f59e0b" if s >= 50 else "#ef4444")
        )
        fig_bar = go.Figure(go.Bar(
            y=avg_loc["Location"],
            x=avg_loc["Health_Score"],
            orientation="h",
            marker_color=avg_loc["Color"],
            text=avg_loc["Health_Score"].round(1),
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
            hovertemplate="<b>%{y}</b><br>Avg Health: %{x:.1f}<extra></extra>",
        ))
        fig_bar.add_vline(x=75, line_dash="dot", line_color="#22c55e", line_width=1,
                          annotation_text="Healthy", annotation_font_color="#22c55e", annotation_font_size=10)
        fig_bar.add_vline(x=50, line_dash="dot", line_color="#f59e0b", line_width=1,
                          annotation_text="Attention", annotation_font_color="#f59e0b", annotation_font_size=10)
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_range=[0, 115])
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: vibration scatter + health heatmap ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("<div class='section-header'>Vibration vs Bearing Temperature</div>", unsafe_allow_html=True)
        fig_sc = px.scatter(
            filtered,
            x="Vibration_mm_s", y="Bearing_Temp_C",
            color="Health_Status",
            color_discrete_map=STATUS_COLORS,
            hover_data=["Motor_ID", "Location", "Health_Score"],
            size="Health_Score",
            size_max=18,
            labels={"Vibration_mm_s": "Vibration (mm/s)", "Bearing_Temp_C": "Bearing Temp (°C)"},
        )
        fig_sc.add_vline(x=2.8, line_dash="dot", line_color="#f59e0b", line_width=1)
        fig_sc.add_hline(y=70,  line_dash="dot", line_color="#f59e0b", line_width=1)
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    with col_d:
        st.markdown("<div class='section-header'>Health Score by Manufacturer</div>", unsafe_allow_html=True)
        avg_mfr = (
            filtered.groupby(["Manufacturer", "Location"])["Health_Score"]
            .mean()
            .reset_index()
        )
        fig_heat = px.density_heatmap(
            filtered,
            x="Manufacturer", y="Location",
            z="Health_Score",
            histfunc="avg",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            range_color=[0, 100],
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_colorbar=dict(title="Score", tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Worst motors table ─────────
    st.markdown("<div class='section-header'>Motors Requiring Attention</div>", unsafe_allow_html=True)

    worst = (
        filtered.sort_values("Health_Score")
        .head(15)[[
            "Motor_ID", "Description", "Location", "Manufacturer",
            "Bearing_Temp_C", "Winding_Temp_C", "Vibration_mm_s",
            "Insulation_Resistance_MOhm", "Health_Score", "Health_Status", "Days_To_Service"
        ]]
    )

    fig_worst = go.Figure(go.Bar(
        x=worst["Motor_ID"],
        y=worst["Health_Score"],
        marker_color=[STATUS_COLORS.get(s, "#94a3b8") for s in worst["Health_Status"]],
        text=worst["Health_Score"].astype(str),
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Health: %{y}<br>"
            "<extra></extra>"
        ),
    ))
    fig_worst.add_hline(y=75, line_dash="dot", line_color="#22c55e", line_width=1)
    fig_worst.add_hline(y=50, line_dash="dot", line_color="#f59e0b", line_width=1)
    fig_worst.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_range=[0, 115])
    st.plotly_chart(fig_worst, use_container_width=True, config={"displayModeBar": False})

    # ── Fleet table ───────────────────────
    with st.expander("📋 Full Fleet Table", expanded=False):
        display = filtered[[
            "Motor_ID", "Description", "Location", "Manufacturer", "Power_kW",
            "Bearing_Temp_C", "Winding_Temp_C", "Vibration_mm_s",
            "Insulation_Resistance_MOhm", "Health_Score", "Health_Status", "Days_To_Service"
        ]].sort_values("Health_Score").copy()

        def color_status(val):
            colors = {"Healthy": "color:#22c55e; font-weight:bold",
                      "Attention": "color:#f59e0b; font-weight:bold",
                      "Critical": "color:#ef4444; font-weight:bold"}
            return colors.get(val, "")

        st.dataframe(
            display.style.map(color_status, subset=["Health_Status"]),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────
# ════════════════════════════════════════
#   PAGE 2 — MOTOR ANALYSIS
# ════════════════════════════════════════
# ─────────────────────────────────────────
elif page == "🔍 Motor Analysis":

    st.markdown("<h1 style='color:#f1f5f9; font-size:26px; font-weight:800; margin-bottom:4px'>Motor Deep Dive</h1>", unsafe_allow_html=True)

    motor_ids = master["Motor_ID"].unique().tolist()
    sel_motor = st.selectbox("Select Motor", motor_ids, key="deep_motor")

    # Motor info
    motor_row  = df[df["Motor_ID"] == sel_motor].iloc[0]
    motor_hist = condition[condition["Motor_ID"] == sel_motor].sort_values("Date")

    # ── Motor identity bar ─────────────────
    st.markdown(f"""
    <div style='background:#1e293b; border-radius:12px; padding:16px 24px; margin-bottom:20px;
                display:flex; justify-content:space-between; align-items:center;
                border:1px solid #334155; flex-wrap:wrap; gap:12px;'>
        <div>
            <span style='font-size:22px; font-weight:800; color:#f1f5f9'>{sel_motor}</span>
            <span style='color:#64748b; margin:0 12px'>|</span>
            <span style='color:#94a3b8'>{motor_row['Description']}</span>
        </div>
        <div style='display:flex; gap:24px; flex-wrap:wrap;'>
            <div><span style='color:#64748b; font-size:11px'>LOCATION</span><br>
                 <span style='color:#e2e8f0; font-weight:600'>{motor_row['Location']}</span></div>
            <div><span style='color:#64748b; font-size:11px'>MANUFACTURER</span><br>
                 <span style='color:#e2e8f0; font-weight:600'>{motor_row['Manufacturer']}</span></div>
            <div><span style='color:#64748b; font-size:11px'>POWER</span><br>
                 <span style='color:#e2e8f0; font-weight:600'>{motor_row['Power_kW']} kW</span></div>
            <div><span style='color:#64748b; font-size:11px'>VOLTAGE</span><br>
                 <span style='color:#e2e8f0; font-weight:600'>{motor_row['Voltage_V']} V</span></div>
            <div><span style='color:#64748b; font-size:11px'>INSTALLED</span><br>
                 <span style='color:#e2e8f0; font-weight:600'>{motor_row['Install_Date'].strftime('%d %b %Y')}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Health gauge + metric cards ────────
    col_g, col_m = st.columns([1, 2])

    with col_g:
        score = motor_row["Health_Score"]
        status = motor_row["Health_Status"]
        gauge_color = STATUS_COLORS[status]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 75, "valueformat": ".1f", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#0f172a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "#1f1515"},
                    {"range": [50, 75], "color": "#1f1a0a"},
                    {"range": [75,100], "color": "#0f1f12"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            title={"text": f"Health Score<br><span style='font-size:14px;color:{gauge_color}'>{status}</span>",
                   "font": {"size": 14, "color": "#e2e8f0"}},
            number={"font": {"color": gauge_color, "size": 48}},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with col_m:
        mc1, mc2, mc3 = st.columns(3)

        def metric_card(col, label, value, unit, threshold_warn, threshold_crit, higher_is_bad=True):
            if higher_is_bad:
                color = "#22c55e" if value < threshold_warn else ("#f59e0b" if value < threshold_crit else "#ef4444")
            else:
                color = "#22c55e" if value > threshold_warn else ("#f59e0b" if value > threshold_crit else "#ef4444")
            col.markdown(f"""
            <div class='detail-card'>
                <div class='detail-label'>{label}</div>
                <div class='detail-value' style='color:{color}'>{value}</div>
                <div class='detail-unit'>{unit}</div>
            </div>""", unsafe_allow_html=True)

        metric_card(mc1, "Bearing Temp",    round(motor_row["Bearing_Temp_C"],1),    "°C",    70,   80)
        metric_card(mc1, "Winding Temp",    round(motor_row["Winding_Temp_C"],1),    "°C",    85,  100)
        metric_card(mc2, "Vibration",       round(motor_row["Vibration_mm_s"],2),    "mm/s",  2.8, 4.5)
        metric_card(mc2, "Current",         round(motor_row["Current_A"],1),         "A",     150, 180)
        metric_card(mc3, "Insulation (IR)", round(motor_row["Insulation_Resistance_MOhm"],1), "MΩ", 50, 10, higher_is_bad=False)
        metric_card(mc3, "Running Hours",   int(motor_row["Running_Hours"]),         "hrs",   4500, 5000)

    # ── Trend charts ───────────────────────
    st.markdown("<div class='section-header'>Historical Trends</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "📳 Vibration", "⚡ Electrical"])

    def add_threshold_line(fig, y, color, name):
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=name, annotation_font_color=color, annotation_font_size=10,
                      annotation_position="bottom right")

    with tab1:
        fig_tmp = go.Figure()
        fig_tmp.add_trace(go.Scatter(
            x=motor_hist["Date"], y=motor_hist["Bearing_Temp_C"],
            name="Bearing Temp", line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"
        ))
        fig_tmp.add_trace(go.Scatter(
            x=motor_hist["Date"], y=motor_hist["Winding_Temp_C"],
            name="Winding Temp", line=dict(color="#f59e0b", width=2),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.06)"
        ))
        add_threshold_line(fig_tmp, 70, "#f59e0b", "Bearing warn")
        add_threshold_line(fig_tmp, 85, "#ef4444", "Winding warn")
        fig_tmp.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_tmp, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        fig_vib = go.Figure()
        fig_vib.add_trace(go.Scatter(
            x=motor_hist["Date"], y=motor_hist["Vibration_mm_s"],
            name="Vibration", line=dict(color="#22c55e", width=2),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"
        ))
        # Rolling average
        motor_hist_copy = motor_hist.copy()
        motor_hist_copy["Vib_7d"] = motor_hist_copy["Vibration_mm_s"].rolling(7, min_periods=1).mean()
        fig_vib.add_trace(go.Scatter(
            x=motor_hist_copy["Date"], y=motor_hist_copy["Vib_7d"],
            name="7-day avg", line=dict(color="#a855f7", width=2, dash="dot")
        ))
        add_threshold_line(fig_vib, 2.8, "#f59e0b", "ISO Watch")
        add_threshold_line(fig_vib, 4.5, "#ef4444", "ISO Alert")
        fig_vib.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Vibration (mm/s)")
        st.plotly_chart(fig_vib, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        fig_elec = make_subplots(rows=1, cols=2, subplot_titles=["Current (A)", "Power Factor"])
        fig_elec.add_trace(go.Scatter(
            x=motor_hist["Date"], y=motor_hist["Current_A"],
            name="Current", line=dict(color="#38bdf8", width=2)
        ), row=1, col=1)
        fig_elec.add_trace(go.Scatter(
            x=motor_hist["Date"], y=motor_hist["Power_Factor"],
            name="PF", line=dict(color="#a78bfa", width=2)
        ), row=1, col=2)
        fig_elec.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        fig_elec.update_xaxes(gridcolor="#334155", linecolor="#475569", tickfont=dict(color="#94a3b8"))
        fig_elec.update_yaxes(gridcolor="#334155", linecolor="#475569", tickfont=dict(color="#94a3b8"))
        st.plotly_chart(fig_elec, use_container_width=True, config={"displayModeBar": False})

    # ── Recommendations ────────────────────
    st.markdown("<div class='section-header'>Diagnostics & Recommendations</div>", unsafe_allow_html=True)

    alerts = []
    if motor_row["Vibration_mm_s"] > 4.5:
        alerts.append(("🔴 CRITICAL", "Vibration exceeds ISO 10816-3 alarm limit (>4.5 mm/s). Inspect bearings, check for misalignment or unbalance immediately.", "alert-box"))
    elif motor_row["Vibration_mm_s"] > 2.8:
        alerts.append(("🟡 WARNING", f"Vibration at {motor_row['Vibration_mm_s']:.2f} mm/s in ISO 'Monitor' band. Schedule bearing inspection.", "warn-box"))

    if motor_row["Bearing_Temp_C"] > 80:
        alerts.append(("🔴 CRITICAL", f"Bearing temperature {motor_row['Bearing_Temp_C']:.1f}°C is critically high. Check lubrication and cooling immediately.", "alert-box"))
    elif motor_row["Bearing_Temp_C"] > 70:
        alerts.append(("🟡 WARNING", f"Bearing temperature {motor_row['Bearing_Temp_C']:.1f}°C approaching limit. Verify cooling and lubrication.", "warn-box"))

    if motor_row["Insulation_Resistance_MOhm"] < 10:
        alerts.append(("🔴 CRITICAL", f"Insulation resistance {motor_row['Insulation_Resistance_MOhm']:.1f} MΩ — below IEEE 43-2013 minimum. Risk of electrical failure.", "alert-box"))
    elif motor_row["Insulation_Resistance_MOhm"] < 50:
        alerts.append(("🟡 WARNING", f"Insulation resistance {motor_row['Insulation_Resistance_MOhm']:.1f} MΩ in 'Monitor' range. Plan dielectric test.", "warn-box"))

    if motor_row["Days_To_Service"] < 0:
        alerts.append(("🔴 OVERDUE", f"Maintenance is overdue by {abs(int(motor_row['Days_To_Service']))} days.", "alert-box"))
    elif motor_row["Days_To_Service"] < 30:
        alerts.append(("🟡 DUE SOON", f"Scheduled maintenance due in {int(motor_row['Days_To_Service'])} days.", "warn-box"))

    if not alerts:
        st.markdown("<div class='ok-box'>✅ No active alerts — all parameters within normal operating limits. Continue scheduled monitoring.</div>", unsafe_allow_html=True)
    else:
        for level, msg, box_class in alerts:
            st.markdown(f"<div class='{box_class}'><b>{level}</b> — {msg}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ════════════════════════════════════════
#   PAGE 3 — MAINTENANCE PLANNER
# ════════════════════════════════════════
# ─────────────────────────────────────────
elif page == "🔧 Maintenance Planner":

    st.markdown("<h1 style='color:#f1f5f9; font-size:26px; font-weight:800; margin-bottom:4px'>Maintenance Planner</h1>", unsafe_allow_html=True)

    maint_df = df[[
        "Motor_ID", "Description", "Location", "Last_Service", "Next_Service",
        "Lubrication_Date", "Insulation_Resistance_MOhm", "Number_of_Starts",
        "Status", "Health_Score", "Health_Status", "Days_To_Service"
    ]].copy()

    maint_df["Days_To_Service"] = maint_df["Days_To_Service"].astype(int)

    # ── KPI row ──────────────────────────
    overdue   = len(maint_df[maint_df["Days_To_Service"] < 0])
    due_30    = len(maint_df[(maint_df["Days_To_Service"] >= 0) & (maint_df["Days_To_Service"] <= 30)])
    due_90    = len(maint_df[(maint_df["Days_To_Service"] > 30) & (maint_df["Days_To_Service"] <= 90)])
    upcoming  = len(maint_df[maint_df["Days_To_Service"] > 90])

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class='kpi-card kpi-critical'>
        <div class='label'>Overdue</div><div class='value'>{overdue}</div>
        <div class='sub'>past scheduled date</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='kpi-card kpi-attention'>
        <div class='label'>Due within 30 days</div><div class='value'>{due_30}</div>
        <div class='sub'>plan this month</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class='kpi-card kpi-total'>
        <div class='label'>Due within 90 days</div><div class='value'>{due_90}</div>
        <div class='sub'>schedule ahead</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class='kpi-card kpi-healthy'>
        <div class='label'>OK > 90 days</div><div class='value'>{upcoming}</div>
        <div class='sub'>no action needed</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gantt-style timeline ──────────────
    col_e, col_f = st.columns([1.8, 1])

    with col_e:
        st.markdown("<div class='section-header'>Service Timeline (Next 180 Days)</div>", unsafe_allow_html=True)
        today = pd.Timestamp.today()
        timeline = maint_df[maint_df["Days_To_Service"] <= 180].copy()
        timeline = timeline.sort_values("Days_To_Service")

        if not timeline.empty:
            fig_gantt = go.Figure()
            for _, row in timeline.iterrows():
                color = "#ef4444" if row["Days_To_Service"] < 0 else \
                        ("#f59e0b" if row["Days_To_Service"] <= 30 else \
                        ("#3b82f6" if row["Days_To_Service"] <= 90 else "#22c55e"))
                fig_gantt.add_trace(go.Bar(
                    y=[row["Motor_ID"]],
                    x=[max(row["Days_To_Service"], -30)],
                    orientation="h",
                    marker_color=color,
                    text=[f"{row['Days_To_Service']}d"],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=10),
                    hovertemplate=(
                        f"<b>{row['Motor_ID']}</b><br>"
                        f"Location: {row['Location']}<br>"
                        f"Next Service: {row['Next_Service'].strftime('%d %b %Y')}<br>"
                        f"Days: {row['Days_To_Service']}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_gantt.add_vline(x=0,  line_dash="solid", line_color="#ef4444", line_width=2, annotation_text="Today", annotation_font_color="#ef4444")
            fig_gantt.add_vline(x=30, line_dash="dot",   line_color="#f59e0b", line_width=1, annotation_text="30d", annotation_font_color="#f59e0b")
            fig_gantt.add_vline(x=90, line_dash="dot",   line_color="#3b82f6", line_width=1, annotation_text="90d", annotation_font_color="#3b82f6")
            fig_gantt.update_layout(**PLOTLY_LAYOUT, height=max(300, len(timeline) * 30 + 60),
                                    xaxis_title="Days Until Service", yaxis_title="")
            st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No motors due within 180 days.")

    with col_f:
        st.markdown("<div class='section-header'>Insulation Resistance Profile</div>", unsafe_allow_html=True)
        fig_ir = go.Figure(go.Bar(
            x=maint_df.sort_values("Insulation_Resistance_MOhm")["Motor_ID"],
            y=maint_df.sort_values("Insulation_Resistance_MOhm")["Insulation_Resistance_MOhm"],
            marker_color=[
                "#ef4444" if v < 10 else ("#f59e0b" if v < 50 else ("#22c55e" if v < 100 else "#3b82f6"))
                for v in maint_df.sort_values("Insulation_Resistance_MOhm")["Insulation_Resistance_MOhm"]
            ],
            hovertemplate="<b>%{x}</b><br>IR: %{y:.1f} MΩ<extra></extra>",
        ))
        fig_ir.add_hline(y=100, line_dash="dot", line_color="#22c55e", line_width=1, annotation_text="Excellent (>100)", annotation_font_color="#22c55e", annotation_font_size=9)
        fig_ir.add_hline(y=50,  line_dash="dot", line_color="#3b82f6", line_width=1, annotation_text="Good (>50)",      annotation_font_color="#3b82f6", annotation_font_size=9)
        fig_ir.add_hline(y=10,  line_dash="dot", line_color="#f59e0b", line_width=1, annotation_text="Monitor (>10)",   annotation_font_color="#f59e0b", annotation_font_size=9)
        fig_ir.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=45, yaxis_title="IR (MΩ)")
        st.plotly_chart(fig_ir, use_container_width=True, config={"displayModeBar": False})

    # ── Priority table ─────────────────────
    st.markdown("<div class='section-header'>Maintenance Priority Table</div>", unsafe_allow_html=True)

    def priority_label(row):
        if row["Days_To_Service"] < 0:         return "🔴 OVERDUE"
        elif row["Days_To_Service"] <= 30:      return "🟠 Due Soon"
        elif row["Health_Status"] == "Critical": return "🔴 Critical"
        elif row["Health_Status"] == "Attention":return "🟡 Attention"
        return "🟢 OK"

    maint_df["Priority"] = maint_df.apply(priority_label, axis=1)

    display_maint = maint_df[[
        "Priority", "Motor_ID", "Description", "Location",
        "Next_Service", "Days_To_Service", "Last_Service",
        "Insulation_Resistance_MOhm", "Health_Score", "Status"
    ]].sort_values("Days_To_Service").rename(columns={
        "Insulation_Resistance_MOhm": "IR (MΩ)",
        "Days_To_Service": "Days Left",
    })

    display_maint["Next_Service"] = display_maint["Next_Service"].dt.strftime("%d %b %Y")
    display_maint["Last_Service"] = display_maint["Last_Service"].dt.strftime("%d %b %Y")

    st.dataframe(display_maint, use_container_width=True, hide_index=True)

    # ── Starts histogram ──────────────────
    st.markdown("<div class='section-header'>Number of Starts per Motor</div>", unsafe_allow_html=True)
    fig_starts = px.bar(
        maint_df.sort_values("Number_of_Starts", ascending=False),
        x="Motor_ID", y="Number_of_Starts",
        color="Location",
        labels={"Number_of_Starts": "Total Starts", "Motor_ID": ""},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_starts.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_tickangle=45)
    st.plotly_chart(fig_starts, use_container_width=True, config={"displayModeBar": False})
