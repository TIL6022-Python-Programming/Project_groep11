"""
Project Proposal Group 11: Schiphol Disruption Dashboard & Analysis
Dataset: Monthly_traffic_data_1992-current (1).xlsx
Run with: python -m streamlit run app.py
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Schiphol Disruption & Traffic Dashboard", layout="wide"
)

# ---------------------------------------------------------------------------
# 1. Data Ingestion & Cleaning Pipeline
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    df_raw = pd.read_excel(file_path, sheet_name="Standard&Poors", header=None)

    # Extract data starting from row index 9
    data = df_raw.iloc[9:].copy()
    data = data.dropna(subset=[1])  # Drop rows where Month is NaN
    data[0] = data[0].ffill()  # Forward fill Year

    # Map column structure based on Excel headers
    cols = [
        "Year",
        "Month",
        "ATM_Europe",
        "ATM_Intercontinental",
        "ATM_Total",
        "sep1",
        "Pax_Europe",
        "Pax_Intercontinental",
        "Pax_Total_1",
        "sep2",
        "Pax_OD",
        "Pax_Transfer",
        "Pax_Total_2",
        "sep3",
        "Pax_Scheduled",
        "Pax_Non_Scheduled",
        "Pax_Transit_Direct",
        "Pax_Total_3",
        "sep4",
        "Cargo_Tonnes",
        "Mail_Tonnes",
    ]
    data.columns = cols

    # Filter out annual 'Total' summary rows
    data = data[data["Month"] != "Total"].copy()

    # Clean numeric columns & dates
    numeric_cols = [
        "Year",
        "Pax_Europe",
        "Pax_Intercontinental",
        "Pax_Total_1",
        "Pax_OD",
        "Pax_Transfer",
        "Cargo_Tonnes",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Create proper datetime index
    month_mapping = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    data["Month_Num"] = data["Month"].map(month_mapping)
    data["date"] = pd.to_datetime(
        data["Year"].astype(str) + "-" + data["Month_Num"].astype(str) + "-01"
    )
    data = data.sort_values("date").reset_index(drop=True)
    return data


# Load data
try:
    df_schiphol = load_and_clean_data(
        "Monthly_traffic_data_1992-current (1).xlsx"
    )
except Exception as e:
    st.error(
        f"Error loading dataset: {e}. Please ensure 'Monthly_traffic_data_1992-current (1).xlsx' is present."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 2. Sidebar & Filtering Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Research Parameters")
selected_disruption = st.sidebar.selectbox(
    "Select Major Disruption",
    ["9/11 Attacks", "Financial Crisis (2008/2009)", "COVID-19"],
)

disruption_dates = {
    "9/11 Attacks": "2001-09-01",
    "Financial Crisis (2008/2009)": "2008-09-01",
    "COVID-19": "2020-03-01",
}

event_date = pd.to_datetime(disruption_dates[selected_disruption])
window_start = event_date - pd.DateOffset(months=24)
window_end = event_date + pd.DateOffset(months=24)

window_df = df_schiphol[
    (df_schiphol["date"] >= window_start)
    & (df_schiphol["date"] <= window_end)
].copy()

# Seasonal Baseline calculation
baseline_start = event_date - pd.DateOffset(months=12)
baseline_df = df_schiphol[
    (df_schiphol["date"] >= baseline_start)
    & (df_schiphol["date"] < event_date)
]
baseline_mean = baseline_df["Pax_Total_1"].mean()

window_df["normalized_traffic"] = (
    window_df["Pax_Total_1"] / baseline_mean
) * 100
window_df["months_from_event"] = (
    window_df["date"].dt.year - event_date.year
) * 12 + (window_df["date"].dt.month - event_date.month)

min_m = int(window_df["months_from_event"].min())
max_m = int(window_df["months_from_event"].max())

# ---------------------------------------------------------------------------
# 3. Dashboard Main Layout & State Management for Animation
# ---------------------------------------------------------------------------
st.title("Amsterdam Airport Schiphol Disruption Dashboard")
st.markdown(
    f"Analyzing passenger traffic dynamics around **{selected_disruption}** with automated timeline playback."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Pre-Disruption Baseline", f"{baseline_mean:,.0f} passengers/mo")
with col2:
    min_pax = window_df[window_df["months_from_event"] >= 0][
        "normalized_traffic"
    ].min()
    max_decline = 100 - min_pax
    st.metric(
        "Max Relative Decline",
        f"{max_decline:.1f}%",
        delta=f"-{max_decline:.1f}%",
        delta_color="inverse",
    )
with col3:
    post_event = window_df[window_df["months_from_event"] > 0]
    recovered = post_event[post_event["normalized_traffic"] >= 100]
    recovery_time = (
        recovered["months_from_event"].min()
        if not recovered.empty
        else "> 24"
    )
    st.metric("Estimated Recovery Time", f"{recovery_time} months")

# Initialize session state for animation timeline
if "timeline_idx" not in st.session_state:
    st.session_state.timeline_idx = min_m
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# ---------------------------------------------------------------------------
# 4. Multi-Continent Interactive Flow Map with Playback Controls
# ---------------------------------------------------------------------------
st.subheader("Global Continental Traffic Flows & Animated Timeline")

# Playback Control Buttons
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
with ctrl_col1:
    if st.button("▶ Play Animation"):
        st.session_state.is_playing = True
with ctrl_col2:
    if st.button("⏸ Pause"):
        st.session_state.is_playing = False

# Slider linked to session state
selected_month_idx = st.slider(
    "Timeline Offset (Months relative to disruption)",
    min_m,
    max_m,
    st.session_state.timeline_idx,
)
st.session_state.timeline_idx = selected_month_idx

current_row = window_df[
    window_df["months_from_event"] == st.session_state.timeline_idx
]
pax_eur = (
    current_row["Pax_Europe"].values[0]
    if not current_row.empty
    else baseline_mean * 0.7
)
pax_inter = (
    current_row["Pax_Intercontinental"].values[0]
    if not current_row.empty
    else baseline_mean * 0.3
)

SCHIPHOL_LAT, SCHIPHOL_LON = 52.31, 4.76
regional_data = pd.DataFrame(
    [
        {
            "region": "Europe (Central Hub)",
            "dep_lat": SCHIPHOL_LAT,
            "dep_lon": SCHIPHOL_LON,
            "arr_lat": 50.11,
            "arr_lon": 8.68,
            "volume": pax_eur * 0.8,
        },
        {
            "region": "North America",
            "dep_lat": SCHIPHOL_LAT,
            "dep_lon": SCHIPHOL_LON,
            "arr_lat": 40.64,
            "arr_lon": -73.78,
            "volume": pax_inter * 0.45,
        },
        {
            "region": "Asia",
            "dep_lat": SCHIPHOL_LAT,
            "dep_lon": SCHIPHOL_LON,
            "arr_lat": 35.68,
            "arr_lon": 139.65,
            "volume": pax_inter * 0.35,
        },
        {
            "region": "Africa",
            "dep_lat": SCHIPHOL_LAT,
            "dep_lon": SCHIPHOL_LON,
            "arr_lat": -26.14,
            "arr_lon": 28.25,
            "volume": pax_inter * 0.12,
        },
        {
            "region": "South America",
            "dep_lat": SCHIPHOL_LAT,
            "dep_lon": SCHIPHOL_LON,
            "arr_lat": -23.55,
            "arr_lon": -46.63,
            "volume": pax_inter * 0.08,
        },
    ]
)

regional_data["arc_width"] = (
    regional_data["volume"] / baseline_mean
) * 12

arc_layer = pdk.Layer(
    "ArcLayer",
    data=regional_data,
    get_source_position=["dep_lon", "dep_lat"],
    get_target_position=["arr_lon", "arr_lat"],
    get_source_color=[255, 140, 0],
    get_target_color=[0, 128, 255],
    get_width="arc_width",
    pickable=True,
)

view_state = pdk.ViewState(latitude=25, longitude=10, zoom=1.2, pitch=20)
st.pydeck_chart(
    pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
        tooltip={
            "text": "{region}\nEstimated Volume: {volume:,.0f} passengers (Month offset: "
            + str(st.session_state.timeline_idx)
            + ")"
        },
    )
)

# Auto-play loop logic
if st.session_state.is_playing:
    if st.session_state.timeline_idx < max_m:
        st.session_state.timeline_idx += 1
        time.sleep(0.033)  # Approx 30 FPS pacing delay per frame increment
        st.rerun()
    else:
        st.session_state.is_playing = False

# ---------------------------------------------------------------------------
# 5. Polished, Individual Trend Analysis Graph for Selected Disruption
# ---------------------------------------------------------------------------
st.subheader(f"Individual Traffic Recovery Trajectory: {selected_disruption}")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 5.5))

# Palette for individual disruptions
palette = {
    "9/11 Attacks": "#e74c3c",
    "Financial Crisis (2008/2009)": "#f39c12",
    "COVID-19": "#3498db",
}
line_color = palette.get(selected_disruption, "#3498db")

# Plot only the active disruption window data
sns.lineplot(
    data=window_df,
    x="months_from_event",
    y="normalized_traffic",
    color=line_color,
    linewidth=2.5,
    ax=ax,
    label=selected_disruption,
)

ax.axvline(0, color="black", linestyle="--", alpha=0.8, label="Disruption Event (Month 0)")
ax.axhline(100, color="gray", linestyle=":", linewidth=1.5, label="Pre-Disruption Baseline (100%)")

ax.set_title(f"Schiphol Passenger Traffic Recovery - {selected_disruption}", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Months Relative to Disruption Event", fontsize=12)
ax.set_ylabel("Passenger Traffic (% of Baseline)", fontsize=12)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
ax.set_xlim(-24, 24)
ax.set_ylim(0, 130)

sns.despine(top=True, right=True)
plt.tight_layout()

st.pyplot(fig)