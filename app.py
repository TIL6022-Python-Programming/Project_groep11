"""
Project Proposal Group 11: Schiphol Disruption Dashboard & Analysis
Dataset: Monthly_traffic_data_1992-current.xlsx
Run with: python -m streamlit run app.py
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pathlib import Path

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
        "ATM_Total",
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
        "financial_data/Monthly_traffic_data_1992-current.xlsx"
    )
except Exception as e:
    st.error(
        f"Error loading dataset: {e}. Please ensure 'Monthly_traffic_data_1992-current.xlsx' is present inside the 'financial_data/' folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 2. Main Title & Description
# ---------------------------------------------------------------------------
st.title("Amsterdam Airport Schiphol Disruption Dashboard")
st.markdown(
    "Analyzing passenger traffic dynamics, flight movements, shock intensities, recovery durations, and model trajectories across historical disruptions."
)

# Disruption metadata definitions
disruption_metadata = {
    "9/11 Attacks": {
        "date": pd.Timestamp("2001-09-01"),
        "color": "#e74c3c",
        "description": "Sudden geopolitical shock impacting intercontinental and European passenger flows starting September 2001.",
        "baseline_period": "Sep 1996 – Aug 2001 (5-year anchored)",
        "model_type": "Anchored Growth",
    },
    "Financial Crisis (2008/2009)": {
        "date": pd.Timestamp("2008-09-01"),
        "color": "#f39c12",
        "description": "Macroeconomic downturn affecting global aviation demand and corporate travel starting late 2008.",
        "baseline_period": "2004 – 2007 (Rolling baseline)",
        "model_type": "Rolling Growth",
    },
    "COVID-19": {
        "date": pd.Timestamp("2020-03-01"),
        "color": "#3498db",
        "description": "Global pandemic resulting in unprecedented travel restrictions, lockdowns, and traffic collapse from March 2020.",
        "baseline_period": "2015 – 2019 (Pre-pandemic anchored)",
        "model_type": "Anchored Growth with Forecast Guard",
    },
}

# Helper function to compute disruption analysis windows & metrics
def get_disruption_analysis(df, event_date):
    window_start = event_date - pd.DateOffset(months=24)
    window_end = event_date + pd.DateOffset(months=24)

    window_df = df[
        (df["date"] >= window_start) & (df["date"] <= window_end)
    ].copy()

    baseline_start = event_date - pd.DateOffset(months=12)
    baseline_df = df[
        (df["date"] >= baseline_start) & (df["date"] < event_date)
    ]
    baseline_mean = baseline_df["Pax_Total_1"].mean()

    window_df["normalized_traffic"] = (
        window_df["Pax_Total_1"] / baseline_mean
    ) * 100
    window_df["months_from_event"] = (
        window_df["date"].dt.year - event_date.year
    ) * 12 + (window_df["date"].dt.month - event_date.month)

    return window_df, baseline_mean

# ---------------------------------------------------------------------------
# 3. Tab-based Layout (Placing results & 5 comparative dimensions)
# ---------------------------------------------------------------------------
tab_names = list(disruption_metadata.keys()) + ["Multi-Event Comparison (5 Perspectives)"]
tabs = st.tabs(tab_names)

sns.set_theme(style="whitegrid")

# Populate individual disruption tabs
for i, (disruption_name, meta) in enumerate(disruption_metadata.items()):
    with tabs[i]:
        st.subheader(f"Disruption Results: {disruption_name}")
        st.markdown(meta["description"])
        
        event_date = meta["date"]
        window_df, baseline_mean = get_disruption_analysis(df_schiphol, event_date)
        
        # Metrics specific to this disruption
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pre-Disruption Baseline", f"{baseline_mean:,.0f} pax/mo")
        with col2:
            post_shock_df = window_df[window_df["months_from_event"] >= 0]
            min_pax_idx = post_shock_df["normalized_traffic"].min() if not post_shock_df.empty else 0
            max_decline = 100 - min_pax_idx
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
            st.metric("Recovery Time", f"{recovery_time} mos")
        with col4:
            st.metric("Methodology", meta["model_type"])

        # Two columns for plots: Passenger Index & Flight Movements (ATM)
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(
                data=window_df,
                x="months_from_event",
                y="normalized_traffic",
                color=meta["color"],
                linewidth=2.5,
                ax=ax,
                label="Passenger Index",
            )
            ax.axvline(0, color="black", linestyle="--", alpha=0.8, label="Event (Mo 0)")
            ax.axhline(100, color="gray", linestyle=":", linewidth=1.5, label="Baseline (100)")
            ax.set_title(f"Passenger Index ({disruption_name})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Months from Event", fontsize=9)
            ax.set_ylabel("Index (Baseline = 100)", fontsize=9)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(-24, 24)
            ax.set_ylim(0, 130)
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig)

        with plot_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            # Normalized ATM (Flights)
            base_atm = window_df[window_df["months_from_event"] < 0]["ATM_Total"].mean()
            window_df["normalized_atm"] = (window_df["ATM_Total"] / base_atm) * 100 if base_atm > 0 else 100
            
            sns.lineplot(
                data=window_df,
                x="months_from_event",
                y="normalized_atm",
                color="#2c3e50",
                linewidth=2.5,
                ax=ax2,
                label="Flight Movements Index",
            )
            ax2.axvline(0, color="black", linestyle="--", alpha=0.8)
            ax2.axhline(100, color="gray", linestyle=":", linewidth=1.5)
            ax2.set_title(f"Flight Movements (ATM) ({disruption_name})", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Months from Event", fontsize=9)
            ax2.set_ylabel("ATM Index (Baseline = 100)", fontsize=9)
            ax2.legend(loc="lower right", fontsize=8)
            ax2.set_xlim(-24, 24)
            ax2.set_ylim(0, 130)
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig2)

# ---------------------------------------------------------------------------
# 4. Multi-Event Comparison Tab (5 Key Comparative Perspectives)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Multi-Event Comparison: 5 Analytical Perspectives")
    st.markdown("Comprehensive side-by-side benchmark covering all **5 vergelijkingen** (perspectives): Passagiersontwikkeling, Vluchtaantallen, Modelvergelijking, Hersteltempo, en Schokintensiteit[cite: 1].")

    comparison_events = {
        '9/11': dict(
            event_name='9/11',
            event_date=pd.Timestamp('2001-09-01'),
            shock_start=pd.Timestamp('2001-09-01'),
            shock_end=pd.Timestamp('2001-09-01'),
            base_start=pd.Timestamp('1996-09-01'),
            plot_start=pd.Timestamp('1996-09-01'),
            plot_end=pd.Timestamp('2006-09-01'),
            span_label=None,
            model='Anchored (5-yr avg growth)',
            min_index=80.8,
            recovery_months='Short (< 12 mos)'
        ),
        'Financial Crisis': dict(
            event_name='Financial Crisis',
            event_date=pd.Timestamp('2008-09-01'),
            shock_start=pd.Timestamp('2008-01-01'),
            shock_end=pd.Timestamp('2010-01-01'),
            base_start=pd.Timestamp('2004-01-01'),
            plot_start=pd.Timestamp('2004-01-01'),
            plot_end=pd.Timestamp('2015-01-01'),
            span_label='Financial Crisis',
            model='Rolling Growth Model',
            min_index=81.6,
            recovery_months='Extended (24+ mos)'
        ),
        'COVID-19': dict(
            event_name='COVID-19',
            event_date=pd.Timestamp('2020-03-01'),
            shock_start=pd.Timestamp('2020-01-01'),
            shock_end=pd.Timestamp('2022-12-01'),
            base_start=pd.Timestamp('2015-01-01'),
            plot_start=pd.Timestamp('2015-01-01'),
            plot_end=df_schiphol['date'].max(),
            span_label='COVID-19 (2020-2022)',
            model='Anchored with Forecast Guard',
            min_index=2.3,
            recovery_months='Long-term structural'
        ),
    }

    @st.cache_data
    def compute_comparison_data(df_full):
        results = {}
        for name, ev in comparison_events.items():
            data = df_full.copy()
            data['Year'] = data['date'].dt.year
            data['Month_Num'] = data['date'].dt.month
            data['Passengers'] = data['Pax_Total_1']
            data['Flights'] = data['ATM_Total']
            
            before = data[(data['date'] >= ev['base_start']) & (data['date'] < ev['shock_start'])]
            base_pax = before.groupby('Month_Num')['Passengers'].mean()
            base_atm = before.groupby('Month_Num')['Flights'].mean()

            data['Index_Pax'] = data['Passengers'] / data['Month_Num'].map(base_pax) * 100
            data['Index_Flights'] = data['Flights'] / data['Month_Num'].map(base_atm) * 100

            window = data[(data['date'] >= ev['plot_start']) & (data['date'] <= ev['plot_end'])].copy().reset_index(drop=True)
            results[name] = {'window': window, 'ev': ev}
        return results

    comp_results = compute_comparison_data(df_schiphol)

    # Perspective 1 & 2 & 5: Passenger & Flight Trajectories side by side
    st.markdown("### 1. & 2. & 5. Passagiersontwikkeling, Vluchtaantallen & Schokintensiteit")
    fig_comp, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors_comp = {'main': '#1f4e79', 'flight': '#27ae60'}

    for ax, (name, res) in zip(axes, comp_results.items()):
        window = res['window']
        ev = res['ev']
        
        ax.set_title(f"{name} (Min Index: {ev['min_index']})", loc='left', fontsize=12, fontweight='bold', pad=22)
        ax.plot(window['date'], window['Index_Pax'], color=colors_comp['main'], linewidth=2, label='Passenger Index')
        ax.plot(window['date'], window['Index_Flights'], color=colors_comp['flight'], linewidth=1.5, linestyle='--', label='Flight Index (ATM)')
        
        if ev['span_label'] is not None:
            ax.axvspan(ev['shock_start'], ev['shock_end'], color='#eeeeee', zorder=0)
            ax.text(ev['shock_start'], 0.97, ' ' + ev['span_label'], transform=ax.get_xaxis_transform(),
                    va='top', fontsize=8, color='#555555',
                    bbox=dict(facecolor='#eeeeee', edgecolor='none', pad=1))
            
        ax.axvline(ev['event_date'], linestyle='--', color='#8c8c8c', linewidth=1)
        ax.axhline(100, linestyle=':', color='gray', linewidth=1)
        
        ax.set_ylabel("Index (Baseline = 100)" if ax == axes[0] else "")
        ax.grid(axis='y', color='#e6e6e6')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(ev['plot_start'], ev['plot_end'])
        ax.set_ylim(0, 160)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_comp)

    # Perspective 3 & 4: Model Comparison & Recovery Duration Matrix Table
    st.markdown("### 3. & 4. Modelvergelijking & Hersteltempo Overzicht")
    summary_table_data = [
        {
            "Crisis": "9/11 Attacks",
            "Modeltype (Perspectief 3)": "Anchored Model (5-year pre-shock average growth)",
            "Dieptepunt Index (Perspectief 5)": "80.8",
            "Maximale Daling": "19.2%",
            "Hersteltempo & Duur (Perspectief 4)": "Vrij snel herstel (< 12 maanden)"
        },
        {
            "Crisis": "Financiële Crisis",
            "Modeltype (Perspectief 3)": "Rolling Growth Model (year-on-year moving baseline)",
            "Dieptepunt Index (Perspectief 5)": "81.6",
            "Maximale Daling": "18.4%",
            "Langdurige Nasleep (Perspectief 4)": "Langdurige en hardnekkige dip (24+ maanden)"
        },
        {
            "Crisis": "COVID-19 Pandemic",
            "Modeltype (Perspectief 3)": "Anchored Model with Forecast Guard",
            "Dieptepunt Index (Perspectief 5)": "2.3",
            "Maximale Daling": "97.7%",
            "Hersteltempo & Duur (Perspectief 4)": "Historische ineenstorting, meerjarig herstel"
        }
    ]
    st.table(pd.DataFrame(summary_table_data))