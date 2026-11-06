"""
Project Group 11: Schiphol Disruption Dashboard
Compares the impact of 9/11, the Financial Crisis and COVID-19 on traffic at Amsterdam Airport Schiphol.

Data
- covid_data/Monthly_traffic_data_1992-current.xlsx  Schiphol monthly traffic (1992 - now, incl. segments)
- covid_data/schiphol_maandelijks.csv                Monthly passengers & flights (1999 - now)
- US_transport_data/US data <year>.zip               US BTS T-100 international segments (1995 - 2006)
- financial_data/AEX Historische Data.csv            Monthly AEX index (2004 - 2015)
- covid_data/validatie_cbs_vs_schiphol.csv           CBS annual totals (validation)
- covid_data/vakanties_vs_schiphol.csv               CBS holidays abroad

The methods follow the notebooks (results 911, Financial_crisis_dataresults, results_covid, results_comparison).

Run with: python -m streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

st.set_page_config(page_title="Schiphol Disruption Dashboard", layout="wide")

# Streamlit >= 1.46 uses width='stretch', older versions use use_container_width=True
_version = tuple(int(x) for x in st.__version__.split('.')[:2])
FULL_WIDTH = {'width': 'stretch'} if _version >= (1, 46) else {'use_container_width': True}

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

colors = {
    'pax': '#2a78d6',       # passengers
    'flights': '#eb6834',   # flight movements / second series
    'third': '#1baf7a',
    'up': '#2a78d6',        # growth / above expectation
    'down': '#e34948',      # decline / below expectation
    'grey': '#8c8c8c',
    'span': 'rgba(140, 140, 140, 0.15)',
}

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Settings per event (taken from results_comparison.ipynb)
EVENTS = {
    '9/11': dict(
        event_date=pd.Timestamp('2001-09-01'),
        shock_start=pd.Timestamp('2001-09-01'),
        shock_end=pd.Timestamp('2001-09-01'),
        base_start=pd.Timestamp('1996-09-01'),     # five historical years
        plot_start=pd.Timestamp('1996-09-01'),
        plot_end=pd.Timestamp('2006-09-01'),
        base_label='Sep 1996 - Aug 2001',
        span_label=None,                           # no grey period, only the dashed line
        model='anchored',
        forecast_guard=False,
        base_year=2000,
        source='excel',                            # the CSV starts in 1999, so 9/11 uses the Excel file
        partial_last_year=False,                   # 9/11 notebook uses the full year 2006
        no_event='9/11',
        kind='Security-related',
        color='#2a78d6',
        description=(
            'The terrorist attacks of 11 September 2001 caused a sudden geopolitical shock. '
            'Intercontinental travel, especially to the United States, dropped immediately, '
            'while European traffic was hit less hard.'
        ),
    ),
    'Financial Crisis': dict(
        event_date=pd.Timestamp('2008-09-01'),
        shock_start=pd.Timestamp('2008-01-01'),
        shock_end=pd.Timestamp('2010-01-01'),
        base_start=pd.Timestamp('2004-01-01'),
        plot_start=pd.Timestamp('2004-01-01'),
        plot_end=pd.Timestamp('2015-01-01'),
        base_label='2004-2007',
        span_label='Financial Crisis',
        model='rolling',
        forecast_guard=False,
        base_year=2007,
        source='csv',
        no_event='the financial crisis',
        kind='Economic',
        color='#eb6834',
        description=(
            'The collapse of Lehman Brothers in September 2008 started a global recession. '
            'Demand for business and leisure travel fell gradually and the dip lasted much longer '
            'than after 9/11.'
        ),
    ),
    'COVID-19': dict(
        event_date=pd.Timestamp('2020-03-01'),
        shock_start=pd.Timestamp('2020-01-01'),
        shock_end=pd.Timestamp('2022-12-01'),
        base_start=pd.Timestamp('2015-01-01'),
        plot_start=pd.Timestamp('2015-01-01'),
        plot_end=None,                             # newest month in the data (set after loading)
        base_label='2015-2019',
        span_label='COVID-19 (2020-2022)',
        model='anchored',
        forecast_guard=True,
        base_year=2019,
        source='csv',
        no_event='coronavirus',
        kind='Health-related',
        color='#1baf7a',
        description=(
            'The first lockdown in March 2020 and the travel restrictions that followed caused '
            'the deepest collapse in Schiphol history. Recovery took several years.'
        ),
    ),
}


# ---------------------------------------------------------------------------
# 2. Loading data
# ---------------------------------------------------------------------------
@st.cache_data
def load_excel():
    path = DATA_DIR / 'covid_data' / 'Monthly_traffic_data_1992-current.xlsx'
    df = pd.read_excel(path, header=[7, 8])
    data = df[
        [
            ('Unnamed: 0_level_0', 'Year'),
            ('Unnamed: 1_level_0', 'Month'),
            ('Passengers', 'Total*'),
            ('Air Transport Movements', 'Total'),
            ('Passengers', 'Europe'),
            ('Passengers', 'Intercontinental'),
            ('Passengers', 'O&D'),
            ('Passengers', 'Transfer'),
            ('Cargo', 'Cargo (tonnes)'),
        ]
    ].copy()
    data.columns = ['Year', 'Month', 'Passengers', 'Flights', 'Pax_Europe',
                    'Pax_Intercontinental', 'Pax_OD', 'Pax_Transfer', 'Cargo']
    data['Year'] = data['Year'].ffill()
    data = data[data['Month'].isin(MONTHS)].copy()

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data = data.dropna(subset=['Year'])
    data['Year'] = data['Year'].astype(int)
    data['Date'] = pd.to_datetime(data['Year'].astype(str) + ' ' + data['Month'], format='%Y %B')

    value_cols = ['Passengers', 'Flights', 'Pax_Europe', 'Pax_Intercontinental',
                  'Pax_OD', 'Pax_Transfer', 'Cargo']
    data[value_cols] = data[value_cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=['Passengers'])
    data['Month_Num'] = data['Date'].dt.month
    return data.sort_values('Date').reset_index(drop=True).drop(columns='Month')


@st.cache_data
def load_csv():
    data = pd.read_csv(DATA_DIR / 'covid_data' / 'schiphol_maandelijks.csv', parse_dates=['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month_Num'] = data['Date'].dt.month
    return data.sort_values('Date').reset_index(drop=True)


@st.cache_data
def load_us_bts():
    # Monthly passengers on all flights between Schiphol (AMS) and the US
    dfs = []
    for year in range(1995, 2007):
        file = DATA_DIR / 'US_transport_data' / f'US data {year}.zip'
        if file.exists():
            dfs.append(pd.read_csv(file, compression='zip'))
    bts = pd.concat(dfs, ignore_index=True)
    bts_ams = bts[(bts['ORIGIN'] == 'AMS') | (bts['DEST'] == 'AMS')]

    monthly = bts_ams.groupby(['YEAR', 'MONTH'])['PASSENGERS'].sum().reset_index()
    monthly['Date'] = pd.to_datetime({'year': monthly['YEAR'], 'month': monthly['MONTH'], 'day': 1})
    monthly = monthly.sort_values('Date').reset_index(drop=True)
    monthly['YoY_US'] = monthly['PASSENGERS'].pct_change(12) * 100
    return monthly.rename(columns={'PASSENGERS': 'US_Passengers'})


@st.cache_data
def load_aex():
    aex = pd.read_csv(DATA_DIR / 'financial_data' / 'AEX Historische Data.csv', encoding='utf-8-sig')
    aex['Date'] = pd.to_datetime(aex['Datum'], format='%d-%m-%Y')
    aex['AEX'] = (aex['Laatste'].astype(str)
                  .str.replace('.', '', regex=False)
                  .str.replace(',', '.', regex=False)
                  .astype(float))
    aex = aex[['Date', 'AEX']].sort_values('Date').reset_index(drop=True)
    aex['YoY_AEX'] = aex['AEX'].pct_change(12) * 100
    return aex


@st.cache_data
def load_cbs():
    cbs = pd.read_csv(DATA_DIR / 'covid_data' / 'validatie_cbs_vs_schiphol.csv')
    vakanties = pd.read_csv(DATA_DIR / 'covid_data' / 'vakanties_vs_schiphol.csv', encoding='utf-8-sig')
    return cbs, vakanties


try:
    sources = {'excel': load_excel(), 'csv': load_csv()}
except FileNotFoundError as e:
    st.error(f'Could not find a data file: {e}. Run the app from the project folder.')
    st.stop()

# COVID: plot up to the newest month in the data
EVENTS['COVID-19']['plot_end'] = sources['csv']['Date'].max()


# ---------------------------------------------------------------------------
# 3. Analysis: baseline, index, year-on-year and prediction model
# ---------------------------------------------------------------------------
def expected_anchored(data, ev, year, month, n_growth_years=5):
    # 9/11 and COVID notebooks: grow from the last normal year with the average growth of the 5 years before
    last_normal_year = ev['shock_start'].year - 1
    anchor = min(year - 1, last_normal_year)
    if ev['forecast_guard'] and anchor < ev['base_start'].year:
        return np.nan
    growth_rows = data[(data['Month_Num'] == month) & (data['Year'] > anchor - n_growth_years) & (data['Year'] <= anchor)]
    avg_growth = growth_rows['YoY'].mean() / 100
    anchor_rows = data[(data['Year'] == anchor) & (data['Month_Num'] == month)]
    if anchor_rows.empty:
        return np.nan
    return anchor_rows['Passengers'].values[0] * (1 + avg_growth) ** (year - anchor)


def expected_rolling(data, year, month):
    # Financial crisis notebook: previous year times the average growth of the 5 years before
    train = data[(data['Year'] >= year - 5) & (data['Year'] <= year - 1) & (data['Month_Num'] == month)]
    avg_growth = train['YoY'].mean() / 100
    prev = data[(data['Year'] == year - 1) & (data['Month_Num'] == month)]
    if prev.empty:
        return np.nan
    return prev['Passengers'].values[0] * (1 + avg_growth)


def seasonal_index(data, column, ev):
    # 100 = average of the same calendar month in the baseline period
    before = data[(data['Date'] >= ev['base_start']) & (data['Date'] < ev['shock_start'])]
    base = before.groupby('Month_Num')[column].mean()
    return data[column] / data['Month_Num'].map(base) * 100


@st.cache_data
def prepare_event(name):
    ev = EVENTS[name]
    data = sources[ev['source']].copy()
    event_date = ev['event_date']

    data['Index_Pax'] = seasonal_index(data, 'Passengers', ev)
    data['Index_Flights'] = seasonal_index(data, 'Flights', ev)
    data['PreviousYear'] = data.groupby('Month_Num')['Passengers'].shift(1)
    data['YoY'] = (data['Passengers'] / data['PreviousYear'] - 1) * 100
    data['Months_From_Event'] = ((data['Date'].dt.year - event_date.year) * 12
                                 + data['Date'].dt.month - event_date.month)

    # Prediction model: computed for the plot window and the comparison range (-24 to +72 months)
    start = min(ev['plot_start'], event_date - pd.DateOffset(months=24))
    end = max(ev['plot_end'], event_date + pd.DateOffset(months=72))
    in_range = (data['Date'] >= start) & (data['Date'] <= end)
    predicted = pd.Series(np.nan, index=data.index)
    for i, row in data[in_range].iterrows():
        if ev['model'] == 'anchored':
            predicted[i] = expected_anchored(data, ev, row['Year'], row['Month_Num'])
        else:
            predicted[i] = expected_rolling(data, row['Year'], row['Month_Num'])
    data['Predicted'] = predicted
    data['Deviation'] = (data['Passengers'] - data['Predicted']) / data['Predicted'] * 100

    window = data[(data['Date'] >= ev['plot_start']) & (data['Date'] <= ev['plot_end'])].reset_index(drop=True)
    return data, window


def recovery_month(data, ev, trough_date):
    # First month after the lowest point in which passengers are back at the level of the same month
    # in the last pre-crisis year, for three months in a row
    base = data[data['Year'] == ev['base_year']].set_index('Month_Num')['Passengers']
    after = data[data['Date'] >= trough_date].copy()
    after['Recovered'] = after['Passengers'] >= after['Month_Num'].map(base)
    streak = after['Recovered'].rolling(3).sum() == 3
    if not streak.any():
        return None
    return after.loc[streak.idxmax() - 2]   # first month of the three-month streak


@st.cache_data
def event_kpis(name):
    ev = EVENTS[name]
    data, window = prepare_event(name)
    event_date = ev['event_date']

    post = data[(data['Date'] >= event_date) & (data['Date'] < event_date + pd.DateOffset(months=36))]
    trough = post.loc[post['Index_Pax'].idxmin()]
    worst_dev = post.loc[post['Deviation'].idxmin()]

    first12 = data[(data['Date'] >= event_date) & (data['Date'] < event_date + pd.DateOffset(months=12))]
    worst_yoy = first12.loc[first12['YoY'].idxmin()]
    change12 = (first12['Passengers'].sum() / first12['PreviousYear'].sum() - 1) * 100

    rec = recovery_month(data, ev, trough['Date'])
    latest = data.iloc[-1]
    base = data[data['Year'] == ev['base_year']].set_index('Month_Num')['Passengers']
    latest_vs_base = latest['Passengers'] / base[latest['Month_Num']] * 100

    return {
        'trough_index': trough['Index_Pax'],
        'trough_date': trough['Date'],
        'trough_months': int(trough['Months_From_Event']),
        'worst_yoy': worst_yoy['YoY'],
        'worst_yoy_date': worst_yoy['Date'],
        'change12': change12,
        'neg_months': int((first12['YoY'] < 0).sum()),
        'worst_dev': worst_dev['Deviation'],
        'worst_dev_date': worst_dev['Date'],
        'recovery_date': None if rec is None else rec['Date'],
        'recovery_months': None if rec is None else int(rec['Months_From_Event']),
        'latest_date': latest['Date'],
        'latest_vs_base': latest_vs_base,
    }


def fmt_month(date):
    return date.strftime('%b %Y')


def fmt_recovery(k):
    if k['recovery_months'] is None:
        return 'Not yet'
    return f"{k['recovery_months']} months ({fmt_month(k['recovery_date'])})"


# ---------------------------------------------------------------------------
# 4. Plot helpers (one consistent layout for all charts)
# ---------------------------------------------------------------------------
def style(fig, title, ytitle, subtitle=None, height=380, legend=True):
    text = f'<b>{title}</b>'
    if subtitle:
        text += f"<br><span style='font-size:12px;color:#777'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=text, x=0, xanchor='left'),
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode='x unified',
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, x=0),
        bargap=0.15,
    )
    fig.update_yaxes(title_text=ytitle, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def mark_event(fig, ev, row=None, col=None, label=True):
    # Grey area for the crisis period and a dashed line at the event itself
    # (annotations are added separately: plotly cannot place them on datetime shapes)
    if ev['span_label'] is not None:
        fig.add_vrect(x0=ev['shock_start'], x1=ev['shock_end'], fillcolor=colors['span'],
                      line_width=0, layer='below', row=row, col=col)
        if label:
            fig.add_annotation(x=ev['shock_start'], y=1, yref='paper', text=ev['span_label'],
                               showarrow=False, xanchor='left', yanchor='bottom',
                               font=dict(size=10, color='#777'))
    fig.add_vline(x=ev['event_date'], line_dash='dash', line_color=colors['grey'],
                  line_width=1, row=row, col=col)
    fig.update_xaxes(range=[ev['plot_start'], ev['plot_end']])


def diverging_bars(dates, values, name, customdata=None, hovertemplate=None):
    return go.Bar(
        x=dates, y=values, name=name,
        marker_color=np.where(values >= 0, colors['up'], colors['down']),
        customdata=customdata,
        hovertemplate=hovertemplate or '%{y:.1f}%<extra></extra>',
    )


def show(fig):
    st.plotly_chart(fig, theme='streamlit', **FULL_WIDTH)


# ---------------------------------------------------------------------------
# 5. Charts per event (same six charts as the notebooks)
# ---------------------------------------------------------------------------
def chart_raw(window, ev):
    fig = go.Figure(go.Scatter(x=window['Date'], y=window['Passengers'] / 1e6, name='Passengers',
                               line=dict(color=colors['pax'], width=2),
                               hovertemplate='%{y:.2f} million<extra></extra>'))
    mark_event(fig, ev)
    return style(fig, 'Passengers per month', 'Passengers (millions)', 'Schiphol, raw numbers', legend=False)


def chart_index(window, ev, show_flights=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window['Date'], y=window['Index_Pax'], name='Passengers',
                             line=dict(color=colors['pax'], width=2),
                             hovertemplate='%{y:.1f}<extra>Passengers</extra>'))
    if show_flights:
        fig.add_trace(go.Scatter(x=window['Date'], y=window['Index_Flights'], name='Flight movements',
                                 line=dict(color=colors['flights'], width=2),
                                 hovertemplate='%{y:.1f}<extra>Flights</extra>'))
    fig.add_hline(y=100, line_dash='dot', line_color=colors['grey'], line_width=1)
    mark_event(fig, ev)
    return style(fig, 'Passengers versus flight movements' if show_flights else 'Passengers compared to baseline',
                 'Index', f"Seasonally adjusted index ({ev['base_label']} = 100)")


def chart_yoy(window, ev):
    fig = go.Figure(diverging_bars(window['Date'], window['YoY'].values, 'Year-on-year change'))
    fig.add_hline(y=0, line_color=colors['grey'], line_width=1)
    mark_event(fig, ev)
    fig = style(fig, 'Year-on-year change in passengers', 'Change (%)',
                'Compared to the same month in the previous year (blue = growth, red = decline)', legend=False)
    if ev['plot_end'] > pd.Timestamp('2021-01-01'):
        # Recovery from a nearly empty 2020 gives peaks of +600%; clip so the rest stays readable
        fig.update_yaxes(range=[-110, 250])
    return fig


def chart_deviation(window, ev):
    w = window.dropna(subset=['Deviation'])
    model_name = ('Anchored model: last normal year + average 5-year growth' if ev['model'] == 'anchored'
                  else 'Rolling model: previous year + average 5-year growth')
    fig = go.Figure(diverging_bars(
        w['Date'], w['Deviation'].values, 'Deviation',
        customdata=np.stack([w['Passengers'] / 1e6, w['Predicted'] / 1e6], axis=-1),
        hovertemplate='%{y:.1f}%<br>Actual: %{customdata[0]:.2f}M<br>Expected: %{customdata[1]:.2f}M<extra></extra>',
    ))
    fig.add_hline(y=0, line_color=colors['grey'], line_width=1)
    mark_event(fig, ev)
    return style(fig, f"Deviation from the forecast model (without {ev['no_event']})", 'Deviation (%)',
                 model_name, legend=False)


def chart_annual(data, ev):
    base_year = ev['base_year']
    last_date = ev['plot_end']
    if not ev.get('partial_last_year', True):
        last_date = pd.Timestamp(year=last_date.year, month=12, day=1)
    last_date = min(last_date, data['Date'].max())
    years = list(range(ev['plot_start'].year, last_date.year + 1))
    totals = data.groupby('Year')['Passengers'].sum()
    month_count = data.groupby('Year')['Passengers'].count()

    percentages, partial = [], []
    for y in years:
        if month_count.get(y, 0) < 12 or (y == last_date.year and last_date.month < 12):
            # incomplete year: only the months we have, compared to the same months in the base year
            months = data[(data['Year'] == y) & (data['Date'] <= last_date)]['Month_Num']
            this_year = data[(data['Year'] == y) & data['Month_Num'].isin(months)]['Passengers'].sum()
            same_base = data[(data['Year'] == base_year) & data['Month_Num'].isin(months)]['Passengers'].sum()
            percentages.append(this_year / same_base * 100)
            partial.append(True)
        else:
            percentages.append(totals[y] / totals[base_year] * 100)
            partial.append(False)

    phase_colors = ['#2a78d6' if y < ev['shock_start'].year
                    else '#e34948' if y <= ev['shock_end'].year
                    else '#eb6834' for y in years]
    fig = go.Figure(go.Bar(
        x=years, y=percentages, marker_color=phase_colors,
        marker_pattern_shape=['/' if p else '' for p in partial],
        text=[f'{p:.0f}%' for p in percentages], textposition='outside',
        hovertemplate='%{x}: %{y:.1f}% of ' + str(base_year) + '<extra></extra>',
    ))
    fig.add_hline(y=100, line_dash='dot', line_color=colors['grey'], line_width=1)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, max(125, max(percentages) + 15)])
    return style(fig, 'Yearly total passengers', f'% of {base_year}',
                 f"As % of {base_year} (blue = before, red = crisis, orange = after, hatched = incomplete year)",
                 legend=False)


SEGMENTS = {
    'Europe vs. intercontinental': [('Pax_Europe', 'Europe'), ('Pax_Intercontinental', 'Intercontinental')],
    'Origin & destination vs. transfer': [('Pax_OD', 'O&D'), ('Pax_Transfer', 'Transfer')],
    'Passengers vs. cargo': [('Passengers', 'Passengers'), ('Cargo', 'Cargo (tonnes)')],
}


def chart_segments(ev, choice):
    # The CSV has no segments, so this chart always uses the Excel file (same baseline method)
    data = sources['excel'].copy()
    window = data[(data['Date'] >= ev['plot_start']) & (data['Date'] <= ev['plot_end'])]
    fig = go.Figure()
    for (column, label), color in zip(SEGMENTS[choice], [colors['pax'], colors['flights']]):
        index = seasonal_index(data, column, ev)[window.index]
        fig.add_trace(go.Scatter(x=window['Date'], y=index, name=label, line=dict(color=color, width=2),
                                 hovertemplate='%{y:.1f}<extra>' + label + '</extra>'))
    fig.add_hline(y=100, line_dash='dot', line_color=colors['grey'], line_width=1)
    mark_event(fig, ev)
    return style(fig, f'Traffic segments: {choice.lower()}', 'Index',
                 f"Seasonally adjusted index ({ev['base_label']} = 100)")


# ---------------------------------------------------------------------------
# 6. Event-specific extra analyses
# ---------------------------------------------------------------------------
def extra_911(data, ev):
    st.markdown('#### US routes vs. total Schiphol traffic (US BTS T-100 data)')
    st.caption('Passengers on all flights between Schiphol and the United States, from the '
               'US Bureau of Transportation Statistics. This shows whether the transatlantic market '
               'was hit harder than Schiphol as a whole.')
    try:
        us = load_us_bts()
    except (FileNotFoundError, ValueError):
        st.info('US data files not found in US_transport_data/.')
        return

    comparison = pd.merge(data[['Date', 'YoY', 'Passengers', 'Month_Num']], us, on='Date')
    comparison['Index_US'] = seasonal_index(comparison, 'US_Passengers', ev)
    comparison['Index_Total'] = seasonal_index(comparison, 'Passengers', ev)
    comparison['US_Share'] = comparison['US_Passengers'] / comparison['Passengers'] * 100
    comparison = comparison[comparison['Date'] >= ev['plot_start']]

    first12 = comparison[(comparison['Date'] >= ev['event_date']) &
                         (comparison['Date'] < ev['event_date'] + pd.DateOffset(months=12))]
    prev12 = comparison[(comparison['Date'] >= ev['event_date'] - pd.DateOffset(months=12)) &
                        (comparison['Date'] < ev['event_date'])]
    us_change = (first12['US_Passengers'].sum() / prev12['US_Passengers'].sum() - 1) * 100
    total_change = (first12['Passengers'].sum() / prev12['Passengers'].sum() - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric('US routes: first 12 months vs. year before', f'{us_change:.1f}%')
    c2.metric('All Schiphol traffic: same period', f'{total_change:.1f}%')
    c3.metric('US share of Schiphol passengers (12 months before)',
              f"{prev12['US_Passengers'].sum() / prev12['Passengers'].sum() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison['Date'], y=comparison['YoY'], name='Total Schiphol traffic',
                                 line=dict(color=colors['pax'], width=2), hovertemplate='%{y:.1f}%'))
        fig.add_trace(go.Scatter(x=comparison['Date'], y=comparison['YoY_US'], name='Schiphol-US traffic',
                                 line=dict(color=colors['flights'], width=2), hovertemplate='%{y:.1f}%'))
        fig.add_hline(y=0, line_color=colors['grey'], line_width=1)
        mark_event(fig, ev)
        show(style(fig, 'Year-on-year passenger growth', 'Change (%)', 'US routes vs. all traffic'))
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison['Date'], y=comparison['Index_Total'], name='Total Schiphol traffic',
                                 line=dict(color=colors['pax'], width=2), hovertemplate='%{y:.1f}'))
        fig.add_trace(go.Scatter(x=comparison['Date'], y=comparison['Index_US'], name='Schiphol-US traffic',
                                 line=dict(color=colors['flights'], width=2), hovertemplate='%{y:.1f}'))
        fig.add_hline(y=100, line_dash='dot', line_color=colors['grey'], line_width=1)
        mark_event(fig, ev)
        show(style(fig, 'US routes compared to baseline', 'Index',
                   f"Seasonally adjusted index ({ev['base_label']} = 100)"))


def extra_financial(data, ev):
    st.markdown('#### Schiphol traffic vs. the AEX stock market index')
    st.caption('Did passenger traffic follow the stock market? Shown as two panels with a shared '
               'time axis instead of one chart with two y-axes, so neither scale distorts the other.')
    try:
        aex = load_aex()
    except FileNotFoundError:
        st.info('AEX data file not found in financial_data/.')
        return

    merged = pd.merge(data[['Date', 'Deviation', 'YoY']], aex, on='Date', how='inner')
    merged = merged[(merged['Date'] >= ev['plot_start']) & (merged['Date'] <= ev['plot_end'])]

    both = merged.dropna(subset=['YoY', 'YoY_AEX'])
    corr = both['YoY'].corr(both['YoY_AEX'])
    # The stock market usually moves first: find the lag (in months) with the strongest correlation
    lags = {lag: both['YoY'].corr(both['YoY_AEX'].shift(lag)) for lag in range(0, 13)}
    best_lag = max(lags, key=lambda lag: lags[lag])

    c1, c2, c3 = st.columns(3)
    c1.metric('AEX fall, peak to trough', f"{(merged['AEX'].min() / merged['AEX'].max() - 1) * 100:.0f}%")
    c2.metric('Correlation YoY passengers vs. YoY AEX', f'{corr:.2f}')
    c3.metric('Strongest correlation when AEX leads by', f'{best_lag} months', f'r = {lags[best_lag]:.2f}',
              delta_color='off')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=('Schiphol: deviation from the forecast model (%)', 'AEX index level'))
    fig.add_trace(diverging_bars(merged['Date'], merged['Deviation'].values, 'Schiphol deviation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged['Date'], y=merged['AEX'], name='AEX',
                             line=dict(color=colors['pax'], width=2), hovertemplate='%{y:.0f}<extra>AEX</extra>'),
                  row=2, col=1)
    for row in (1, 2):
        mark_event(fig, ev, row=row, col=1, label=False)
    fig.update_layout(height=520)
    show(style(fig, 'Passenger traffic and the stock market', '', height=520, legend=False))


def extra_covid(data, ev):
    st.markdown('#### Validation with CBS and holidays abroad')
    try:
        cbs, vakanties = load_cbs()
    except FileNotFoundError:
        st.info('CBS data files not found in covid_data/.')
        return

    year_totals = data.groupby('Year')[['Passengers', 'Flights']].sum()
    val = cbs.join(year_totals, on='jaar')
    val['Passengers diff (%)'] = (val['Passengers'] - val['pax_CBS']) / val['pax_CBS'] * 100
    val['Flights diff (%)'] = (val['Flights'] - val['atm_CBS']) / val['atm_CBS'] * 100

    # CBS counts holiday years (1 Oct - 30 Sep) up to 2020 and calendar years from 2021; count flights the same way
    d = data.copy()
    d['CBS_year'] = d['Year'].astype(float)
    d.loc[(d['Year'] <= 2019) & (d['Month_Num'] >= 10), 'CBS_year'] += 1
    d.loc[(d['Year'] == 2020) & (d['Month_Num'] >= 10), 'CBS_year'] = np.nan
    cbs_flights = d.groupby('CBS_year')['Flights'].sum()
    cbs_flights.index = cbs_flights.index.astype(int)
    flights_idx = cbs_flights / cbs_flights.loc[ev['base_year']] * 100
    years = vakanties['jaar']
    flights_vak = flights_idx.reindex(years).values
    corr = np.corrcoef(flights_vak, vakanties['buitenl_vakanties_idx_2019=100'])[0, 1]

    c1, c2, c3 = st.columns(3)
    c1.metric('Largest difference with CBS (passengers)', f"{val['Passengers diff (%)'].abs().max():.4f}%")
    c2.metric('Largest difference with CBS (flights)', f"{val['Flights diff (%)'].abs().max():.4f}%")
    c3.metric('Correlation flights vs. holidays abroad', f'{corr:.2f}')

    left, right = st.columns(2)
    with left:
        labels = [f'{y}*' if y == 2025 else str(y) for y in years]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=flights_vak, name='Flights Schiphol', mode='lines+markers',
                                 line=dict(color=colors['pax'], width=2), marker_size=8))
        fig.add_trace(go.Scatter(x=labels, y=vakanties['buitenl_vakanties_idx_2019=100'], name='Holidays abroad',
                                 mode='lines+markers', line=dict(color=colors['flights'], width=2), marker_size=8))
        fig.add_trace(go.Scatter(x=labels, y=vakanties['buitenl_deelnemers_idx_2019=100'],
                                 name='Dutch people holidaying abroad', mode='lines+markers',
                                 line=dict(color=colors['third'], width=2), marker_size=8))
        fig.add_hline(y=100, line_dash='dot', line_color=colors['grey'], line_width=1)
        show(style(fig, 'Flights from Schiphol and holidays abroad (CBS)', 'Index (2019 = 100)',
                   'Holiday years (Oct-Sep) up to 2020, calendar years after; * = provisional'))
    with right:
        per_participant = vakanties['buitenl_vakanties_x1000'] / vakanties['buitenl_deelnemers_x1000']
        fig = go.Figure(go.Bar(
            x=labels, y=per_participant,
            marker_color=['#e34948' if y in (2020, 2021) else colors['pax'] for y in years],
            text=[f'{v:.2f}' for v in per_participant], textposition='outside',
            hovertemplate='%{y:.2f} holidays<extra></extra>',
        ))
        fig.update_yaxes(range=[0, 2.6])
        show(style(fig, 'Holidays abroad per participant', 'Holidays per person',
                   'Did people who still travelled go less often? (red = 2020-2021)', legend=False))

    with st.expander('Validation table: project data vs. CBS'):
        st.dataframe(
            val[['jaar', 'Passengers', 'pax_CBS', 'Passengers diff (%)', 'Flights', 'atm_CBS', 'Flights diff (%)']]
            .rename(columns={'jaar': 'Year', 'pax_CBS': 'Passengers (CBS)', 'atm_CBS': 'Flights (CBS)'}),
            hide_index=True, **FULL_WIDTH,
        )


EXTRAS = {'9/11': extra_911, 'Financial Crisis': extra_financial, 'COVID-19': extra_covid}


# ---------------------------------------------------------------------------
# 7. Interactive map
# ---------------------------------------------------------------------------
# Airport coordinates (name, lat, lon) for the US airports with regular Schiphol traffic in the BTS data.
# Smaller airports (charters, diversions; < 0.1% of passengers) are counted in the totals but not drawn.
AIRPORTS = {
    'AMS': ('Amsterdam Schiphol', 52.3086, 4.7639),
    'DTW': ('Detroit', 42.2124, -83.3534),
    'JFK': ('New York JFK', 40.6413, -73.7781),
    'MSP': ('Minneapolis', 44.8848, -93.2223),
    'IAD': ('Washington Dulles', 38.9531, -77.4565),
    'EWR': ('Newark', 40.6895, -74.1745),
    'ORD': ('Chicago', 41.9742, -87.9073),
    'LAX': ('Los Angeles', 33.9416, -118.4085),
    'IAH': ('Houston', 29.9902, -95.3368),
    'ATL': ('Atlanta', 33.6407, -84.4277),
    'BOS': ('Boston', 42.3656, -71.0096),
    'SFO': ('San Francisco', 37.6213, -122.3790),
    'MIA': ('Miami', 25.7959, -80.2870),
    'MEM': ('Memphis', 35.0424, -89.9767),
    'SEA': ('Seattle', 47.4502, -122.3088),
    'MCO': ('Orlando', 28.4312, -81.3081),
    'PHL': ('Philadelphia', 39.8744, -75.2424),
    'CVG': ('Cincinnati', 39.0488, -84.6678),
    'OAK': ('Oakland', 37.7126, -122.2197),
    'TPA': ('Tampa', 27.9755, -82.5332),
    'SJU': ('San Juan', 18.4394, -66.0018),
    'DEN': ('Denver', 39.8561, -104.6737),
}

# Countries coloured as 'Europe' on the regional map. Schiphol only reports Europe vs. intercontinental,
# so the colouring shows the region as a whole, not individual countries.
EUROPE_ISO3 = [
    'ALB', 'AND', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST',
    'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'LTU', 'LUX', 'LVA', 'MDA', 'MKD',
    'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB', 'SVK', 'SVN', 'SWE', 'UKR',
]

# Months shown on the map slider per event
MAP_WINDOWS = {
    '9/11': (pd.Timestamp('2000-01-01'), pd.Timestamp('2004-12-01')),
    'Financial Crisis': (pd.Timestamp('2007-01-01'), pd.Timestamp('2012-12-01')),
    'COVID-19': (pd.Timestamp('2019-01-01'), None),       # up to the newest month
}

# red (decline) - grey (no change) - blue (growth)
DIVERGING = [[0, '#e34948'], [0.5, '#f0efec'], [1, '#2a78d6']]


def change_vs_reference(df, column, ev, reference):
    # Percentage change compared to the same month one year earlier, or to the same month in the
    # last pre-crisis year (2000 / 2007 / 2019)
    if reference == 'previous year':
        ref = df.groupby('Month_Num')[column].shift(1)
    else:
        base = df[df['Year'] == ev['base_year']].set_index('Month_Num')[column]
        ref = df['Month_Num'].map(base)
    return (df[column] / ref - 1) * 100


@st.cache_data
def us_routes():
    # Monthly passengers per route between Schiphol and US airports
    dfs = []
    for year in range(1995, 2007):
        file = DATA_DIR / 'US_transport_data' / f'US data {year}.zip'
        if file.exists():
            dfs.append(pd.read_csv(file, compression='zip'))
    bts = pd.concat(dfs, ignore_index=True)
    bts = bts[(bts['ORIGIN'] == 'AMS') | (bts['DEST'] == 'AMS')].copy()
    bts['Airport'] = bts['ORIGIN'].where(bts['DEST'] == 'AMS', bts['DEST'])
    routes = bts.groupby(['Airport', 'YEAR', 'MONTH'])['PASSENGERS'].sum().reset_index()
    routes['Date'] = pd.to_datetime({'year': routes['YEAR'], 'month': routes['MONTH'], 'day': 1})

    # Months without flights get 0 passengers, so every route has a complete series
    full = pd.MultiIndex.from_product([sorted(routes['Airport'].unique()), sorted(routes['Date'].unique())],
                                      names=['Airport', 'Date'])
    routes = (routes.set_index(['Airport', 'Date'])['PASSENGERS'].reindex(full, fill_value=0)
              .reset_index().rename(columns={'PASSENGERS': 'Passengers'}))
    routes['Year'] = routes['Date'].dt.year
    routes['Month_Num'] = routes['Date'].dt.month
    return routes


@st.cache_data
def route_changes(name, reference):
    ev = EVENTS[name]
    parts = []
    for airport, df in us_routes().groupby('Airport'):
        df = df.sort_values('Date').copy()
        df['Change'] = change_vs_reference(df, 'Passengers', ev, reference)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def color_for(value, limit):
    if pd.isna(value) or np.isinf(value):
        return '#bbbbbb'
    position = (np.clip(value, -limit, limit) + limit) / (2 * limit)
    return sample_colorscale(DIVERGING, position)[0]


def geo_layout(fig, lon_range, lat_range, height=520):
    fig.update_geos(
        projection_type='natural earth', showland=True, landcolor='#e9e8e4',
        showocean=True, oceancolor='#f4f7fa', showcountries=True, countrycolor='#ffffff',
        showcoastlines=False, showframe=False, lonaxis_range=lon_range, lataxis_range=lat_range,
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    return fig


def route_map(name, month, reference, limit):
    # Line width = passengers on the route that month (same scale for every month, so widths can be
    # compared over time); colour = change compared to the reference month
    routes = route_changes(name, reference)
    routes = routes[routes['Airport'].isin(AIRPORTS)]
    max_pax = routes['Passengers'].max()
    now = routes[(routes['Date'] == month) & (routes['Passengers'] > 0)].sort_values('Passengers')
    _, ams_lat, ams_lon = AIRPORTS['AMS']

    fig = go.Figure()
    for _, r in now.iterrows():
        _, lat, lon = AIRPORTS[r['Airport']]
        fig.add_trace(go.Scattergeo(
            lon=[ams_lon, lon], lat=[ams_lat, lat], mode='lines', opacity=0.85, hoverinfo='skip',
            line=dict(width=1 + 9 * np.sqrt(r['Passengers'] / max_pax), color=color_for(r['Change'], limit)),
        ))

    change = now['Change'].replace([np.inf, -np.inf], np.nan)
    fig.add_trace(go.Scattergeo(
        lon=[AIRPORTS[a][2] for a in now['Airport']], lat=[AIRPORTS[a][1] for a in now['Airport']],
        mode='markers',
        marker=dict(size=9, color=[color_for(c, limit) for c in change], line=dict(width=1, color='white')),
        customdata=np.stack([[AIRPORTS[a][0] for a in now['Airport']], now['Passengers'], change], axis=-1),
        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} passengers'
                      '<br>%{customdata[2]:+.1f}%<extra></extra>',
    ))
    # Invisible trace, only for the colour bar (grey dots = no flights in the reference month)
    fig.add_trace(go.Scattergeo(lon=[ams_lon], lat=[ams_lat], mode='markers', hoverinfo='skip',
                                marker=dict(size=0.1, color=[0], colorscale=DIVERGING, cmin=-limit, cmax=limit,
                                            colorbar=dict(title='Change (%)', thickness=12, len=0.6))))
    fig.add_trace(go.Scattergeo(lon=[ams_lon], lat=[ams_lat], mode='markers+text', text=['Schiphol'],
                                textposition='top center', marker=dict(size=12, color='#0b0b0b'),
                                hoverinfo='skip'))
    fig.update_layout(title=dict(text=f'<b>Schiphol - US routes, {month:%B %Y}</b>', x=0))
    return geo_layout(fig, [-130, 15], [12, 72])


def region_map(regions, month, limit):
    # Europe and the rest of the world coloured by the change in Schiphol passengers for that region
    row = regions[regions['Date'] == month].iloc[0]
    europe, interc = row['Change_Europe'], row['Change_Intercontinental']
    others = [c for c in WORLD_ISO3 if c not in EUROPE_ISO3]

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=EUROPE_ISO3, z=[europe] * len(EUROPE_ISO3), zmin=-limit, zmax=limit,
        colorscale=DIVERGING, marker_line_color='white', marker_line_width=0.5,
        colorbar=dict(title='Change (%)', thickness=12, len=0.6),
        hovertemplate=f"Europe: {europe:+.1f}%<br>{row['Pax_Europe']:,.0f} passengers<extra></extra>",
    ))
    fig.add_trace(go.Choropleth(
        locations=others, z=[interc] * len(others), zmin=-limit, zmax=limit, colorscale=DIVERGING,
        showscale=False, marker_line_color='white', marker_line_width=0.3,
        hovertemplate=(f"Intercontinental: {interc:+.1f}%<br>{row['Pax_Intercontinental']:,.0f} passengers"
                       '<extra></extra>'),
    ))
    _, ams_lat, ams_lon = AIRPORTS['AMS']
    fig.add_trace(go.Scattergeo(
        lon=[ams_lon, 15, -40], lat=[ams_lat, 66, 18], mode='markers+text', hoverinfo='skip',
        text=['Schiphol', f'<b>Europe {europe:+.0f}%</b>', f'<b>Intercontinental {interc:+.0f}%</b>'],
        textposition=['bottom center', 'middle center', 'middle center'],
        textfont=dict(size=[11, 14, 14], color='#0b0b0b'),
        marker=dict(size=[12, 0, 0], color='#0b0b0b'),
    ))
    fig.update_layout(title=dict(text=f'<b>Schiphol passengers per region, {month:%B %Y}</b>', x=0))
    return geo_layout(fig, [-170, 190], [-58, 84], height=480)


# ISO-3 codes of (almost) all countries, used to colour 'the rest of the world'
WORLD_ISO3 = [
    'AFG', 'AGO', 'ARE', 'ARG', 'ARM', 'AUS', 'AZE', 'BDI', 'BEN', 'BFA', 'BGD', 'BHS', 'BLZ', 'BOL', 'BRA',
    'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COL', 'CRI', 'CUB', 'DJI',
    'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ETH', 'FJI', 'GAB', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRL',
    'GTM', 'GUY', 'HND', 'HTI', 'IDN', 'IND', 'IRN', 'IRQ', 'ISR', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ',
    'KHM', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LKA', 'LSO', 'MAR', 'MDG', 'MEX', 'MLI', 'MMR', 'MNG',
    'MOZ', 'MRT', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NPL', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL',
    'PNG', 'PRI', 'PRK', 'PRY', 'QAT', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SLB', 'SLE', 'SLV', 'SOM', 'SSD',
    'SUR', 'SWZ', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TTO', 'TUN', 'TUR', 'TWN', 'TZA', 'UGA',
    'URY', 'USA', 'UZB', 'VEN', 'VNM', 'VUT', 'YEM', 'ZAF', 'ZMB', 'ZWE', 'ESH', 'NCL',
]


def map_tab(reference_label):
    reference = 'previous year' if reference_label == 'Same month a year earlier' else 'pre-crisis'

    name = st.radio('Disruption', list(EVENTS), horizontal=True, key='map_event')
    ev = EVENTS[name]
    start, end = MAP_WINDOWS[name]
    end = end or sources['excel']['Date'].max()
    months = list(pd.date_range(start, end, freq='MS'))

    # Regional data from the Excel file (the CSV has no Europe / intercontinental split)
    regions = sources['excel'].copy()
    regions['Change_Total'] = change_vs_reference(regions, 'Passengers', ev, reference)
    regions['Change_Europe'] = change_vs_reference(regions, 'Pax_Europe', ev, reference)
    regions['Change_Intercontinental'] = change_vs_reference(regions, 'Pax_Intercontinental', ev, reference)
    regions = regions[(regions['Date'] >= start) & (regions['Date'] <= end)]

    # Colour scale wide enough for the event, so both small and large shocks stay readable
    limit = 100 if name == 'COVID-19' else 30

    # The slider starts at the moment of the largest impact
    default = pd.Timestamp('2001-10-01') if name == '9/11' else event_kpis(name)['trough_date']
    labels = [m.strftime('%b %Y') for m in months]
    label = st.select_slider('Month', options=labels, value=default.strftime('%b %Y'), key=f'map_month_{name}')
    month = months[labels.index(label)]
    ref_text = ('the same month a year earlier' if reference == 'previous year'
                else f"the same month in {ev['base_year']} (before the crisis)")
    st.caption(f'Colours show the change in passengers compared to {ref_text}: blue = more passengers, '
               'red = fewer. Move the slider to see the impact unfold.')

    row = regions[regions['Date'] == month].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('All Schiphol passengers', f"{row['Passengers'] / 1e6:.2f} M", f"{row['Change_Total']:+.1f}%")
    c2.metric('Europe', f"{row['Pax_Europe'] / 1e6:.2f} M", f"{row['Change_Europe']:+.1f}%")
    c3.metric('Intercontinental', f"{row['Pax_Intercontinental'] / 1e6:.2f} M",
              f"{row['Change_Intercontinental']:+.1f}%")

    if name == '9/11':
        try:
            routes = route_changes(name, reference)
        except (FileNotFoundError, ValueError):
            routes = None
        if routes is None:
            st.info('US data files not found in US_transport_data/, so only the regional map is shown.')
        else:
            ref_month = (month - pd.DateOffset(years=1) if reference == 'previous year'
                         else pd.Timestamp(year=ev['base_year'], month=month.month, day=1))
            us_now = routes[routes['Date'] == month]
            us_ref = routes[routes['Date'] == ref_month]['Passengers'].sum()
            c4.metric('US routes (BTS)', f"{us_now['Passengers'].sum() / 1e6:.2f} M",
                      f"{(us_now['Passengers'].sum() / us_ref - 1) * 100:+.1f}%")
            show(route_map(name, month, reference, limit))
            st.caption('Line width = passengers on the route in that month (same scale for every month). '
                       'Hover over an airport for the numbers.')

            with st.expander('Routes this month'):
                table = us_now[us_now['Passengers'] > 0].copy()
                table['Airport'] = table['Airport'].map(lambda a: f'{AIRPORTS[a][0]} ({a})' if a in AIRPORTS else a)
                table = table.sort_values('Passengers', ascending=False)[['Airport', 'Passengers', 'Change']]
                st.dataframe(table.rename(columns={'Change': 'Change (%)'}).round(1), hide_index=True, **FULL_WIDTH)

    show(region_map(regions, month, limit))
    st.caption("Schiphol only reports 'Europe' and 'intercontinental' passengers, so all countries in a region "
               'get the same colour.')

    # Timeline under the maps: where is the selected month in the crisis?
    fig = go.Figure()
    for column, label, color in [('Change_Total', 'All passengers', '#52514e'),
                                 ('Change_Europe', 'Europe', colors['pax']),
                                 ('Change_Intercontinental', 'Intercontinental', colors['flights'])]:
        fig.add_trace(go.Scatter(x=regions['Date'], y=regions[column], name=label, line=dict(color=color, width=2),
                                 hovertemplate='%{y:+.1f}%<extra>' + label + '</extra>'))
    fig.add_hline(y=0, line_color=colors['grey'], line_width=1)
    fig.add_vline(x=ev['event_date'], line_dash='dash', line_color=colors['grey'], line_width=1)
    fig.add_vline(x=month, line_color='#0b0b0b', line_width=2)
    show(style(fig, 'Timeline', 'Change (%)', f'Compared to {ref_text}; black line = selected month',
               height=320))


# ---------------------------------------------------------------------------
# 8. Page layout
# ---------------------------------------------------------------------------
st.title('Amsterdam Airport Schiphol: Disruption Dashboard')
st.markdown(
    'How did passenger traffic at Schiphol react to **9/11**, the **Financial Crisis** and **COVID-19**? '
    'All figures below are calculated live from the data, using the same methods as the project notebooks.'
)

with st.sidebar:
    st.header('Settings')
    show_flights = st.toggle('Show flight movements in index charts', value=True)
    map_reference = st.radio('Map: compare each month with',
                             ['Same month a year earlier', 'Same month before the crisis'])
    horizon = st.slider('Comparison: months after the event', 12, 72, 48, step=12)
    st.divider()
    st.caption(f"Schiphol data up to **{fmt_month(sources['csv']['Date'].max())}**.")
    st.caption('Group 11, TIL6022 Python Programming')

tabs = st.tabs(['Map', 'Comparison'] + list(EVENTS) + ['Data & methodology'])

# --- Map tab --------------------------------------------------------------
with tabs[0]:
    map_tab(map_reference)

# --- Comparison tab -------------------------------------------------------
with tabs[1]:
    kpis = {name: event_kpis(name) for name in EVENTS}

    cols = st.columns(3)
    for col, (name, k) in zip(cols, kpis.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}** · {EVENTS[name]['kind']}")
                st.metric('Lowest seasonal index', f"{k['trough_index']:.1f}",
                          f"{k['trough_index'] - 100:.1f} vs. baseline ({fmt_month(k['trough_date'])})")
                st.metric('Passengers, first 12 months vs. year before', f"{k['change12']:.1f}%")
                st.metric('Back at pre-crisis level after', fmt_recovery(k))

    st.markdown('### Event-aligned comparison')
    measure = st.radio(
        'Measure', ['Seasonal index (baseline = 100)', 'Deviation from forecast (%)', 'Year-on-year change (%)'],
        horizontal=True, label_visibility='collapsed',
    )
    column = {'Seasonal index (baseline = 100)': 'Index_Pax',
              'Deviation from forecast (%)': 'Deviation',
              'Year-on-year change (%)': 'YoY'}[measure]

    fig = go.Figure()
    for name, ev in EVENTS.items():
        data, _ = prepare_event(name)
        aligned = data[(data['Months_From_Event'] >= -24) & (data['Months_From_Event'] <= horizon)]
        fig.add_trace(go.Scatter(
            x=aligned['Months_From_Event'], y=aligned[column], name=name,
            line=dict(color=ev['color'], width=2),
            customdata=aligned['Date'].dt.strftime('%b %Y'),
            hovertemplate='%{y:.1f} (%{customdata})<extra>' + name + '</extra>',
        ))
    fig.add_vline(x=0, line_dash='dash', line_color=colors['grey'], line_width=1)
    fig.add_hline(y=100 if column == 'Index_Pax' else 0, line_dash='dot', line_color=colors['grey'], line_width=1)
    fig.update_xaxes(title_text='Months from event (0 = Sep 2001 / Sep 2008 / Mar 2020)', dtick=6)
    if column == 'YoY':
        fig.update_yaxes(range=[-110, 150])
    show(style(fig, 'Three disruptions on one time axis', measure.split(' (')[0],
               'Each event has its own five-year baseline and forecast model', height=450))

    st.markdown('### Summary table')
    summary = pd.DataFrame({
        name: {
            'Type of disruption': EVENTS[name]['kind'],
            'Baseline period': EVENTS[name]['base_label'],
            'Forecast model': 'Anchored' if EVENTS[name]['model'] == 'anchored' else 'Rolling',
            'Lowest seasonal index (month)': f"{k['trough_index']:.1f} ({fmt_month(k['trough_date'])})",
            'Months from event to lowest point': str(k['trough_months']),
            'Largest monthly YoY decline (first 12 months)': f"{k['worst_yoy']:.1f}% ({fmt_month(k['worst_yoy_date'])})",
            'Passenger change, first 12 months': f"{k['change12']:.1f}%",
            'Months with negative YoY growth': f"{k['neg_months']} of 12",
            'Largest deviation from forecast': f"{k['worst_dev']:.1f}% ({fmt_month(k['worst_dev_date'])})",
            'Back at pre-crisis level': fmt_recovery(k),
        }
        for name, k in kpis.items()
    })
    st.dataframe(summary, **FULL_WIDTH)
    st.caption(
        'Lowest point and largest deviation: within 36 months after the event. '
        'Back at pre-crisis level: first month after the lowest point where passengers are at least '
        'equal to the same month of the last pre-crisis year (2000 / 2007 / 2019), for three months in a row.'
    )

# --- One tab per event ----------------------------------------------------
for tab, (name, ev) in zip(tabs[2:5], EVENTS.items()):
    with tab:
        data, window = prepare_event(name)
        k = event_kpis(name)

        st.subheader(name)
        st.markdown(ev['description'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Lowest seasonal index', f"{k['trough_index']:.1f}", f"{fmt_month(k['trough_date'])}",
                  delta_color='off')
        c2.metric('Passengers, first 12 months', f"{k['change12']:.1f}%", 'vs. 12 months before',
                  delta_color='off')
        c3.metric('Largest deviation from forecast', f"{k['worst_dev']:.1f}%", fmt_month(k['worst_dev_date']),
                  delta_color='off')
        c4.metric('Back at pre-crisis level', fmt_recovery(k).split(' (')[0],
                  f"{k['latest_vs_base']:.0f}% of {ev['base_year']} in {fmt_month(k['latest_date'])}"
                  if k['recovery_months'] is None else fmt_month(k['recovery_date']),
                  delta_color='off')

        left, right = st.columns(2)
        with left:
            show(chart_raw(window, ev))
        with right:
            show(chart_index(window, ev, show_flights))

        left, right = st.columns(2)
        with left:
            show(chart_yoy(window, ev))
        with right:
            show(chart_deviation(window, ev))

        left, right = st.columns(2)
        with left:
            show(chart_annual(data, ev))
        with right:
            choice = st.selectbox('Segment', list(SEGMENTS), key=f'segment_{name}',
                                  label_visibility='collapsed')
            show(chart_segments(ev, choice))

        st.divider()
        EXTRAS[name](data, ev)

        with st.expander('Show data'):
            table = window[['Date', 'Passengers', 'Flights', 'Index_Pax', 'Index_Flights',
                            'YoY', 'Predicted', 'Deviation']]
            st.dataframe(table.round({c: 1 for c in table.columns[1:]}), hide_index=True, **FULL_WIDTH)
            st.download_button('Download as CSV', table.to_csv(index=False).encode('utf-8'),
                               file_name=f"schiphol_{name.replace('/', '').replace(' ', '_').lower()}.csv",
                               mime='text/csv', key=f'download_{name}')

# --- Data & methodology tab -----------------------------------------------
with tabs[5]:
    st.subheader('Methodology')
    st.markdown(
        """
**Seasonal index.** Every month is compared with the average of the *same calendar month* in the five
years before the disruption (100 = normal level). This removes the summer peak and winter dip.

**Year-on-year change.** The percentage change compared with the same month one year earlier.

**Forecast models.** What would traffic have been without the disruption?
- *Anchored model* (9/11, COVID-19): start from the last normal year and grow with the average
  year-on-year growth of that calendar month over the five years before.
- *Rolling model* (Financial Crisis): previous year times the average growth of the five years before.
  Because it keeps updating, it shows the *extra* decline compared to the recent trend.

**Deviation** = (actual - expected) / expected.
        """
    )

    st.subheader('Data sources')
    st.markdown(
        """
| Dataset | Period | Used for |
|---|---|---|
| Schiphol monthly traffic (Excel) | 1992 - now | 9/11 analysis and traffic segments |
| Schiphol monthly passengers & flights (CSV) | 1999 - now | Financial Crisis and COVID-19 |
| US BTS T-100 international segments | 1995 - 2006 | Schiphol-US routes around 9/11 |
| AEX index (Euronext) | 2004 - 2015 | Financial Crisis vs. stock market |
| CBS traffic & holidays abroad | 2017 - 2025 | COVID-19 validation |
        """
    )

    st.subheader('Data check: CSV vs. Excel')
    excel, csv = sources['excel'], sources['csv']
    check = pd.merge(csv, excel[['Date', 'Passengers', 'Flights']], on='Date', suffixes=('_csv', '_excel'))
    pax_diff = (check['Passengers_csv'] - check['Passengers_excel']) / check['Passengers_excel'] * 100
    flight_diff = (check['Flights_csv'] - check['Flights_excel']) / check['Flights_excel'] * 100
    c1, c2, c3 = st.columns(3)
    c1.metric('Overlapping months', len(check))
    c2.metric('Months with different passengers', int((pax_diff != 0).sum()),
              f'max {pax_diff.abs().max():.3f}%', delta_color='off')
    c3.metric('Months with different flights', int((flight_diff != 0).sum()),
              f'max {flight_diff.abs().max():.3f}%', delta_color='off')
    st.caption('The two Schiphol files differ by small revisions only, so the choice of file does not '
               'change the conclusions.')
