"""
Streamlit dashboard for Forecasting Airline Realized Volatility
with HAR-RV, IV, OVX, and TOSI.

Reads precomputed artifacts from data/processed/ (built by HAR_model.ipynb).
Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / 'data' / 'processed'
FIGURES = PROCESSED / 'figures'

st.set_page_config(
    page_title='Airline Volatility Forecasting',
    page_icon='✈️',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ---------------- Theme constants ----------------

PALETTE = {
    'ink':        '#1B2733',
    'ink_soft':   '#3A4A5C',
    'paper':      '#FBF9F4',
    'paper_alt':  '#F2EDE2',
    'card':       '#FFFFFF',
    'card_edge':  '#E5DFD0',
    'accent':     '#3C91E6',
    'accent_dk':  '#1F6FBF',
    'mute':       '#6B7785',
    'good':       '#0B6E4F',
    'warn':       '#D97706',
    'bad':        '#C75146',
}


# ---------------- CSS injection ----------------

def inject_css():
    st.markdown(
        f"""
        <style>
        /* ---------- Layout ---------- */
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}
        section.main > div {{ padding-top: 0; }}

        /* ---------- Headings ---------- */
        h1 {{
            color: {PALETTE['ink']};
            font-weight: 700;
            letter-spacing: -0.02em;
            border-bottom: 3px solid {PALETTE['accent']};
            padding-bottom: 0.45rem;
            margin-bottom: 0.4rem !important;
        }}
        h2 {{
            color: {PALETTE['ink']};
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-top: 2.2rem !important;
            border-left: 4px solid {PALETTE['accent']};
            padding-left: 0.75rem;
        }}
        h3 {{
            color: {PALETTE['ink']};
            font-weight: 600;
            margin-top: 1.6rem !important;
        }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: {PALETTE['mute']};
        }}

        /* ---------- Metric cards ---------- */
        [data-testid="stMetric"] {{
            background: {PALETTE['card']};
            border: 1px solid {PALETTE['card_edge']};
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 1px 3px rgba(27, 39, 51, 0.04);
        }}
        [data-testid="stMetricLabel"] {{
            color: {PALETTE['mute']} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        [data-testid="stMetricValue"] {{
            color: {PALETTE['ink']} !important;
            font-weight: 700;
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1B2733 0%, #2D3F50 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #E8EEF4 !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 0 0.25rem 0 !important;
            letter-spacing: -0.01em;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.12);
            margin: 1.1rem 0;
        }}
        /* Rectangular tab-style nav (radio group) */
        [data-testid="stSidebar"] [role="radiogroup"] {{
            gap: 0.3rem !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 3px solid transparent;
            border-radius: 8px;
            padding: 0.7rem 0.95rem !important;
            margin: 0 !important;
            min-height: 2.6rem;
            display: flex !important;
            align-items: center;
            width: 100% !important;
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
            cursor: pointer;
            position: relative;
        }}
        /* Hide the radio circle entirely — every visual variant Streamlit uses */
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child,
        [data-testid="stSidebar"] [role="radiogroup"] label [data-baseweb="radio"],
        [data-testid="stSidebar"] [role="radiogroup"] label svg,
        [data-testid="stSidebar"] [role="radiogroup"] label input[type="radio"] {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0 !important;
        }}
        /* Label text — use full width, slightly larger */
        [data-testid="stSidebar"] [role="radiogroup"] label p,
        [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {{
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            color: #E8EEF4 !important;
            margin: 0 !important;
            width: 100%;
        }}
        /* Hover state — whole rectangle subtly lifts */
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: rgba(60, 145, 230, 0.12);
            border-color: rgba(60, 145, 230, 0.35);
            transform: translateX(2px);
        }}
        /* Selected state — whole rectangle highlighted with strong accent */
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
            background: linear-gradient(90deg, rgba(60,145,230,0.32) 0%, rgba(60,145,230,0.10) 100%) !important;
            border-color: rgba(60, 145, 230, 0.5) !important;
            border-left: 3px solid {PALETTE['accent']} !important;
            box-shadow: 0 1px 6px rgba(60, 145, 230, 0.25);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:last-child {{
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}
        /* Fallback for browsers without :has() — older state attribute */
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {{
            background: linear-gradient(90deg, rgba(60,145,230,0.32) 0%, rgba(60,145,230,0.10) 100%) !important;
            border-color: rgba(60, 145, 230, 0.5) !important;
            border-left: 3px solid {PALETTE['accent']} !important;
        }}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {PALETTE['card_edge']};
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PALETTE['card']} !important;
            color: {PALETTE['accent_dk']} !important;
        }}

        /* ---------- Tables ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE['card_edge']};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---------- Misc ---------- */
        footer {{ visibility: hidden; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .stAlert {{ border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Card helpers ----------------

def hypothesis_card(title: str, verdict: str, tone: str, body: str):
    """Renders a colored verdict card via raw HTML."""
    tone_map = {
        'good': (PALETTE['good'], '#E8F4EF'),
        'warn': (PALETTE['warn'], '#FDF4E5'),
        'bad':  (PALETTE['bad'],  '#FAECEA'),
    }
    border, bg = tone_map.get(tone, (PALETTE['mute'], '#EFEFEF'))
    return f"""
    <div style="background:{bg}; border-left:5px solid {border};
                border-radius:10px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;
                height:100%;">
      <div style="font-size:0.72rem; letter-spacing:0.08em; color:{border};
                  font-weight:700; text-transform:uppercase;">{verdict}</div>
      <div style="font-size:1.05rem; font-weight:700; color:{PALETTE['ink']};
                  margin:0.15rem 0 0.4rem 0;">{title}</div>
      <div style="font-size:0.88rem; color:{PALETTE['ink_soft']}; line-height:1.45;">
        {body}
      </div>
    </div>
    """


def stat_card(label: str, value: str, sub: str = '', accent: str | None = None):
    accent = accent or PALETTE['accent']
    return f"""
    <div style="background:{PALETTE['card']}; border:1px solid {PALETTE['card_edge']};
                border-top:3px solid {accent}; border-radius:10px;
                padding:0.85rem 1rem; height:100%;">
      <div style="font-size:0.72rem; letter-spacing:0.06em; color:{PALETTE['mute']};
                  text-transform:uppercase;">{label}</div>
      <div style="font-size:1.55rem; font-weight:700; color:{PALETTE['ink']};
                  margin:0.15rem 0 0.1rem 0;">{value}</div>
      <div style="font-size:0.8rem; color:{PALETTE['mute']};">{sub}</div>
    </div>
    """


def labeled_card(label: str, body: str, accent: str) -> str:
    return f"""
    <div style="background:{PALETTE['card']}; border:1px solid {PALETTE['card_edge']};
                border-top:3px solid {accent}; border-radius:10px;
                padding:0.9rem 1.05rem; height:100%;">
      <div style="font-size:0.72rem; letter-spacing:0.08em; color:{accent};
                  font-weight:700; text-transform:uppercase;">{label}</div>
      <div style="margin-top:0.45rem; color:{PALETTE['ink_soft']};
                  line-height:1.5; font-size:0.9rem;">{body}</div>
    </div>
    """


def section_intro(text: str):
    st.markdown(
        f"""<div style="color:{PALETTE['ink_soft']}; font-size:0.98rem;
                       line-height:1.55; margin:0.4rem 0 1.1rem 0;">{text}</div>""",
        unsafe_allow_html=True,
    )


# ---------------- Plotly defaults ----------------

def style_fig(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template='plotly_white',
        paper_bgcolor=PALETTE['paper'],
        plot_bgcolor=PALETTE['card'],
        margin=dict(l=60, r=30, t=50 if title else 30, b=50),
        font=dict(family='Inter, system-ui, sans-serif', color=PALETTE['ink'], size=12),
        title=dict(text=title or '', font=dict(size=15, color=PALETTE['ink']), subtitle=dict(text='')),
        xaxis=dict(showgrid=True, gridcolor='#EFEAE0'),
        yaxis=dict(showgrid=True, gridcolor='#EFEAE0'),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor=PALETTE['card_edge'],
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor=PALETTE['ink'], font_color='white', font_size=11),
    )
    return fig


# ---------------- Artifact loading ----------------

@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    pq = PROCESSED / f'{name}.parquet'
    if pq.exists():
        return pd.read_parquet(pq)
    csv = PROCESSED / f'{name}.csv'
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f'Missing artifact: {name}')


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads((PROCESSED / 'meta.json').read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def load_figure(name: str) -> go.Figure:
    fig = pio.from_json((FIGURES / f'{name}.json').read_text(encoding='utf-8'))
    fig.update_layout(
        paper_bgcolor=PALETTE['paper'],
        plot_bgcolor=PALETTE['card'],
        font=dict(family='Inter, system-ui, sans-serif', color=PALETTE['ink']),
    )
    return fig


def artifacts_ready() -> bool:
    required = ['meta.json', 'summary.parquet', 'predictions.parquet']
    return all((PROCESSED / r).exists() for r in required) and FIGURES.exists()


# ---------------- Shared helpers ----------------

def render_metric_row(metrics: list[tuple[str, str, str | None]]):
    cols = st.columns(len(metrics))
    for col, (label, value, helptxt) in zip(cols, metrics):
        col.metric(label, value, help=helptxt)


def format_pct(x):
    return '—' if pd.isna(x) else f'{x * 100:.1f}%'


def format_num(x, digits=3):
    return '—' if pd.isna(x) else f'{x:,.{digits}f}'


def _short_month_year(date_str: str) -> str:
    ts = pd.to_datetime(date_str)
    return ts.strftime('%b %Y')


# ---------------- Pages ----------------

def page_background(meta):
    st.title('Background')
    st.caption('Why oil-market turbulence is an airline-investor problem — and the data we assembled to study it.')

    # ---- Opening: the question ----
    st.markdown(
        f"""
        <div style="background:{PALETTE['paper_alt']}; border-left:5px solid {PALETTE['accent']};
                    padding:1.1rem 1.3rem; border-radius:10px; margin:0.4rem 0 1.4rem 0;">
          <div style="font-size:0.72rem; letter-spacing:0.08em; color:{PALETTE['accent_dk']};
                      font-weight:700; text-transform:uppercase;">Our question</div>
          <div style="font-size:1.25rem; font-weight:600; color:{PALETTE['ink']};
                      line-height:1.45; margin-top:0.25rem;">
            How does volatility in crude oil futures — along with oil-market sentiment —
            impact the volatility of airline stocks?
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Why airlines are unique ----
    st.markdown('## Why airlines are the natural testbed')
    st.markdown(
        """
        Jet fuel is the single largest variable line on an airline's income statement —
        **17–21% of total operating expenses** for the major U.S. carriers. Unlike
        labor or maintenance, jet fuel prices track crude oil in near real-time, so
        oil-market uncertainty flows directly into airline profitability, investor
        expectations, and ultimately stock-price behavior.

        What makes the setup interesting is the framing. We are not asking whether
        oil *prices* move airline stocks — that link is well known. We are asking
        whether oil *volatility* and *sentiment* **lead** airline stock volatility:

        > *Can movements in oil markets today help predict airline stock volatility tomorrow?*
        """
    )

    fuel_labels = ['AAL<br>(American)', 'UAL<br>(United)', 'DAL<br>(Delta)', 'LUV<br>(Southwest)']
    fuel_values = [20.2, 21.0, 17.0, 19.0]
    bar_colors  = [meta['airline_colors'].get(t, PALETTE['accent'])
                   for t in ['AAL', 'UAL', 'DAL', 'LUV']]
    avg = sum(fuel_values) / len(fuel_values)

    fig_fuel = go.Figure(go.Bar(
        x=fuel_labels, y=fuel_values,
        marker_color=bar_colors,
        text=[f'{v}%' for v in fuel_values],
        textposition='outside',
        textfont=dict(color=PALETTE['ink'], size=14),
        width=0.5,
    ))
    fig_fuel.add_hline(
        y=avg, line_dash='dash', line_color=PALETTE['mute'],
        annotation_text=f'sector avg {avg:.1f}%',
        annotation_font_color=PALETTE['ink_soft'],
    )
    style_fig(fig_fuel, title='Fuel as % of Annual Operating Expenses', height=380)
    fig_fuel.update_layout(
        yaxis=dict(title='% of Operating Expenses', range=[0, 26]),
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

    # ---- Why it matters ----
    st.markdown('## Why a predictive answer would matter')
    c1, c2, c3 = st.columns(3)
    who = [
        ('Portfolio managers',
         'Adjust airline exposure ahead of oil-driven volatility shifts.',
         PALETTE['accent']),
        ('Options traders',
         'Build forecasts that generate alpha against ATM straddle pricing.',
         PALETTE['good']),
        ('Researchers',
         'Better understand cross-market volatility spillovers between commodities and equities.',
         PALETTE['warn']),
    ]
    for col, (lbl, body, accent) in zip([c1, c2, c3], who):
        col.markdown(labeled_card(lbl, body, accent), unsafe_allow_html=True)

    # ---- The data ----
    st.markdown('## The evidence we assembled')
    section_intro(
        'Three Bloomberg-sourced datasets let us measure oil-market uncertainty '
        'alongside airline stock behavior, day by day:'
        '<ul style="margin-top:0.5rem; line-height:1.7;">'
        '<li><b>CBOE Crude Oil Volatility Index (OVX)</b> — market expectations of oil price volatility.</li>'
        '<li><b>Text Oil Sentiment Indicator (TOSI)</b> — sentiment scores derived from oil-market news.</li>'
        '<li><b>Daily implied volatility</b> for American, Delta, United, Southwest, and the JETS ETF '
        '— derived from the options market as a forward-looking view of airline risk.</li>'
        '</ul>'
    )

    inventory = load_table('data_inventory')
    st.dataframe(inventory, use_container_width=True, hide_index=True)

    # ---- One-chart narrative: realized vol across the sector ----
    st.markdown('### The airline sector is a volatility-clustered asset class')
    section_intro(
        'Realized volatility across the four airlines and JETS moves in tight formation — '
        'huge shared spikes, long calm stretches in between. Overlaying OVX (right axis) '
        'shows that the biggest airline-vol regimes are oil-vol regimes too. This is the '
        'behavior our forecasts will have to beat.'
    )
    rv_fig = load_figure('realized_vol')

    # ---- Overlay OVX on a secondary y-axis (from main branch) ----
    ovx_dates, ovx_vals = None, None
    for table_name in ['ovx_data', 'ovx_series', 'ovx', 'market_data', 'raw_features']:
        try:
            df = load_table(table_name)
            date_col = next((c for c in df.columns if 'date' in c.lower()), None)
            val_col  = next((c for c in df.columns if 'ovx' in c.lower()), None)
            if date_col and val_col:
                ovx_dates = pd.to_datetime(df[date_col])
                ovx_vals  = df[val_col]
                break
        except FileNotFoundError:
            continue

    if ovx_dates is None:
        try:
            preds = load_table('predictions')
            val_col = next((c for c in preds.columns if 'ovx' in c.lower()), None)
            if val_col and 'trade_date' in preds.columns:
                tmp = (preds[['trade_date', val_col]]
                       .drop_duplicates('trade_date')
                       .sort_values('trade_date'))
                ovx_dates = pd.to_datetime(tmp['trade_date'])
                ovx_vals  = tmp[val_col]
        except (FileNotFoundError, StopIteration):
            pass

    if ovx_dates is not None:
        rv_fig.add_trace(go.Scatter(
            x=ovx_dates, y=ovx_vals,
            mode='lines', name='OVX',
            line=dict(color='#E07B39', width=2),
            yaxis='y2',
        ))
        rv_fig.update_layout(
            yaxis2=dict(
                title='OVX', overlaying='y', side='right',
                showgrid=False,
                title_font=dict(color='#E07B39'),
                tickfont=dict(color='#E07B39'),
            ),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
        )

    rv_fig.add_annotation(
        x='2025-02-15', y=0.060,
        text='Tariff shock drove recession fears;<br>airlines withdrew full-year guidance',
        showarrow=True, arrowhead=2, arrowwidth=1,
        arrowcolor=PALETTE['ink'], arrowsize=0.8,
        ax=-180, ay=30,
        font=dict(size=11, color=PALETTE['ink']),
        align='center',
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
    )
    rv_fig.add_annotation(
        x='2026-03-10', y=0.042,
        text='U.S. strikes on Iran sent oil surging;<br>Middle East airspace closures hit airlines',
        showarrow=True, arrowhead=2, arrowwidth=1,
        arrowcolor=PALETTE['ink'], arrowsize=0.8,
        ax=-80, ay=-90,
        font=dict(size=11, color=PALETTE['ink']),
        align='center',
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
    )
    rv_fig.update_xaxes(range=['2020-09-01', '2026-06-01'])
    st.plotly_chart(rv_fig, use_container_width=True)

    # ---- The co-movement chart that sets up the hypothesis ----
    st.markdown('### Oil uncertainty visibly tracks airline volatility')
    section_intro(
        'Overlaying OVX and TOSI on airline realized vol, the co-movement is obvious '
        "at crisis moments. The open question — which we test in the next tab — is "
        'whether it is *predictive* or merely *contemporaneous*.'
    )
    oil_fig = load_figure('oil_drivers')
    oil_fig.data = tuple(t for t in oil_fig.data if t.name != 'Sector RV')
    st.plotly_chart(oil_fig, use_container_width=True)


def page_hypothesis(meta):
    st.title('Hypothesis')
    st.caption('From an open question to a falsifiable prediction — and how we will put it to the test.')

    # ---- Scientific-method scaffolding ----
    st.markdown('## From question to testable prediction')
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ('1 · Observation',
         'Airlines spend 25–35% of opex on jet fuel. Oil-market uncertainty appears to '
         'move with airline volatility.',
         PALETTE['mute']),
        ('2 · Question',
         'Does oil-market volatility and sentiment *lead* airline stock volatility, '
         'or only co-move with it?',
         PALETTE['accent']),
        ('3 · Hypothesis',
         'Lagged OVX and TOSI carry predictive information about airline realized '
         'volatility beyond what prices and options already reveal.',
         PALETTE['warn']),
        ('4 · Test',
         'Forecast next-day volatility walk-forward out-of-sample and check whether '
         'adding each signal beats the baseline on Diebold-Mariano and straddle P&L.',
         PALETTE['good']),
    ]
    for col, (lbl, body, accent) in zip([c1, c2, c3, c4], steps):
        col.markdown(labeled_card(lbl, body, accent), unsafe_allow_html=True)

    # ---- The forecasting target ----
    st.markdown('## What exactly we forecast')
    st.markdown(
        """
        Our target is the **Variance Risk Premium (VRP)** — the gap between what the
        options market *expects* tomorrow's volatility to be and what *actually* happens:
        """
    )
    st.latex(r'\text{VRP}_t = \log\text{RV}_{t+1} - \log\text{IV}_t')
    st.markdown(
        """
        A positive predicted VRP says the market is under-pricing risk (buy the
        straddle); negative says it is over-pricing (sell). If our forecast beats the
        market's implicit one, we earn a persistent premium — and we gain a clean way
        to *measure* whether each new signal actually helps.

        The baseline model is **HAR (Heterogeneous AutoRegressive)**, which decomposes
        volatility memory into daily, weekly, and monthly components — short-, mid-,
        and long-horizon traders each contributing to today's price formation.
        """
    )

    # ---- Feature-level hypotheses ----
    st.markdown('## Four nested hypotheses')
    section_intro(
        'Each feature we add to HAR-RV encodes a distinct economic story. '
        'We keep the one whose story survives an out-of-sample test.'
    )
    st.markdown(
        """
        | Spec | Economic story we're testing |
        | --- | --- |
        | **HAR-RV** | Volatility clustering alone explains next-day vol — the null. |
        | **+ IV** | Options markets contain forward-looking information that HAR misses. |
        | **+ OVX** | Oil-market uncertainty is a jet-fuel cost signal that leaks into airline vol. |
        | **+ TOSI** | Oil-news sentiment front-runs oil-price moves — and therefore airline vol. |
        """
    )

    st.info(
        "**Central claim to be tested:** The Text Oil Sentiment Indicator (TOSI) — "
        "a sentiment signal from oil-market news — adds statistically significant "
        "predictive power for airline realized volatility *beyond* what HAR-RV, IV, "
        "and OVX already capture. If oil-news sentiment leads jet-fuel cost "
        "expectations, it should front-run airline vol moves that lagged RV and "
        "options pricing have not yet priced in."
    )

    # ---- Visual predictions ----
    st.markdown('## What each relationship should look like if true')
    section_intro(
        'If the hypothesis holds, we should see a visible slope when we line up each '
        "driver against future airline volatility. These scatters are the visual form "
        'of each prediction — the next tabs put them through formal tests.'
    )
    tab_iv, tab_ovx, tab_tosi = st.tabs(['IV → next-day RV', 'OVX → next-month RV', 'TOSI → next-month RV'])
    with tab_iv:
        st.plotly_chart(load_figure('iv_scatter'), use_container_width=True)
    with tab_ovx:
        st.plotly_chart(load_figure('ovx_scatter'), use_container_width=True)
    with tab_tosi:
        st.plotly_chart(load_figure('tosi_scatter'), use_container_width=True)

    # ---- Scope of the experiment ----
    st.markdown('## Scope of the experiment')
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Tickers**  \n{', '.join(meta['tickers'])}")
    c2.markdown(f"**Feature specs**  \n{len(meta['feature_specs'])} (HAR-RV, +IV, +OVX, +TOSI and combos)")
    c3.markdown(f"**Model families**  \n{', '.join(meta['model_families'])}")
    c4.markdown(f"**OOS window**  \n{meta['oos_start']} → {meta['oos_end']}")


def page_models(meta):
    st.title('Models')

    # ── HAR-X formula family ──────────────────────────────────────
    st.subheader('The HAR-X Model Family')
    section_intro(
        'All models forecast the <strong>Variance Risk Premium</strong> one trading day ahead. '
        'The target is:'
    )
    st.latex(r'\text{VRP}_{t+1} = \log\text{RV}_{t+1} - \log\text{IV}_t')

    section_intro(
        'The <strong>HAR (Heterogeneous AutoRegressive)</strong> baseline decomposes '
        'realized-vol memory into three horizons, reflecting how different trader types — '
        'day traders, weekly desks, and monthly macro funds — each contribute to current '
        'price formation:'
    )
    st.latex(
        r'\underbrace{\text{VRP}_{t+1}}_{\text{target}} = \alpha'
        r'+ \beta_d \underbrace{\log\text{RV}_t^{(d)}}_{\text{daily}}'
        r'+ \beta_w \underbrace{\log\text{RV}_t^{(w)}}_{\text{5-day avg}}'
        r'+ \beta_m \underbrace{\log\text{RV}_t^{(m)}}_{\text{22-day avg}}'
        r'+ \varepsilon_{t+1}'
    )

    section_intro(
        'Adding <strong>implied volatility</strong> captures the options market\'s '
        'forward-looking view — the single biggest improvement over the baseline:'
    )
    st.latex(
        r'\text{VRP}_{t+1} = \alpha'
        r'+ \beta_d \log\text{RV}_t^{(d)} + \beta_w \log\text{RV}_t^{(w)} + \beta_m \log\text{RV}_t^{(m)}'
        r'+ \gamma\,\log\text{IV}_t'
        r'+ \varepsilon_{t+1}'
    )

    section_intro(
        'The <strong>full HAR-X</strong> spec adds oil-market signals — OVX at three '
        'horizons and TOSI (level + monthly change) as a sentiment layer:'
    )
    st.latex(
        r'\text{VRP}_{t+1} = \alpha'
        r'+ \beta_d \log\text{RV}_t^{(d)} + \beta_w \log\text{RV}_t^{(w)} + \beta_m \log\text{RV}_t^{(m)}'
        r'+ \gamma\,\log\text{IV}_t'
        r'+ \delta_d \log\text{OVX}_t^{(d)} + \delta_w \log\text{OVX}_t^{(w)} + \delta_m \log\text{OVX}_t^{(m)}'
        r'+ \theta_1\,\text{TOSI}_t + \theta_2\,\Delta\text{TOSI}_t'
        r'+ \varepsilon_{t+1}'
    )

    st.markdown('---')

    # ── Walk-forward protocol (condensed) ────────────────────────
    st.subheader('Walk-forward evaluation protocol')
    render_metric_row([
        ('Min train days', '756', '3 trading years'),
        ('Test window', '63 days', '≈ one quarter'),
        ('Step', '63 days', 'Expanding window'),
        ('Threshold grid', '25–75 pctiles', 'of |predicted VRP| on in-sample fold'),
    ])

    st.markdown('---')

    # ── Viz 1 & 2: JETS RMSE and Sharpe by feature spec ──────────
    st.subheader('Feature spec performance — JETS')
    section_intro(
        'Each bar below represents OOS performance for JETS under that feature '
        'specification. Left: forecast accuracy (lower RMSE = better fit). '
        'Right: straddle Sharpe for OLS (higher = more tradeable edge). '
        'The dashed line marks the HAR-RV baseline.'
    )

    spec_order = [
        'HAR-RV', 'HAR-RV+IV', 'HAR-RV+OVX', 'HAR-RV+OVX+TOSI',
        'HAR-RV+IV+OVX', 'HAR-RV+IV+OVX+TOSI',
    ]
    spec_short = ['HAR-RV', '+IV', '+OVX', '+OVX\n+TOSI', '+IV\n+OVX', '+IV+OVX\n+TOSI']

    results = load_table('results')
    jets = results[results['Ticker'] == 'JETS']
    baseline_rmse  = jets.loc[(jets['Feature_Spec'] == 'HAR-RV') & (jets['Model_Family'] == 'OLS'), 'RMSE'].iloc[0]
    baseline_sharpe = jets.loc[(jets['Feature_Spec'] == 'HAR-RV') & (jets['Model_Family'] == 'OLS'), 'Sharpe_Straddle'].iloc[0]

    fam_colors = {f: meta['model_colors'].get(f, PALETTE['mute']) for f in meta['model_families']}

    col1, col2 = st.columns(2)

    # — RMSE chart (all 4 families, grouped) —
    with col1:
        fig_rmse = go.Figure()
        for fam in meta['model_families']:
            sub = jets[jets['Model_Family'] == fam].set_index('Feature_Spec').reindex(spec_order)
            fig_rmse.add_trace(go.Bar(
                name=fam, x=spec_short, y=sub['RMSE'].values,
                marker_color=fam_colors[fam],
            ))
        fig_rmse.add_hline(
            y=baseline_rmse, line_dash='dash', line_color=PALETTE['mute'],
            annotation_text='HAR-RV baseline', annotation_font_color=PALETTE['ink_soft'],
        )
        style_fig(fig_rmse, title='RMSE by Feature Spec (lower = better)', height=420)
        fig_rmse.update_layout(
            barmode='group',
            yaxis=dict(title='RMSE', range=[0.793, 0.836]),
            legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center'),
            margin=dict(l=50, r=20, t=50, b=80),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # — Sharpe chart (OLS only — cleanest linear story) —
    with col2:
        ols = jets[jets['Model_Family'] == 'OLS'].set_index('Feature_Spec').reindex(spec_order)
        bar_colors_sharpe = [
            PALETTE['bad'] if v < baseline_sharpe else PALETTE['good']
            for v in ols['Sharpe_Straddle'].values
        ]
        fig_sharpe = go.Figure(go.Bar(
            x=spec_short, y=ols['Sharpe_Straddle'].values,
            marker_color=bar_colors_sharpe,
            text=[f'{v:.2f}' for v in ols['Sharpe_Straddle'].values],
            textposition='outside', textfont=dict(color=PALETTE['ink']),
        ))
        fig_sharpe.add_hline(
            y=baseline_sharpe, line_dash='dash', line_color=PALETTE['mute'],
            annotation_text='HAR-RV baseline', annotation_font_color=PALETTE['ink_soft'],
        )
        style_fig(fig_sharpe, title='Straddle Sharpe — OLS (higher = better)', height=420)
        fig_sharpe.update_layout(
            yaxis=dict(title='OOS Sharpe', range=[0, 3.2]),
            margin=dict(l=50, r=20, t=50, b=80),
            showlegend=False,
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    st.markdown('---')

    # ── Feature importance ────────────────────────────────────────
    st.subheader('Tree-based feature importance — JETS')
    st.plotly_chart(load_figure('feature_importance_jets'), use_container_width=True)

    fi = load_table('feature_importances')
    st.caption('All tickers — top drivers under HAR-RV+IV+OVX+TOSI')
    ticker = st.selectbox('Ticker', options=meta['tickers'], index=meta['tickers'].index('JETS'))
    family = st.selectbox('Family', options=['XGBoost', 'Random Forest'])
    view = fi[(fi['Ticker'] == ticker) & (fi['Model_Family'] == family)].sort_values('Importance', ascending=True)
    if view.empty:
        st.info('No feature importances available for this selection.')
    else:
        fig = go.Figure(go.Bar(
            x=view['Importance'], y=view['Feature'],
            orientation='h', marker_color=meta['model_colors'].get(family, PALETTE['accent']),
        ))
        style_fig(fig, title=f'{ticker} — {family} feature importance', height=520)
        fig.update_layout(margin=dict(l=160, r=30, t=50, b=30),
                          xaxis_title='Importance', yaxis_title='Feature')
        st.plotly_chart(fig, use_container_width=True)


def page_results(meta):
    st.title('Results')
    summary = load_table('summary')
    results = load_table('results')
    best = load_table('best_models')
    dm = load_table('dm_results')

    st.subheader('Headline — best model per ticker')
    cols = ['Ticker', 'Model', 'RMSE', 'Sharpe_Straddle', 'Directional_Acc']
    best_view = best[cols].copy()
    best_view['RMSE'] = best_view['RMSE'].map(lambda x: format_num(x, 4))
    best_view['Sharpe_Straddle'] = best_view['Sharpe_Straddle'].map(lambda x: format_num(x, 2))
    best_view['Directional_Acc'] = best_view['Directional_Acc'].map(format_pct)
    st.dataframe(best_view, use_container_width=True, hide_index=True)

    st.subheader('Average metrics by feature spec × model family')
    st.dataframe(
        summary.style.format({
            'Avg_RMSE': '{:.4f}', 'Avg_MAE': '{:.4f}', 'Avg_R2': '{:.3f}',
            'Avg_Directional_Acc': '{:.1%}', 'Avg_Pct_Days_Traded': '{:.1%}',
            'Avg_Sharpe_Straddle': '{:.2f}',
            'Avg_RMSE_vs_OLS_HAR_RV': '{:.3f}',
            'Avg_Sharpe_vs_OLS_HAR_RV': '{:+.2f}',
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader('RMSE heatmap — feature spec × model family, averaged across tickers')
    rmse_pivot = results.pivot_table(
        index='Feature_Spec', columns='Model_Family', values='RMSE', aggfunc='mean'
    ).reindex(index=list(meta['feature_specs'].keys()), columns=meta['model_families'])
    z_vals = rmse_pivot.values
    text_vals = [['' if np.isnan(v) else f'{v:.4f}' for v in row] for row in z_vals]
    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(rmse_pivot.columns), y=list(rmse_pivot.index),
        colorscale='RdYlGn_r',
        text=text_vals, texttemplate='%{text}',
        hovertemplate='Spec=%{y}<br>Family=%{x}<br>Avg RMSE=%{z}<extra></extra>',
        colorbar=dict(title='Avg RMSE'),
    ))
    style_fig(fig, height=460)
    fig.update_layout(margin=dict(l=180, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('OLS Sharpe by feature spec — per ticker')
    st.plotly_chart(load_figure('ols_sharpe'), use_container_width=True)

    st.subheader('IV lift — HAR-RV vs +IV vs +IV+OVX+TOSI (OLS)')
    st.plotly_chart(load_figure('iv_lift'), use_container_width=True)

    st.subheader('Diebold-Mariano significance')
    st.plotly_chart(load_figure('dm_heatmap'), use_container_width=True)
    with st.expander('Raw DM test table'):
        st.dataframe(dm, use_container_width=True, hide_index=True)

    st.subheader('TOSI hypothesis — verdict')
    st.markdown(
        """
        **Does TOSI add predictive power?**  The Diebold-Mariano results above let us
        answer directly:

        * **No ticker reaches the 5% significance threshold** on either TOSI-increment
          test ("TOSI adds to HAR-RV+OVX" or "TOSI adds to HAR-RV+IV+OVX").  The
          smallest p-value is ~0.10 (DAL), and for LUV the DM statistic is *negative*
          — TOSI hurt that ticker.
        * **But the full HAR-RV+IV+OVX+TOSI spec achieves the lowest average RMSE**
          of any combination (OLS, 0.7637 vs. HAR-RV baseline 0.7684) and is selected
          as the per-ticker best for 3 of 4 names (DAL, UAL, JETS).
        * **Conclusion:** TOSI's incremental contribution is too small for the DM test
          to detect over the ~450-day OOS window, but its directional nudge is real
          enough to improve ensemble point forecasts.  Treat it as a **weak auxiliary**
          — useful in combination, not sufficient on its own.
        """
    )

    st.subheader('Forecast vs actual — JETS')
    st.plotly_chart(load_figure('jets_forecast'), use_container_width=True)

    st.subheader('Explore any ticker × spec × family')
    preds = load_table('predictions')
    preds['trade_date'] = pd.to_datetime(preds['trade_date'])
    c1, c2, c3 = st.columns(3)
    sel_t = c1.selectbox('Ticker', meta['tickers'], key='res_ticker')
    sel_s = c2.selectbox('Feature spec', list(meta['feature_specs'].keys()), key='res_spec')
    sel_f = c3.selectbox('Model family', meta['model_families'], key='res_fam')
    view = preds[
        (preds['Ticker'] == sel_t) & (preds['Feature_Spec'] == sel_s) & (preds['Model_Family'] == sel_f)
    ].sort_values('trade_date')
    if view.empty:
        st.info('No predictions for this combination.')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view['trade_date'], y=view['vrp_actual'], mode='lines',
            name='Actual VRP', line=dict(color=PALETTE['ink'], width=1.6),
        ))
        fig.add_trace(go.Scatter(
            x=view['trade_date'], y=view['y_pred_vrp'], mode='lines',
            name='Predicted VRP',
            line=dict(color=meta['airline_colors'].get(sel_t, PALETTE['accent']), width=2.0),
        ))
        style_fig(fig, title=f'{sel_t} — {sel_f} | {sel_s}', height=460)
        fig.update_layout(xaxis_title='Date', yaxis_title='VRP (log-variance)',
                          hovermode='x unified',
                          legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        r = results[
            (results['Ticker'] == sel_t) & (results['Feature_Spec'] == sel_s) & (results['Model_Family'] == sel_f)
        ]
        if not r.empty:
            row = r.iloc[0]
            render_metric_row([
                ('RMSE', format_num(row['RMSE'], 4), None),
                ('Sharpe', format_num(row['Sharpe_Straddle'], 2), 'Straddle Sharpe (OOS)'),
                ('Directional acc', format_pct(row['Directional_Acc']), None),
                ('Days traded', format_pct(row['Pct_Days_Traded']), None),
            ])


# ---------------- Strategy page ----------------

def _build_pnl(view: pd.DataFrame, sqrt2pi: float) -> pd.DataFrame:
    out = view.sort_values('trade_date').copy()
    out['pnl_daily'] = out['signal'] * (out['abs_daily_return_tplus1'] - sqrt2pi * out['iv_daily_vol'])
    out['cum_pnl'] = out['pnl_daily'].cumsum()
    return out


def page_strategy(meta):
    st.title('Strategy')
    st.caption('From a one-number forecast to an out-of-sample equity curve.')

    sqrt2pi = float(meta['sqrt2_pi'])
    preds = load_table('predictions').copy()
    preds['trade_date'] = pd.to_datetime(preds['trade_date'])
    results = load_table('results')

    # ------------- Three-step explanation -------------
    st.markdown('## How the strategy works')
    c1, c2, c3 = st.columns(3)
    steps = [
        ('1 · Forecast',
         'Predict tomorrow\'s VRP — the gap between realized and implied volatility — '
         'from HAR-RV, IV, OVX and TOSI features.'),
        ('2 · Filter',
         'Trade only when |predicted VRP| exceeds a per-fold threshold chosen by '
         'nested walk-forward Sharpe.  No data peeking.'),
        ('3 · Trade',
         'Long the ATM straddle if predicted VRP > 0, short otherwise.  '
         'P&L = signal × (|close-to-close return| − √(2/π)·σ IV).'),
    ]
    for col, (title, body) in zip([c1, c2, c3], steps):
        col.markdown(
            f"""<div style="background:{PALETTE['card']}; border:1px solid {PALETTE['card_edge']};
                          border-top:3px solid {PALETTE['accent']}; border-radius:10px;
                          padding:1rem 1.1rem; height:100%;">
                  <div style="font-size:0.78rem; color:{PALETTE['accent_dk']};
                              font-weight:700; letter-spacing:0.05em;">{title}</div>
                  <div style="margin-top:0.4rem; color:{PALETTE['ink_soft']};
                              line-height:1.5; font-size:0.9rem;">{body}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ------------- Headline strategy stats (JETS, full spec, OLS) -------------
    st.markdown('## Headline — JETS, full spec, OLS')
    headline = results[
        (results['Ticker'] == 'JETS')
        & (results['Feature_Spec'] == 'HAR-RV+IV+OVX+TOSI')
        & (results['Model_Family'] == 'OLS')
    ]
    if not headline.empty:
        h = headline.iloc[0]
        cols = st.columns(4)
        cards = [
            ('Sharpe (OOS)', format_num(h['Sharpe_Straddle'], 2),
             'Annualised straddle Sharpe', PALETTE['good']),
            ('Mean daily P&L', format_num(h['Mean_Straddle_PnL'], 5),
             'Log-return units, per active day', PALETTE['accent']),
            ('Hit rate (traded)', format_pct(h['Signal_Hit_Rate']),
             '% of taken trades with P&L > 0', PALETTE['warn']),
            ('Days traded', format_pct(h['Pct_Days_Traded']),
             'Threshold-filtered fraction of OOS days', PALETTE['ink_soft']),
        ]
        for col, (lbl, val, sub, accent) in zip(cols, cards):
            col.markdown(stat_card(lbl, val, sub, accent), unsafe_allow_html=True)

    st.markdown('### JETS cumulative straddle P&L')
    st.plotly_chart(load_figure('jets_pnl'), use_container_width=True)

    # ------------- HAR baseline vs full spec equity curves -------------
    st.markdown('## Does the macro layer pay off?')
    section_intro(
        'The full feature set adds IV, OVX and TOSI on top of pure HAR-RV.  '
        'Below we trade the same rule with each spec and compare cumulative P&L — '
        'this is how the &lt;1% RMSE difference compounds into Sharpe.'
    )

    compare_ticker = st.selectbox(
        'Compare on ticker', meta['tickers'],
        index=meta['tickers'].index('JETS'), key='compare_ticker',
    )

    base = preds[(preds['Ticker'] == compare_ticker)
                 & (preds['Feature_Spec'] == 'HAR-RV')
                 & (preds['Model_Family'] == 'OLS')]
    full = preds[(preds['Ticker'] == compare_ticker)
                 & (preds['Feature_Spec'] == 'HAR-RV+IV+OVX+TOSI')
                 & (preds['Model_Family'] == 'OLS')]

    if base.empty or full.empty or 'iv_daily_vol' not in base.columns:
        st.info('Comparison data unavailable for this ticker.')
    else:
        base_p = _build_pnl(base, sqrt2pi)
        full_p = _build_pnl(full, sqrt2pi)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=base_p['trade_date'], y=base_p['cum_pnl'],
            mode='lines', name='HAR-RV (baseline)',
            line=dict(color=PALETTE['mute'], width=2, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=full_p['trade_date'], y=full_p['cum_pnl'],
            mode='lines', name='HAR-RV + IV + OVX + TOSI',
            line=dict(color=meta['airline_colors'].get(compare_ticker, PALETTE['accent']), width=2.6),
        ))
        fig.add_hline(y=0, line_dash='dot', line_color=PALETTE['mute'])
        style_fig(fig, title=f'{compare_ticker} — cumulative straddle P&L (OLS)', height=420)
        fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative P&L',
                          hovermode='x unified',
                          legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

    # ------------- Sharpe-by-ticker bar (best model per ticker) -------------
    st.markdown('## Where the strategy works — Sharpe by ticker (best model)')
    best = load_table('best_models').copy()
    best = best.sort_values('Sharpe_Straddle', ascending=True)
    sharpe_fig = go.Figure(go.Bar(
        x=best['Sharpe_Straddle'],
        y=best['Ticker'],
        orientation='h',
        marker=dict(
            color=[meta['airline_colors'].get(t, PALETTE['accent']) for t in best['Ticker']],
            line=dict(color=PALETTE['ink'], width=0.5),
        ),
        text=[f"{s:.2f}" for s in best['Sharpe_Straddle']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Best model: %{customdata}<br>Sharpe: %{x:.2f}<extra></extra>',
        customdata=best['Model'],
    ))
    sharpe_fig.add_vline(x=1.0, line_dash='dot', line_color=PALETTE['mute'],
                         annotation_text='Sharpe = 1', annotation_position='top right')
    style_fig(sharpe_fig, height=320)
    sharpe_fig.update_layout(xaxis_title='Out-of-sample Sharpe', yaxis_title='',
                             margin=dict(l=80, r=80, t=20, b=40))
    st.plotly_chart(sharpe_fig, use_container_width=True)

    # ------------- Sandbox -------------
    st.markdown('## Build your own — sandbox')
    section_intro(
        'Pick any ticker × feature spec × model family.  '
        'P&L is recomputed from the stored OOS predictions and the ATM straddle premium.'
    )
    c1, c2, c3 = st.columns(3)
    sel_t = c1.selectbox('Ticker', meta['tickers'], key='strat_t')
    sel_s = c2.selectbox(
        'Feature spec', list(meta['feature_specs'].keys()),
        index=list(meta['feature_specs'].keys()).index('HAR-RV+IV+OVX+TOSI'), key='strat_s',
    )
    sel_f = c3.selectbox('Model family', meta['model_families'], key='strat_f')

    view = preds[
        (preds['Ticker'] == sel_t) & (preds['Feature_Spec'] == sel_s) & (preds['Model_Family'] == sel_f)
    ]
    if view.empty or 'iv_daily_vol' not in view.columns:
        st.info('No predictions for this combination.')
        return
    sandbox = _build_pnl(view, sqrt2pi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sandbox['trade_date'], y=sandbox['cum_pnl'],
        mode='lines', name='Cumulative P&L',
        line=dict(color=meta['airline_colors'].get(sel_t, PALETTE['accent']), width=2.4),
    ))
    fig.add_hline(y=0, line_dash='dot', line_color=PALETTE['mute'])
    style_fig(fig, title=f'{sel_t} — {sel_f} | {sel_s}', height=400)
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative P&L',
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    r = results[
        (results['Ticker'] == sel_t) & (results['Feature_Spec'] == sel_s) & (results['Model_Family'] == sel_f)
    ]
    if not r.empty:
        row = r.iloc[0]
        render_metric_row([
            ('Sharpe', format_num(row['Sharpe_Straddle'], 2), None),
            ('Mean daily P&L', format_num(row['Mean_Straddle_PnL'], 5), None),
            ('Hit rate (traded)', format_pct(row['Signal_Hit_Rate']), '% of active trades with P&L > 0'),
            ('Days traded', format_pct(row['Pct_Days_Traded']), None),
        ])

    with st.expander('Per-fold threshold behavior'):
        thresh = load_table('threshold_summary')
        st.dataframe(
            thresh[(thresh['Ticker'] == sel_t) & (thresh['Feature_Spec'] == sel_s) & (thresh['Model_Family'] == sel_f)],
            use_container_width=True, hide_index=True,
        )


# ---------------- Conclusion page ----------------

def page_conclusion(meta):
    st.title('Conclusion')
    st.caption('Did the hypotheses hold up? — and what did the model comparison teach us?')

    dm = load_table('dm_results')
    best = load_table('best_models')
    summary = load_table('summary')
    results = load_table('results')

    best_ols_full = summary[
        (summary['Feature_Spec'] == 'HAR-RV+IV+OVX+TOSI') & (summary['Model_Family'] == 'OLS')
    ].iloc[0]
    base_ols = summary[
        (summary['Feature_Spec'] == 'HAR-RV') & (summary['Model_Family'] == 'OLS')
    ].iloc[0]

    # ------------- Headline metric cards -------------
    rmse_lift = (1 - best_ols_full['Avg_RMSE'] / base_ols['Avg_RMSE']) * 100
    best_idx = best['Sharpe_Straddle'].idxmax()
    st.markdown('## Headline numbers')
    cols = st.columns(4)
    cards = [
        ('Best avg RMSE', format_num(best_ols_full['Avg_RMSE'], 4),
         'Full spec, OLS — lowest of 24 combos', PALETTE['good']),
        ('Baseline HAR-RV RMSE', format_num(base_ols['Avg_RMSE'], 4),
         f'Lift from full spec: {rmse_lift:+.2f}%', PALETTE['ink_soft']),
        ('Best ticker Sharpe', format_num(best['Sharpe_Straddle'].max(), 2),
         f"{best.loc[best_idx, 'Ticker']} — {best.loc[best_idx, 'Model']}", PALETTE['accent']),
        ('Avg directional acc', format_pct(summary['Avg_Directional_Acc'].mean()),
         'Sign-of-VRP correct', PALETTE['warn']),
    ]
    for col, (lbl, val, sub, accent) in zip(cols, cards):
        col.markdown(stat_card(lbl, val, sub, accent), unsafe_allow_html=True)

    # ------------- Hypothesis scorecard as visual cards -------------
    st.markdown('## Hypothesis scorecard')
    section_intro(
        'Each card shows whether the Diebold-Mariano test rejects equality '
        'between the spec and its lower-feature counterpart at the 5% level, '
        'and how to interpret the verdict.'
    )

    def _verdict(hypothesis_label):
        rows = dm[dm['Hypothesis'] == hypothesis_label]
        sig = rows[rows['Significant_5pct']]['Ticker'].tolist()
        return sig, rows['P_Value'].min() if not rows.empty else float('nan')

    iv_sig, iv_p = _verdict('IV adds to HAR-RV?')
    ovx_sig, ovx_p = _verdict('OVX adds to HAR-RV?')
    tosi_sig, tosi_p = _verdict('TOSI adds to HAR-RV+IV+OVX?')

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        hypothesis_card(
            'IV → HAR-RV',
            'Partially supported',
            'warn',
            f"Significant for <b>{', '.join(iv_sig) if iv_sig else 'no ticker'}</b> "
            f"(best p = {iv_p:.4f}). Forward-looking option content shows up at the "
            f"sector-aggregate level (JETS), but not for individual airlines."
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        hypothesis_card(
            'OVX → HAR-RV',
            'Not supported',
            'bad',
            f"No ticker reaches significance (best p = {ovx_p:.4f}). "
            f"Oil vol co-moves with airline RV contemporaneously but does not "
            f"<i>lead</i> next-day RV once HAR knows recent history."
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        hypothesis_card(
            'TOSI → HAR-RV+IV+OVX',
            'Weakly supported',
            'warn',
            f"No DM-significant ticker (best p = {tosi_p:.4f}), but the full spec "
            f"<b>HAR-RV+IV+OVX+TOSI</b> achieves the lowest avg RMSE and is the per-ticker "
            f"best for 3 of 4 names. A useful weak auxiliary."
        ),
        unsafe_allow_html=True,
    )

    # ------------- Model family comparison (supports new script beat) -------------
    st.markdown('## On using four model families')
    section_intro(
        'We ran <b>OLS, Ridge, Random Forest and XGBoost</b> across every spec.  '
        'The flat-ish bar chart below is itself a finding: linear OLS matches or beats '
        'the more flexible families on average RMSE — feature engineering beat model complexity.'
    )

    fam_avg = (
        results.groupby('Model_Family', as_index=False)['RMSE']
        .mean()
        .rename(columns={'RMSE': 'Avg_RMSE'})
    )
    fam_avg = fam_avg.set_index('Model_Family').reindex(meta['model_families']).reset_index()

    cA, cB = st.columns([1, 1])
    with cA:
        fam_fig = go.Figure(go.Bar(
            x=fam_avg['Model_Family'],
            y=fam_avg['Avg_RMSE'],
            marker=dict(
                color=[meta['model_colors'].get(f, PALETTE['accent']) for f in fam_avg['Model_Family']],
                line=dict(color=PALETTE['ink'], width=0.5),
            ),
            text=[f"{v:.4f}" for v in fam_avg['Avg_RMSE']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg RMSE: %{y:.4f}<extra></extra>',
        ))
        ymin = float(fam_avg['Avg_RMSE'].min()) * 0.995
        ymax = float(fam_avg['Avg_RMSE'].max()) * 1.005
        style_fig(fam_fig, title='Avg RMSE by model family (lower = better)', height=340)
        fam_fig.update_layout(yaxis=dict(range=[ymin, ymax], gridcolor='#EFEAE0'),
                              xaxis_title='', yaxis_title='Avg RMSE')
        st.plotly_chart(fam_fig, use_container_width=True)

    with cB:
        spec_avg = (
            results[results['Model_Family'] == 'OLS']
            .groupby('Feature_Spec', as_index=False)['RMSE'].mean()
            .rename(columns={'RMSE': 'Avg_RMSE'})
        )
        spec_avg = spec_avg.set_index('Feature_Spec').reindex(list(meta['feature_specs'].keys())).reset_index()
        spec_fig = go.Figure(go.Bar(
            x=spec_avg['Feature_Spec'].map(meta['spec_short']),
            y=spec_avg['Avg_RMSE'],
            marker=dict(
                color=[meta['spec_colors'].get(s, PALETTE['accent']) for s in spec_avg['Feature_Spec']],
                line=dict(color=PALETTE['ink'], width=0.5),
            ),
            text=[f"{v:.4f}" for v in spec_avg['Avg_RMSE']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg RMSE: %{y:.4f}<extra></extra>',
        ))
        ymin = float(spec_avg['Avg_RMSE'].min()) * 0.995
        ymax = float(spec_avg['Avg_RMSE'].max()) * 1.005
        style_fig(spec_fig, title='Avg RMSE by feature spec (OLS)', height=340)
        spec_fig.update_layout(yaxis=dict(range=[ymin, ymax], gridcolor='#EFEAE0'),
                               xaxis_title='', yaxis_title='Avg RMSE')
        st.plotly_chart(spec_fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="background:{PALETTE['paper_alt']}; border-left:4px solid {PALETTE['accent']};
                    padding:0.85rem 1.05rem; border-radius:8px; margin-top:0.4rem;">
          <b>Reading the chart:</b> the y-axis is zoomed — the spread between best and
          worst model family is &lt;1% of RMSE. Tree-based models matter for exactly one
          ticker (LUV under bare HAR-RV); everywhere else, the linear baseline holds up.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------- Best model per ticker as cards -------------
    st.markdown('## Best model per ticker')
    cols = st.columns(len(best))
    for col, (_, row) in zip(cols, best.iterrows()):
        accent = meta['airline_colors'].get(row['Ticker'], PALETTE['accent'])
        col.markdown(
            f"""<div style="background:{PALETTE['card']}; border:1px solid {PALETTE['card_edge']};
                          border-top:4px solid {accent}; border-radius:10px;
                          padding:0.85rem 1rem; height:100%;">
                  <div style="font-size:1.4rem; font-weight:800; color:{accent};">{row['Ticker']}</div>
                  <div style="font-size:0.78rem; color:{PALETTE['mute']}; text-transform:uppercase;
                              letter-spacing:0.05em; margin-top:0.15rem;">{row['Model']}</div>
                  <div style="margin-top:0.55rem; display:flex; gap:0.9rem; flex-wrap:wrap;">
                    <div><div style="font-size:0.7rem; color:{PALETTE['mute']};">RMSE</div>
                         <div style="font-weight:700; color:{PALETTE['ink']};">{row['RMSE']:.4f}</div></div>
                    <div><div style="font-size:0.7rem; color:{PALETTE['mute']};">Sharpe</div>
                         <div style="font-weight:700; color:{PALETTE['ink']};">{row['Sharpe_Straddle']:.2f}</div></div>
                    <div><div style="font-size:0.7rem; color:{PALETTE['mute']};">Dir. acc</div>
                         <div style="font-weight:700; color:{PALETTE['ink']};">{row['Directional_Acc']*100:.1f}%</div></div>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ------------- Trading implications -------------
    st.markdown('## Trading implications')
    st.markdown(
        f"""
        <ul style="line-height:1.7; color:{PALETTE['ink_soft']};">
          <li><b>JETS straddle edge is real.</b> Full-spec OLS yields OOS Sharpe ≈ 2.7
              on threshold-filtered ATM straddles — economically meaningful.</li>
          <li><b>LUV is the outlier.</b> Plain HAR-RV + Random Forest wins on LUV; extra
              macro features degrade performance, consistent with LUV's domestic-only,
              narrow-body route mix.</li>
          <li><b>Single-name tests are hard.</b> Statistical power is limited; ensemble
              spec + threshold selection is how the marginal signal becomes tradeable.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )


# ---------------- App shell ----------------

def render_sidebar(meta):
    sb = st.sidebar
    sb.markdown(
        f"""<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.1rem;">
              <div style="font-size:1.6rem;">✈️</div>
              <div>
                <div style="font-size:1.15rem; font-weight:800; color:white; line-height:1.1;">
                  Airline Volatility</div>
                <div style="font-size:0.72rem; color:#A8B5C2; letter-spacing:0.06em;
                            text-transform:uppercase;">CS 329e · Group 23</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    sb.markdown('<hr/>', unsafe_allow_html=True)

    sb.markdown(
        '<div style="font-size:0.7rem; letter-spacing:0.08em; color:#A8B5C2; '
        'text-transform:uppercase; margin-bottom:0.3rem;">Sections</div>',
        unsafe_allow_html=True,
    )
    page = sb.radio(
        'Section',
        ['Background', 'Hypothesis', 'Models', 'Results', 'Strategy', 'Conclusion'],
        label_visibility='collapsed',
    )

    sb.markdown('<hr/>', unsafe_allow_html=True)

    # Highlights panel
    try:
        best = load_table('best_models')
        summary = load_table('summary')
        top = best.loc[best['Sharpe_Straddle'].idxmax()]
        full_ols = summary[(summary['Feature_Spec'] == 'HAR-RV+IV+OVX+TOSI')
                           & (summary['Model_Family'] == 'OLS')].iloc[0]
        sb.markdown(
            f"""<div style="font-size:0.7rem; letter-spacing:0.08em; color:#A8B5C2;
                          text-transform:uppercase; margin-bottom:0.4rem;">Highlights</div>
                <div style="background:rgba(255,255,255,0.05); padding:0.7rem 0.8rem;
                            border-radius:8px; border:1px solid rgba(255,255,255,0.08);
                            margin-bottom:0.6rem;">
                  <div style="font-size:0.7rem; color:#A8B5C2;">Top Sharpe</div>
                  <div style="font-size:1.25rem; font-weight:800; color:white;">
                    {top['Sharpe_Straddle']:.2f}</div>
                  <div style="font-size:0.78rem; color:#D8E0E8;">{top['Ticker']} · {top['Model']}</div>
                </div>
                <div style="background:rgba(255,255,255,0.05); padding:0.7rem 0.8rem;
                            border-radius:8px; border:1px solid rgba(255,255,255,0.08);">
                  <div style="font-size:0.7rem; color:#A8B5C2;">Best avg RMSE</div>
                  <div style="font-size:1.25rem; font-weight:800; color:white;">
                    {full_ols['Avg_RMSE']:.4f}</div>
                  <div style="font-size:0.78rem; color:#D8E0E8;">Full spec · OLS</div>
                </div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    sb.markdown('<hr/>', unsafe_allow_html=True)
    sb.markdown(
        f"""<div style="font-size:0.78rem; color:#D8E0E8; line-height:1.5;">
              <div style="font-size:0.7rem; letter-spacing:0.08em; color:#A8B5C2;
                          text-transform:uppercase; margin-bottom:0.3rem;">OOS Window</div>
              {meta['oos_start']} → {meta['oos_end']}
              <div style="font-size:0.7rem; letter-spacing:0.08em; color:#A8B5C2;
                          text-transform:uppercase; margin:0.7rem 0 0.3rem 0;">Tickers</div>
              {' · '.join(meta['tickers'])}
            </div>""",
        unsafe_allow_html=True,
    )

    return page


def main():
    inject_css()

    if not artifacts_ready():
        st.error(
            'Precomputed artifacts not found.  Run HAR_model.ipynb end-to-end first '
            'to populate `data/processed/`.'
        )
        st.stop()

    meta = load_meta()
    page = render_sidebar(meta)

    {
        'Background': page_background,
        'Hypothesis': page_hypothesis,
        'Models':     page_models,
        'Results':    page_results,
        'Strategy':   page_strategy,
        'Conclusion': page_conclusion,
    }[page](meta)


if __name__ == '__main__':
    main()
