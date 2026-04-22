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
    return pio.from_json((FIGURES / f'{name}.json').read_text(encoding='utf-8'))


def artifacts_ready() -> bool:
    required = ['meta.json', 'summary.parquet', 'predictions.parquet']
    return all((PROCESSED / r).exists() for r in required) and FIGURES.exists()


# ---------------- Shared styling ----------------

def render_metric_row(metrics: list[tuple[str, str, str | None]]):
    cols = st.columns(len(metrics))
    for col, (label, value, helptxt) in zip(cols, metrics):
        col.metric(label, value, help=helptxt)


def format_pct(x):
    return '—' if pd.isna(x) else f'{x * 100:.1f}%'


def format_num(x, digits=3):
    return '—' if pd.isna(x) else f'{x:,.{digits}f}'


# ---------------- Pages ----------------

def _short_month_year(date_str: str) -> str:
    ts = pd.to_datetime(date_str)
    return ts.strftime('%b %Y')


def page_hypothesis(meta):
    st.title('Forecasting Airline Realized Volatility')
    st.caption('HAR-RV + IV + OVX + TOSI  ·  walk-forward out-of-sample evaluation')

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Tickers**  \n{', '.join(meta['tickers'])}")
    c2.markdown(f"**Feature specs**  \n{len(meta['feature_specs'])} (HAR-RV, +IV, +OVX, +TOSI and combos)")
    c3.markdown(f"**Model families**  \n{', '.join(meta['model_families'])}")
    c4.markdown(f"**OOS window**  \n{meta['oos_start']} → {meta['oos_end']}")
    st.markdown('---')

    st.markdown('## The thesis')
    st.markdown(
        """
        **Variance Risk Premium (VRP)** is the gap between what option markets *expect*
        and what actually *happens*:
        """
    )
    st.latex(r'\text{VRP}_t = \log\text{RV}_{t+1} - \log\text{IV}_t')
    st.markdown(
        """
        When predicted VRP is positive, realized vol is expected to exceed implied vol —
        buy the straddle. When negative, sell it. If we forecast VRP more accurately than
        the market prices it, we earn a persistent premium.

        **HAR (Heterogeneous AutoRegressive)** decomposes volatility memory into
        daily, weekly (5-day), and monthly (22-day) components — short-, mid-, and
        long-horizon traders each contribute to today's price formation.
        """
    )

    st.markdown("## What we're testing")
    st.markdown(
        """
        Each additional feature has an economic story. We test whether the story
        shows up out-of-sample:

        | Spec                     | Hypothesis                                                              |
        | ------------------------ | ----------------------------------------------------------------------- |
        | **HAR-RV**               | Volatility clustering alone explains next-day vol                       |
        | **+ IV**                 | Options markets contain forward-looking information HAR misses          |
        | **+ OVX**                | Oil vol is a jet-fuel cost signal that leaks into airline vol          |
        | **+ TOSI**               | Oil-stock sentiment (Texas Oil Stock Index) provides additional signal |

        We evaluate with **Diebold-Mariano** tests on squared forecast errors and
        back out a straddle P&L using close-to-close returns against ATM premiums.
        """
    )

    st.info(
        "**Central TOSI hypothesis:** The Texas Oil Stock Index (TOSI) — a sentiment "
        "signal derived from oil-sector equity performance — adds statistically significant "
        "predictive power for airline realized volatility *beyond* what price-based indicators "
        "(HAR-RV, IV, OVX) already capture.  If oil-sector equity sentiment leads jet-fuel "
        "cost expectations, it should front-run airline vol moves that lagged RV and options "
        "pricing have not yet priced in."
    )

    st.markdown("## Why it's non-trivial")
    st.markdown(
        """
        * **IV contamination** — naively regressing RV on IV lets the target leak.
          We forecast the VRP *difference*, and IV enters only as the current level.
        * **Threshold selection** — the trading signal uses a |VRP| threshold chosen
          by nested walk-forward Sharpe, so the threshold itself isn't cherry-picked.
        * **Straddle economics** — P&L uses actual close-to-close |return| against
          the Black-Scholes ATM premium ($\\sqrt{2/\\pi} \\cdot \\sigma_{\\text{IV}}$),
          not intraday sqrt(RV). That includes overnight gaps the buyer actually receives.
        """
    )


def page_data(meta):
    st.title('Data')
    inventory = load_table('data_inventory')
    cleaning = load_table('cleaning_log')

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader('Inventory')
        st.dataframe(inventory, use_container_width=True, hide_index=True)
    with c2:
        st.subheader('Cleaning log')
        st.dataframe(cleaning, use_container_width=True, hide_index=True)

    st.subheader('Realized volatility across the sector')
    st.plotly_chart(load_figure('realized_vol'), use_container_width=True)

    st.subheader('Oil volatility, oil sentiment, and airline RV')
    st.plotly_chart(load_figure('oil_drivers'), use_container_width=True)

    st.subheader('Monthly correlation structure')
    st.plotly_chart(load_figure('correlation'), use_container_width=True)

    st.subheader('Driver scatter facets')
    tab_iv, tab_ovx, tab_tosi = st.tabs(['IV → next-day RV', 'OVX → next-month RV', 'TOSI → next-month RV'])
    with tab_iv:
        st.plotly_chart(load_figure('iv_scatter'), use_container_width=True)
    with tab_ovx:
        st.plotly_chart(load_figure('ovx_scatter'), use_container_width=True)
    with tab_tosi:
        st.plotly_chart(load_figure('tosi_scatter'), use_container_width=True)


def page_models(meta):
    st.title('Models')
    feature_specs = load_table('feature_specs')
    feature_map = load_table('feature_map')

    st.markdown(
        """
        We evaluate **four model families** across **six feature specifications**
        on a common walk-forward OOS schedule.  OLS and Ridge are linear;
        Random Forest and XGBoost capture non-linear interactions between
        HAR components and macro signals.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Feature specifications')
        st.dataframe(feature_specs, use_container_width=True, hide_index=True)
    with c2:
        st.subheader('Model families')
        st.dataframe(
            pd.DataFrame({
                'Family': meta['model_families'],
                'Role': [
                    'Baseline linear HAR',
                    'Regularised linear (scaled)',
                    'Non-linear, nested CV over depth & leaf size',
                    'Gradient-boosted, nested CV over depth & learning rate',
                ],
            }),
            use_container_width=True, hide_index=True,
        )

    st.subheader('Walk-forward protocol')
    render_metric_row([
        ('Min train days', '756', '3 trading years'),
        ('Test window', '63 days', '≈ one quarter'),
        ('Step', '63 days', 'Expanding training'),
        ('Threshold grid', '25–75 pctiles', 'of |predicted VRP| on training'),
    ])

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
            orientation='h', marker_color=meta['model_colors'].get(family, '#3C91E6'),
        ))
        fig.update_layout(
            height=520, margin=dict(l=160, r=30, t=40, b=30),
            template='plotly_white', paper_bgcolor='#F7F4ED', plot_bgcolor='#FFFFFF',
            title=f'{ticker} — {family} feature importance',
            xaxis_title='Importance', yaxis_title='Feature',
        )
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
    fig = go.Figure(go.Heatmap(
        z=rmse_pivot.values,
        x=rmse_pivot.columns, y=rmse_pivot.index,
        colorscale='RdYlGn_r',
        text=np.round(rmse_pivot.values, 4), texttemplate='%{text}',
        hovertemplate='Spec=%{y}<br>Family=%{x}<br>Avg RMSE=%{z:.4f}<extra></extra>',
        colorbar=dict(title='Avg RMSE'),
    ))
    fig.update_layout(
        height=460, template='plotly_white', paper_bgcolor='#F7F4ED',
        margin=dict(l=180, r=40, t=40, b=40),
    )
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
            name='Actual VRP', line=dict(color='#24323D', width=1.6),
        ))
        fig.add_trace(go.Scatter(
            x=view['trade_date'], y=view['y_pred_vrp'], mode='lines',
            name='Predicted VRP',
            line=dict(color=meta['airline_colors'].get(sel_t, '#3C91E6'), width=2.0),
        ))
        fig.update_layout(
            height=460, template='plotly_white', paper_bgcolor='#F7F4ED',
            title=f'{sel_t} — {sel_f} | {sel_s}',
            xaxis_title='Date', yaxis_title='VRP (log-variance)',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
        )
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


def page_strategy(meta):
    st.title('Strategy')
    st.markdown(
        """
        For each day in the OOS window the model predicts VRP.  If |predicted VRP|
        exceeds the per-fold threshold, we **take a position in the ATM straddle**:
        long when predicted VRP > 0 (expected realized > implied), short otherwise.

        **P&L per day** = `signal × (|close-to-close return| − √(2/π) × σ_IV)`
        """
    )

    st.subheader('JETS cumulative straddle P&L')
    st.plotly_chart(load_figure('jets_pnl'), use_container_width=True)

    preds = load_table('predictions').copy()
    results = load_table('results')
    preds['trade_date'] = pd.to_datetime(preds['trade_date'])
    sqrt2pi = float(meta['sqrt2_pi'])

    st.subheader('Build your own strategy — any ticker × spec × family')
    c1, c2, c3 = st.columns(3)
    sel_t = c1.selectbox('Ticker', meta['tickers'], key='strat_t')
    sel_s = c2.selectbox('Feature spec', list(meta['feature_specs'].keys()),
                         index=list(meta['feature_specs'].keys()).index('HAR-RV+IV+OVX+TOSI'), key='strat_s')
    sel_f = c3.selectbox('Model family', meta['model_families'], key='strat_f')

    view = preds[
        (preds['Ticker'] == sel_t) & (preds['Feature_Spec'] == sel_s) & (preds['Model_Family'] == sel_f)
    ].sort_values('trade_date').copy()
    if view.empty or 'iv_daily_vol' not in view.columns:
        st.info('No predictions for this combination.')
        return

    view['pnl_daily'] = view['signal'] * (view['abs_daily_return_tplus1'] - sqrt2pi * view['iv_daily_vol'])
    view['cum_pnl'] = view['pnl_daily'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view['trade_date'], y=view['cum_pnl'],
        mode='lines', name='Cumulative P&L',
        line=dict(color=meta['airline_colors'].get(sel_t, '#3C91E6'), width=2.2),
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='#7F8C8D')
    fig.update_layout(
        height=440, template='plotly_white', paper_bgcolor='#F7F4ED',
        title=f'{sel_t} — {sel_f} | {sel_s} cumulative straddle P&L',
        xaxis_title='Date', yaxis_title='Cumulative P&L (log-return units)',
        hovermode='x unified',
    )
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

    st.subheader('Per-fold threshold behavior')
    thresh = load_table('threshold_summary')
    st.dataframe(
        thresh[(thresh['Ticker'] == sel_t) & (thresh['Feature_Spec'] == sel_s) & (thresh['Model_Family'] == sel_f)],
        use_container_width=True, hide_index=True,
    )


def page_conclusion(meta):
    st.title('Conclusion')

    dm = load_table('dm_results')
    best = load_table('best_models')
    summary = load_table('summary')

    best_ols_full = summary[
        (summary['Feature_Spec'] == 'HAR-RV+IV+OVX+TOSI') & (summary['Model_Family'] == 'OLS')
    ].iloc[0]
    base_ols = summary[
        (summary['Feature_Spec'] == 'HAR-RV') & (summary['Model_Family'] == 'OLS')
    ].iloc[0]

    render_metric_row([
        ('Best avg RMSE', format_num(best_ols_full['Avg_RMSE'], 4),
         'Full spec, OLS — lowest of 24 spec×family combos'),
        ('Baseline HAR-RV RMSE', format_num(base_ols['Avg_RMSE'], 4), 'OLS only'),
        ('Best ticker Sharpe',
         format_num(best['Sharpe_Straddle'].max(), 2),
         f"{best.loc[best['Sharpe_Straddle'].idxmax(), 'Ticker']} — "
         f"{best.loc[best['Sharpe_Straddle'].idxmax(), 'Model']}"),
        ('Avg directional acc',
         format_pct(summary['Avg_Directional_Acc'].mean()),
         'Sign-of-VRP correct'),
    ])

    st.markdown('## Headline')
    st.markdown(
        """
        The HAR baseline is hard to beat.  Across 4 tickers × 6 specifications × 4 model
        families, the best-average-RMSE spec (**HAR-RV+IV+OVX+TOSI, OLS**) improves on
        the plain HAR-RV baseline by less than **1% in RMSE** — a marginal statistical
        lift, but one that compounds into real economic difference via the threshold-
        selected straddle trade for certain tickers.
        """
    )

    st.markdown('## Hypothesis scorecard')

    def _verdict_row(hypothesis_label, friendly):
        rows = dm[dm['Hypothesis'] == hypothesis_label]
        sig_tickers = rows[rows['Significant_5pct']]['Ticker'].tolist()
        min_p = rows['P_Value'].min()
        return {
            'Hypothesis': friendly,
            'Significant (p<0.05)': ', '.join(sig_tickers) if sig_tickers else 'None',
            'Best p-value': f"{min_p:.4f}",
            'Verdict': '✅ Supported' if sig_tickers else '❌ Not supported',
        }

    scorecard = pd.DataFrame([
        _verdict_row('IV adds to HAR-RV?',          'IV improves HAR-RV'),
        _verdict_row('OVX adds to HAR-RV?',         'OVX improves HAR-RV'),
        _verdict_row('OVX adds to HAR-RV+IV?',      'OVX improves HAR-RV+IV'),
        _verdict_row('TOSI adds to HAR-RV+OVX?',    'TOSI improves HAR-RV+OVX'),
        _verdict_row('TOSI adds to HAR-RV+IV+OVX?', 'TOSI improves HAR-RV+IV+OVX'),
        _verdict_row('IV adds to HAR-RV+OVX?',      'IV improves HAR-RV+OVX'),
    ])
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

    st.markdown('## What we learned')
    st.markdown(
        """
        **1. HAR captures most of the signal.**  Volatility is highly persistent — daily,
        weekly, and monthly RV components already explain ~5–6% of next-day log-variance
        and achieve ~76% directional accuracy on the VRP target.  Any extra feature
        has to beat a very strong baseline.

        **2. IV has genuine forward-looking content — but only visibly in JETS.**
        JETS is the only ticker where the Diebold-Mariano test rejects "HAR-RV equals
        HAR-RV+IV" at the 5% level (p = 0.0014, DM stat = +3.19).  The ETF aggregates
        airline-specific noise, and the option market's forward view appears to price
        the sector-wide component that single-name HAR misses.  For the individual
        airlines (DAL, UAL, LUV) the IV lift is directionally positive for DAL and UAL
        but doesn't cross the 5% bar.

        **3. OVX alone adds nothing.**  Every DM test for "OVX adds to HAR-RV" has a
        negative stat and large p-value.  Oil vol co-moves with airline vol
        contemporaneously (the correlation heatmap shows that) but it does not
        *lead* next-day RV once HAR already knows recent RV history.
        """
    )

    st.markdown('## On the TOSI hypothesis')
    st.markdown(
        """
        We hypothesised that Texas Oil Stock Index sentiment would contain additional
        signal about airline vol beyond price-based indicators (IV, OVX).  The evidence
        is **mixed but leans against it**:

        * **No ticker shows a DM-significant TOSI effect.**  Across the two tests
          ("TOSI adds to HAR-RV+OVX" and "TOSI adds to HAR-RV+IV+OVX"), the smallest
          p-value is **0.10 (DAL, HAR-RV+OVX branch)** — suggestive but not significant
          at 5%.  For LUV the test stat is actually *negative*, meaning TOSI hurt.
        * **However, the full spec HAR-RV+IV+OVX+TOSI does produce the lowest average
          RMSE** of any spec×family combination (OLS, RMSE 0.7637 vs HAR-RV OLS 0.7684),
          and is chosen as the *per-ticker best* for 3 of 4 names (DAL, UAL, JETS).
          So TOSI isn't harmful and the combination of predictors nudges point forecasts
          forward, even though no single TOSI-increment test rejects equality.

        **Reconciling the two views:**  TOSI's incremental contribution is small enough
        that the Diebold-Mariano test lacks power to detect it over a ~450-day OOS
        window, but large enough in direction to improve ensemble fit.  A longer
        sample — or an event-study conditioning on actual oil-sector stress periods —
        would be needed to settle the case.  For now we treat TOSI as a **weak
        auxiliary**: include it, but don't base a trading decision on it alone.
        """
    )

    st.markdown('## Trading implications')
    st.markdown(
        """
        * **JETS straddle edge is real.**  Full-spec OLS yields OOS Sharpe ≈ 2.7 on
          threshold-filtered ATM straddles — economically meaningful after cost.
        * **LUV is the outlier.**  Plain HAR-RV + Random Forest wins on LUV; extra
          macro features degrade performance, consistent with LUV's idiosyncratic
          route concentration (domestic, narrow-body) differing from sector aggregate.
        * **Single-name tests are hard.**  Statistical power is limited;
          ensemble spec + threshold selection is how the marginal signal becomes tradeable.
        """
    )


# ---------------- App shell ----------------

def main():
    if not artifacts_ready():
        st.error(
            'Precomputed artifacts not found.  Run HAR_model.ipynb end-to-end first '
            'to populate `data/processed/`.'
        )
        st.stop()

    meta = load_meta()

    st.sidebar.title('✈️ Airline Volatility')
    st.sidebar.caption('CS 329e · Group 23')
    page = st.sidebar.radio(
        'Section',
        ['Hypothesis', 'Data', 'Models', 'Results', 'Strategy', 'Conclusion'],
        label_visibility='collapsed',
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**OOS window**\n\n{meta['oos_start']} → {meta['oos_end']}"
    )
    st.sidebar.markdown(
        f"**Tickers**  \n{', '.join(meta['tickers'])}"
    )

    {
        'Hypothesis': page_hypothesis,
        'Data':       page_data,
        'Models':     page_models,
        'Results':    page_results,
        'Strategy':   page_strategy,
        'Conclusion': page_conclusion,
    }[page](meta)


if __name__ == '__main__':
    main()
