import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from IPython.display import display
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

pio.renderers.default = 'notebook_connected'
pd.options.display.float_format = '{:,.3f}'.format

DATA_DIR = Path('data')
TICKERS = ['AAL', 'DAL', 'UAL', 'LUV', 'JETS']
AIRLINE_COLORS = {
    'AAL': '#C75146',
    'DAL': '#0B6E4F',
    'UAL': '#3C91E6',
    'LUV': '#F4A259',
    'JETS': '#7A5195',
}
MODEL_COLORS = {
    'MIDAS | OVX Only': '#8C564B',
    'MIDAS | TOSI Only': '#4C78A8',
    'MIDAS | OVX + TOSI': '#2D3047',
    'XGBoost | OVX Only': '#E45756',
    'XGBoost | TOSI Only': '#72B7B2',
    'XGBoost | OVX + TOSI': '#1B998B',
    'Random Forest | OVX Only': '#F58518',
    'Random Forest | TOSI Only': '#FF9DA6',
    'Random Forest | OVX + TOSI': '#FF6B35',
    'OVX Only': '#C75146',
    'TOSI Only': '#3C91E6',
    'OVX + TOSI': '#0B6E4F',
}
CRISIS_PERIODS = [
    ('2008-09-01', '2009-06-30', 'Financial crisis'),
    ('2020-02-01', '2020-07-31', 'COVID travel shock'),
    ('2022-02-01', '2022-08-31', 'Energy shock'),
]
BASE_LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='#F7F4ED',
    plot_bgcolor='#FFFFFF',
    font=dict(family='Aptos, Segoe UI, sans-serif', size=13, color='#24323D'),
    colorway=['#0B6E4F', '#C75146', '#3C91E6', '#F4A259', '#7A5195', '#2D3047'],
    hoverlabel=dict(bgcolor='white', font_size=12),
    margin=dict(l=60, r=30, t=80, b=55),
)


def style_figure(fig, title, height=550, legend_y=1.08):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor='left'),
        height=height,
        legend=dict(orientation='h', y=legend_y, x=0),
        hovermode='x unified',
        **BASE_LAYOUT,
    )
    return fig


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def directional_accuracy(actual, predicted, previous):
    actual_direction = np.sign(np.asarray(actual) - np.asarray(previous))
    predicted_direction = np.sign(np.asarray(predicted) - np.asarray(previous))
    return float((actual_direction == predicted_direction).mean())


def evaluate_forecast(actual, predicted, previous):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    previous = np.asarray(previous, dtype=float)
    errors = actual - predicted
    abs_actual = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    smape_denom = np.abs(actual) + np.abs(predicted)

    return {
        'RMSE': rmse(actual, predicted),
        'MAE': float(mean_absolute_error(actual, predicted)),
        'MedAE': float(median_absolute_error(actual, predicted)),
        'MAPE': float(np.nanmean(np.abs(errors) / abs_actual) * 100),
        'sMAPE': float(np.nanmean(2 * np.abs(errors) / np.where(smape_denom < 1e-8, np.nan, smape_denom)) * 100),
        'Bias': float(np.mean(predicted - actual)),
        'Explained_Variance': float(explained_variance_score(actual, predicted)) if len(actual) > 1 else np.nan,
        'R2': float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        'Directional_Accuracy': directional_accuracy(actual, predicted, previous),
    }


def almon_weighted_average(values, theta1=-0.15, theta2=-0.01):
    values = np.asarray(values, dtype=float)
    lags = np.arange(len(values), dtype=float)
    raw = np.exp(theta1 * lags + theta2 * (lags ** 2))
    weights = raw / raw.sum()
    return float(np.dot(values[::-1], weights))


THETA1_GRID = [-0.35, -0.20, -0.10, -0.05, 0.00]
THETA2_GRID = [0.00, -0.005, -0.010, -0.020]


def polynomial_fit_scores(x, y):
    scores = {}
    for degree, label in [(1, 'Linear'), (2, 'Quadratic'), (3, 'Cubic')]:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        pred = poly(x)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        scores[f'R2_{label}'] = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    scores['Best_Fit'] = max(['Linear', 'Quadratic', 'Cubic'], key=lambda name: scores[f'R2_{name}'])
    return scores


def load_project_data():
    iv_data = {}
    inventory_rows = []
    cleaning_rows = []

    for ticker in TICKERS:
        raw = pd.read_excel(DATA_DIR / f'{ticker}_IV.xlsx', skiprows=6)
        raw['Date'] = pd.to_datetime(raw['Date'])
        col = f'{ticker}_Volatility'
        raw = raw.rename(columns={'HIST_CALL_IMP_VOL': col})[['Date', col]].sort_values('Date').reset_index(drop=True)
        missing_before = int(raw[col].isna().sum())
        raw_rows = len(raw)

        clean = raw.copy()
        clean[col] = clean[col].bfill()
        clean = clean.dropna(subset=[col]).reset_index(drop=True)
        iv_data[ticker] = clean

        inventory_rows.append({
            'Dataset': f'{ticker} IV',
            'Rows': raw_rows,
            'Columns': 1,
            'Start': raw['Date'].min().date(),
            'End': raw['Date'].max().date(),
            'Missing_Before_Cleaning': missing_before,
        })
        cleaning_rows.append({
            'Dataset': f'{ticker} IV',
            'Method': 'Backfill then drop leading structural gaps',
            'Rows_Removed': raw_rows - len(clean),
            'Missing_Before': missing_before,
            'Missing_After': int(clean[col].isna().sum()),
        })

    ovx_raw = pd.read_csv(DATA_DIR / 'OVXCLS.csv')
    ovx_raw['observation_date'] = pd.to_datetime(ovx_raw['observation_date'])
    ovx_raw = ovx_raw.rename(columns={'observation_date': 'Date', 'OVXCLS': 'OVX'})
    ovx_raw['OVX'] = pd.to_numeric(ovx_raw['OVX'], errors='coerce')
    ovx_missing = int(ovx_raw['OVX'].isna().sum())
    ovx = ovx_raw.dropna(subset=['OVX']).sort_values('Date').reset_index(drop=True)

    inventory_rows.append({
        'Dataset': 'OVX',
        'Rows': len(ovx_raw),
        'Columns': 1,
        'Start': ovx_raw['Date'].min().date(),
        'End': ovx_raw['Date'].max().date(),
        'Missing_Before_Cleaning': ovx_missing,
    })
    cleaning_rows.append({
        'Dataset': 'OVX',
        'Method': 'Drop holiday rows with no published index value',
        'Rows_Removed': len(ovx_raw) - len(ovx),
        'Missing_Before': ovx_missing,
        'Missing_After': int(ovx['OVX'].isna().sum()),
    })

    tosi = pd.read_csv(DATA_DIR / 'TOSI.csv', usecols=['Date', 'TOSI'])
    tosi['Date'] = pd.to_datetime(tosi['Date'], format='%b-%y')
    tosi = tosi.sort_values('Date').reset_index(drop=True)

    inventory_rows.append({
        'Dataset': 'TOSI',
        'Rows': len(tosi),
        'Columns': 1,
        'Start': tosi['Date'].min().date(),
        'End': tosi['Date'].max().date(),
        'Missing_Before_Cleaning': int(tosi['TOSI'].isna().sum()),
    })
    cleaning_rows.append({
        'Dataset': 'TOSI',
        'Method': 'No cleaning needed',
        'Rows_Removed': 0,
        'Missing_Before': int(tosi['TOSI'].isna().sum()),
        'Missing_After': int(tosi['TOSI'].isna().sum()),
    })

    daily_panel = ovx.set_index('Date')[['OVX']].copy()
    for ticker in TICKERS:
        col = f'{ticker}_Volatility'
        daily_panel = daily_panel.join(iv_data[ticker].set_index('Date')[[col]], how='left')
    daily_panel = daily_panel.join(tosi.set_index('Date')[['TOSI']], how='left')
    daily_panel['TOSI'] = daily_panel['TOSI'].ffill()
    daily_panel = daily_panel.sort_index()

    monthly_iv = [
        iv_data[ticker].set_index('Date')[[f'{ticker}_Volatility']].resample('MS').mean()
        for ticker in TICKERS
    ]
    monthly_panel = pd.concat(monthly_iv, axis=1)
    ovx_monthly = ovx.set_index('Date').resample('MS').agg(
        OVX_mean=('OVX', 'mean'),
        OVX_max=('OVX', 'max'),
        OVX_min=('OVX', 'min'),
        OVX_std=('OVX', 'std'),
    )
    monthly_panel = monthly_panel.join(ovx_monthly, how='outer')
    monthly_panel = monthly_panel.join(tosi.set_index('Date')[['TOSI']], how='outer').sort_index()

    return (
        iv_data,
        ovx,
        tosi,
        daily_panel,
        monthly_panel,
        pd.DataFrame(inventory_rows),
        pd.DataFrame(cleaning_rows),
    )


def build_midas_windows(monthly_target, ovx_daily, tosi_monthly, horizon_months, trading_days_per_month=21):
    monthly_target = monthly_target.dropna().sort_index()
    rows = []

    for feature_month in monthly_target.index:
        target_month = feature_month + pd.offsets.MonthBegin(1)
        if target_month not in monthly_target.index:
            continue

        month_end = feature_month + pd.offsets.MonthEnd(0)
        ovx_window = ovx_daily.loc[:month_end].tail(horizon_months * trading_days_per_month)
        tosi_window = tosi_monthly.loc[:feature_month].tail(horizon_months)

        if len(ovx_window) < horizon_months * 15 or len(tosi_window) < horizon_months:
            continue

        rows.append({
            'feature_month': feature_month,
            'target_date': target_month,
            'previous_vol': float(monthly_target.loc[feature_month]),
            'target_next_month': float(monthly_target.loc[target_month]),
            'ovx_window': ovx_window.to_numpy(dtype=float),
            'tosi_window': tosi_window.to_numpy(dtype=float),
        })

    return pd.DataFrame(rows).set_index('feature_month')


def build_midas_design(window_df, spec, theta_params):
    design = pd.DataFrame(index=window_df.index)
    if 'OVX' in spec:
        theta1, theta2 = theta_params['ovx']
        design['OVX_midas'] = window_df['ovx_window'].apply(lambda arr: almon_weighted_average(arr, theta1, theta2))
    if 'TOSI' in spec:
        theta1, theta2 = theta_params['tosi']
        design['TOSI_midas'] = window_df['tosi_window'].apply(lambda arr: almon_weighted_average(arr, theta1, theta2))
    return design


def tune_midas_weights(window_df, spec):
    train_cut = max(24, int(np.ceil(len(window_df) * 0.75)))
    tune_cut = max(18, train_cut - 12)
    tune_train = window_df.iloc[:tune_cut]
    tune_val = window_df.iloc[tune_cut:train_cut]
    if tune_val.empty:
        tune_train = window_df.iloc[:-6]
        tune_val = window_df.iloc[-6:]

    candidate_pairs = [(t1, t2) for t1 in THETA1_GRID for t2 in THETA2_GRID]
    ovx_pairs = candidate_pairs if 'OVX' in spec else [(None, None)]
    tosi_pairs = candidate_pairs if 'TOSI' in spec else [(None, None)]

    best_score = np.inf
    best_params = {}

    for ovx_pair in ovx_pairs:
        for tosi_pair in tosi_pairs:
            theta_params = {}
            if 'OVX' in spec:
                theta_params['ovx'] = ovx_pair
            if 'TOSI' in spec:
                theta_params['tosi'] = tosi_pair

            X_train = build_midas_design(tune_train, spec, theta_params)
            X_val = build_midas_design(tune_val, spec, theta_params)
            model = LinearRegression().fit(X_train, tune_train['target_next_month'])
            pred = model.predict(X_val)
            score = rmse(tune_val['target_next_month'], pred)

            if score < best_score:
                best_score = score
                best_params = theta_params.copy()

    return best_params


def rolling_midas_forecast(window_df, spec, min_train=36):
    if len(window_df) <= min_train:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Directional_Accuracy': np.nan}, pd.DataFrame(), {}

    start_test = max(min_train, int(np.ceil(len(window_df) * 0.70)))
    theta_params = tune_midas_weights(window_df.iloc[:start_test], spec)
    prediction_rows = []

    for end_idx in range(start_test, len(window_df)):
        train = window_df.iloc[:end_idx]
        test = window_df.iloc[[end_idx]]

        X_train = build_midas_design(train, spec, theta_params)
        X_test = build_midas_design(test, spec, theta_params)
        model = LinearRegression().fit(X_train, train['target_next_month'])
        pred = float(model.predict(X_test)[0])

        prediction_rows.append({
            'target_date': test['target_date'].iloc[0],
            'previous_vol': float(test['previous_vol'].iloc[0]),
            'actual': float(test['target_next_month'].iloc[0]),
            'predicted': pred,
        })

    prediction_df = pd.DataFrame(prediction_rows)
    metrics = evaluate_forecast(prediction_df['actual'], prediction_df['predicted'], prediction_df['previous_vol'])
    return metrics, prediction_df, theta_params


def build_monthly_ml_frame(monthly_panel, ticker, feature_spec='OVX + TOSI', lags=(1, 3, 6)):
    target_col = f'{ticker}_Volatility'
    base_cols = [target_col, 'OVX_mean', 'OVX_max', 'OVX_min', 'OVX_std', 'TOSI']
    frame = monthly_panel[base_cols].dropna().copy()
    feature_cols = []

    include_ovx = feature_spec in ['OVX Only', 'OVX + TOSI']
    include_tosi = feature_spec in ['TOSI Only', 'OVX + TOSI']

    for lag in lags:
        for col in ['OVX_mean', 'OVX_max', 'OVX_min', 'OVX_std', 'TOSI', target_col]:
            feature_name = f'{col}_lag{lag}m'
            frame[feature_name] = frame[col].shift(lag)
            if col.startswith('OVX') and include_ovx:
                feature_cols.append(feature_name)
            elif col == 'TOSI' and include_tosi:
                feature_cols.append(feature_name)
            elif col == target_col:
                feature_cols.append(feature_name)

    frame['OVX_range_lag1m'] = frame['OVX_max'].shift(1) - frame['OVX_min'].shift(1)
    if include_ovx:
        feature_cols.append('OVX_range_lag1m')
    frame['target_next_month'] = frame[target_col].shift(-1)
    frame['previous_vol'] = frame[target_col]
    frame['target_date'] = frame.index + pd.offsets.MonthBegin(1)
    frame = frame.dropna(subset=feature_cols + ['target_next_month'])
    return frame, feature_cols


def make_xgb_model():
    return xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=1.25,
        random_state=42,
    )


def make_rf_model():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )


def time_series_cv(df_ml, feature_cols, spec_label='OVX + TOSI'):
    split_idx = int(len(df_ml) * 0.80)
    X_train = df_ml[feature_cols].iloc[:split_idx]
    y_train = df_ml['target_next_month'].iloc[:split_idx]
    n_splits = min(5, max(2, len(X_train) // 12))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for model_name, factory in [('XGBoost', make_xgb_model), ('Random Forest', make_rf_model)]:
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train), start=1):
            model = factory()
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            rows.append({
                'Model': model_name,
                'Specification': spec_label,
                'Fold': fold,
                'RMSE': rmse(y_train.iloc[val_idx], pred),
            })
    return pd.DataFrame(rows)


iv_data, ovx, tosi, daily_panel, monthly_panel, data_inventory, cleaning_log = load_project_data()

monthly_corr_cols = [f'{ticker}_Volatility' for ticker in TICKERS] + ['OVX_mean', 'TOSI']
monthly_corr = monthly_panel[monthly_corr_cols].corr()
full_overlap = monthly_panel[monthly_corr_cols].dropna()

fit_rows = []
for ticker in TICKERS:
    vol_col = f'{ticker}_Volatility'

    ovx_pair = monthly_panel[['OVX_mean', vol_col]].dropna()
    ovx_stats = polynomial_fit_scores(ovx_pair['OVX_mean'].to_numpy(), ovx_pair[vol_col].to_numpy())
    fit_rows.append({
        'Pair': f'OVX vs {ticker}',
        'Correlation': ovx_pair['OVX_mean'].corr(ovx_pair[vol_col]),
        **ovx_stats,
    })

    tosi_pair = monthly_panel[['TOSI', vol_col]].dropna()
    tosi_stats = polynomial_fit_scores(tosi_pair['TOSI'].to_numpy(), tosi_pair[vol_col].to_numpy())
    fit_rows.append({
        'Pair': f'TOSI vs {ticker}',
        'Correlation': tosi_pair['TOSI'].corr(tosi_pair[vol_col]),
        **tosi_stats,
    })

fit_summary = pd.DataFrame(fit_rows)

print(f'Daily modeling panel: {daily_panel.shape[0]:,} trading days from {daily_panel.index.min().date()} to {daily_panel.index.max().date()}')
print(f'Monthly full-overlap panel: {full_overlap.shape[0]} months from {full_overlap.index.min().date()} to {full_overlap.index.max().date()}')


ovx_daily = ovx.set_index('Date')['OVX']
tosi_monthly = tosi.set_index('Date')['TOSI']

SPECS = ['OVX Only', 'TOSI Only', 'OVX + TOSI']
MODEL_FAMILIES = ['MIDAS', 'XGBoost', 'Random Forest']

midas_horizon_rows = []
midas_jets_predictions = {}

for horizon in [1, 3, 6]:
    jets_windows = build_midas_windows(monthly_panel['JETS_Volatility'], ovx_daily, tosi_monthly, horizon)
    for spec in SPECS:
        metrics, pred_df, theta_params = rolling_midas_forecast(jets_windows, spec)
        midas_horizon_rows.append({
            'Horizon_Months': horizon,
            'Horizon_Label': f'{horizon}-month',
            'Specification': spec,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MedAE': metrics['MedAE'],
            'MAPE': metrics['MAPE'],
            'sMAPE': metrics['sMAPE'],
            'Bias': metrics['Bias'],
            'Explained_Variance': metrics['Explained_Variance'],
            'R2': metrics['R2'],
            'Directional_Accuracy': metrics['Directional_Accuracy'],
            'Test_Months': len(pred_df),
            'OVX_Theta': theta_params.get('ovx'),
            'TOSI_Theta': theta_params.get('tosi'),
        })
        if spec == 'OVX + TOSI':
            midas_jets_predictions[horizon] = pred_df.copy()

midas_horizon_df = pd.DataFrame(midas_horizon_rows).sort_values(['Horizon_Months', 'Specification']).reset_index(drop=True)

midas_rows = []
midas_predictions = {}

for ticker in TICKERS:
    windows = build_midas_windows(monthly_panel[f'{ticker}_Volatility'], ovx_daily, tosi_monthly, 1)
    for spec in SPECS:
        metrics, pred_df, theta_params = rolling_midas_forecast(windows, spec)
        model_label = f'MIDAS | {spec}'
        if spec == 'OVX + TOSI':
            midas_predictions[ticker] = pred_df.copy()
        midas_rows.append({
            'Ticker': ticker,
            'Family': 'MIDAS',
            'Specification': spec,
            'Model': model_label,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MedAE': metrics['MedAE'],
            'MAPE': metrics['MAPE'],
            'sMAPE': metrics['sMAPE'],
            'Bias': metrics['Bias'],
            'Explained_Variance': metrics['Explained_Variance'],
            'R2': metrics['R2'],
            'Directional_Accuracy': metrics['Directional_Accuracy'],
            'Test_Months': len(pred_df),
            'OVX_Theta': theta_params.get('ovx'),
            'TOSI_Theta': theta_params.get('tosi'),
        })

midas_all_df = pd.DataFrame(midas_rows)

ml_rows = []
ml_predictions = {}
feature_importances = {}

for ticker in TICKERS:
    combined_prediction_frame = None
    for spec in SPECS:
        df_ml, feature_cols = build_monthly_ml_frame(monthly_panel, ticker, feature_spec=spec)
        split_idx = int(len(df_ml) * 0.80)

        X_train = df_ml[feature_cols].iloc[:split_idx]
        X_test = df_ml[feature_cols].iloc[split_idx:]
        y_train = df_ml['target_next_month'].iloc[:split_idx]
        y_test = df_ml['target_next_month'].iloc[split_idx:]
        previous_test = df_ml['previous_vol'].iloc[split_idx:]

        prediction_frame = pd.DataFrame({
            'target_date': df_ml['target_date'].iloc[split_idx:].to_numpy(),
            'actual': y_test.to_numpy(),
            'previous_vol': previous_test.to_numpy(),
        })

        for model_name, factory in [('XGBoost', make_xgb_model), ('Random Forest', make_rf_model)]:
            model = factory()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = evaluate_forecast(y_test, pred, previous_test)
            model_label = f'{model_name} | {spec}'

            ml_rows.append({
                'Ticker': ticker,
                'Family': model_name,
                'Specification': spec,
                'Model': model_label,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MedAE': metrics['MedAE'],
                'MAPE': metrics['MAPE'],
                'sMAPE': metrics['sMAPE'],
                'Bias': metrics['Bias'],
                'Explained_Variance': metrics['Explained_Variance'],
                'R2': metrics['R2'],
                'Directional_Accuracy': metrics['Directional_Accuracy'],
                'Train_Months': len(X_train),
                'Test_Months': len(X_test),
            })

            if spec == 'OVX + TOSI':
                prediction_frame[model_name] = pred
                feature_importances.setdefault(ticker, {})[model_name] = pd.Series(
                    model.feature_importances_,
                    index=feature_cols,
                ).sort_values(ascending=False)

        if spec == 'OVX + TOSI':
            combined_prediction_frame = prediction_frame.copy()

    ml_predictions[ticker] = combined_prediction_frame

ml_results_df = pd.DataFrame(ml_rows)

cv_rows = []
for spec in SPECS:
    jets_ml_frame, jets_feature_cols = build_monthly_ml_frame(monthly_panel, 'JETS', feature_spec=spec)
    cv_rows.append(time_series_cv(jets_ml_frame, jets_feature_cols, spec_label=spec))
cv_results_df = pd.concat(cv_rows, ignore_index=True)
cv_summary_df = cv_results_df.groupby(['Model', 'Specification'], as_index=False).agg(
    Fold_RMSE_Mean=('RMSE', 'mean'),
    Fold_RMSE_Std=('RMSE', 'std'),
)

model_comparison_df = pd.concat([
    midas_all_df[['Ticker', 'Family', 'Specification', 'Model', 'RMSE', 'MAE', 'MedAE', 'MAPE', 'sMAPE', 'Bias', 'Explained_Variance', 'R2', 'Directional_Accuracy']],
    ml_results_df[['Ticker', 'Family', 'Specification', 'Model', 'RMSE', 'MAE', 'MedAE', 'MAPE', 'sMAPE', 'Bias', 'Explained_Variance', 'R2', 'Directional_Accuracy']],
], ignore_index=True)

best_models_df = model_comparison_df.loc[
    model_comparison_df.groupby('Ticker')['RMSE'].idxmin()
].sort_values('Ticker').reset_index(drop=True)

family_spec_summary_df = model_comparison_df.groupby(['Family', 'Specification'], as_index=False).agg(
    Avg_RMSE=('RMSE', 'mean'),
    Avg_MAE=('MAE', 'mean'),
    Avg_MAPE=('MAPE', 'mean'),
    Avg_R2=('R2', 'mean'),
    Avg_Directional_Accuracy=('Directional_Accuracy', 'mean'),
)

metrics_snapshot_df = model_comparison_df[[
    'Ticker', 'Family', 'Specification', 'Model', 'RMSE', 'MAE', 'MedAE', 'MAPE', 'sMAPE',
    'Bias', 'Explained_Variance', 'R2', 'Directional_Accuracy'
]].sort_values(['Ticker', 'RMSE']).reset_index(drop=True)

aggregate_metrics_df = model_comparison_df.groupby('Model', as_index=False).agg(
    Avg_RMSE=('RMSE', 'mean'),
    Avg_MAE=('MAE', 'mean'),
    Avg_MedAE=('MedAE', 'mean'),
    Avg_MAPE=('MAPE', 'mean'),
    Avg_sMAPE=('sMAPE', 'mean'),
    Avg_Abs_Bias=('Bias', lambda s: np.mean(np.abs(s))),
    Avg_Explained_Variance=('Explained_Variance', 'mean'),
    Avg_R2=('R2', 'mean'),
    Avg_Directional_Accuracy=('Directional_Accuracy', 'mean'),
).sort_values('Avg_RMSE').reset_index(drop=True)

visual_catalog = pd.DataFrame([
    {'Visual': '1. Daily airline volatility lines', 'Interactive_Element': 'Range slider + hover + crisis shading', 'Annotated': 'Yes'},
    {'Visual': '2. OVX/TOSI dual-axis drivers', 'Interactive_Element': 'Unified hover + crisis callouts', 'Annotated': 'Yes'},
    {'Visual': '3. Monthly correlation heatmap', 'Interactive_Element': 'Hover correlation lookup', 'Annotated': 'Yes'},
    {'Visual': '4. OVX vs airline scatter facets', 'Interactive_Element': 'Hover + fitted cubic curves', 'Annotated': 'Yes'},
    {'Visual': '5. TOSI vs airline scatter facets', 'Interactive_Element': 'Hover + fitted trend lines', 'Annotated': 'Yes'},
    {'Visual': '6. Model/spec comparison bars', 'Interactive_Element': 'Hover metrics + best-model labels', 'Annotated': 'Yes'},
    {'Visual': '7. Forecast comparison panels', 'Interactive_Element': 'Legend filtering + hover', 'Annotated': 'Yes'},
    {'Visual': '8. MIDAS horizon comparison', 'Interactive_Element': 'Grouped interactive bars', 'Annotated': 'Yes'},
    {'Visual': '9. Feature importance panels', 'Interactive_Element': 'Hover-ranked features', 'Annotated': 'Yes'},
])
print('Modeling complete.')
print('Best RMSE by ticker:')
print(best_models_df[['Ticker', 'Model', 'RMSE']].to_string(index=False))


