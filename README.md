# Forecasting Airline Stock Volatility with HAR-RV, IV, OVX, and TOSI

**CS 329E — Elements of Data Science | Group 23**
Blake Stanley, Shivsagar Palla, Raghuvendra Chowdhry

---

## Goal

Forecast the **variance risk premium (VRP)** — the difference between realized and implied volatility — for major U.S. airlines (AAL, DAL, UAL, LUV) and the JETS ETF, and use that forecast to drive a volatility trading strategy (buy/sell straddles).

We investigate whether crude oil market signals — the **CBOE Crude Oil Volatility Index (OVX)** and the **Text Oil Sentiment Indicator (TOSI)** — improve forecast accuracy beyond the baseline HAR-RV model.

---

## Repository Structure

```
.
├── HAR_model.ipynb                  # Final notebook — HAR-RV modeling and vol strategy
├── app.py                           # Streamlit dashboard (reads data/processed/)
├── Predicting_Airline_Stock_Volatility.ipynb  # Earlier exploration notebook
├── CS329E_Phase2_Group23.ipynb      # Phase 2 submission notebook
├── analysis.ipynb                   # Supporting analysis
├── data/
│   ├── alpaca_intraday/             # 5-minute bars (Alpaca) for RV computation
│   ├── OVXCLS.csv                   # CBOE Crude Oil Volatility Index (daily)
│   ├── TOSI.csv                     # Text Oil Sentiment Indicator (monthly)
│   ├── {AAL,DAL,UAL,LUV,JETS}_IV.xlsx  # Implied volatility per ticker
│   ├── {aal,dal,ual,luv,jet}_gjr-garch.csv  # GJR-GARCH estimates
│   └── processed/                   # Artifacts output by HAR_model.ipynb
└── requirements.txt
```

---

## Datasets

| File | Description | Frequency |
|------|-------------|-----------|
| `data/alpaca_intraday/` | 5-minute OHLCV bars for AAL, DAL, UAL, LUV, JETS | Intraday |
| `data/OVXCLS.csv` | CBOE Crude Oil Volatility Index (OVX) | Daily |
| `data/TOSI.csv` | Text Oil Sentiment Indicator | Monthly |
| `data/*_IV.xlsx` | ATM implied volatility per ticker | Daily |
| `data/*_gjr-garch.csv` | GJR-GARCH conditional variance estimates | Daily |

Realized variance is computed directly from the Alpaca 5-minute bars inside `HAR_model.ipynb`.

---

## Final Notebook: HAR_model.ipynb

This is the primary deliverable. It:

1. Computes **realized variance (RV)** from raw 5-minute intraday bars
2. Reframes the forecast target as **VRP**: `y_{t+1} = log(RV_{t+1}) − log(IV_t)`
3. Builds six **HAR-style feature specifications**:
   - HAR-RV
   - HAR-RV + OVX
   - HAR-RV + OVX + TOSI
   - HAR-RV + IV
   - HAR-RV + IV + OVX
   - HAR-RV + IV + OVX + TOSI
4. Estimates **walk-forward forecasts** (min 756-day train, 63-day test folds) for four model families:
   - OLS
   - Ridge
   - Random Forest
   - XGBoost
5. Selects trading thresholds **adaptively** inside each fold using a nested validation window (no look-ahead)
6. Evaluates a **straddle trading strategy** (Sharpe ratio) on held-out test folds
7. Writes all results and figures to `data/processed/`

---

## Streamlit Dashboard

`app.py` provides an interactive dashboard over the precomputed artifacts in `data/processed/`.

```bash
streamlit run app.py
```

Requires `HAR_model.ipynb` to have been run first to populate `data/processed/`.

---

## Models

| Model | Feature Specifications |
|-------|----------------------|
| OLS | 6 specs (see above) |
| Ridge | 6 specs |
| Random Forest | 6 specs |
| XGBoost | 6 specs |

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `matplotlib`, `plotly`, `scikit-learn`, `xgboost`, `statsmodels`, `scipy`, `streamlit`, `openpyxl`

---

## Running

```bash
# 1. Run the final model notebook (Alpaca data is already local)
jupyter notebook HAR_model.ipynb

# 2. Launch the dashboard
streamlit run app.py
```
