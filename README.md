# Forecasting Airline Stock Volatility with Oil Futures Volatility and Oil News Sentiment

**CS 329E — Elements of Data Science | Group 23**
Blake Stanley, Shivsagar Palla, Raghuvendra Chowdhry

---

## Goal

Investigate whether volatility in crude oil futures — measured by the **CBOE Crude Oil Volatility Index (OVX)** — and oil market news sentiment — captured by the **Text Oil Sentiment Indicator (TOSI)** — can predict stock price volatility for major U.S. airlines (AAL, DAL, UAL, LUV) and the JETS ETF. We hypothesize that increases in OVX and negative shifts in TOSI lead to increased airline stock volatility, with the strongest predictive power at a 1-month lag.

---

## Datasets

| File | Description | Frequency |
|------|-------------|-----------|
| `data/OVXCLS.csv` | CBOE Crude Oil Volatility Index (OVX) closing values | Daily (2007–present) |
| `data/TOSI.csv` | Text Oil Sentiment Indicator — oil market news sentiment scores | Monthly (1982–present) |

Airline realized volatility is computed from daily price data fetched via `yfinance` for tickers **AAL, DAL, UAL, LUV**, and **JETS**.

---

## Notebook Structure

[Predicting_Airline_Stock_Volatility.ipynb](Predicting_Airline_Stock_Volatility.ipynb) is organized into six sections:

1. **Setup** — imports, shared color palette, plotting defaults
2. **Metrics & Data Preparation** — evaluation metrics, data loading, cleaning, and alignment of airline volatility, OVX, and TOSI
3. **Modeling Utilities** — MIDAS windows, monthly ML features, time-series cross-validation helpers
4. **Train & Evaluate Models** — three specifications per model family: OVX only, TOSI only, OVX + TOSI
   - MIDAS regression (rolling out-of-sample)
   - XGBoost & Random Forest (80/20 temporal split)
5. **Interactive Visual Design** — Plotly figures for data exploration and model comparison
6. **Results Summary** — performance tables and final figures

---

## Models

| Model | Specifications |
|-------|---------------|
| MIDAS Regression | OVX only · TOSI only · OVX + TOSI |
| XGBoost | OVX only · TOSI only · OVX + TOSI |
| Random Forest | OVX only · TOSI only · OVX + TOSI |

---

## Requirements

```
pandas
numpy
matplotlib
plotly
scikit-learn
xgboost
yfinance
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib plotly scikit-learn xgboost yfinance statsmodels
```

---

## Running the Notebook

```bash
jupyter notebook Predicting_Airline_Stock_Volatility.ipynb
```

Run all cells in order. The notebook fetches airline price data from Yahoo Finance on the fly, so an internet connection is required.
