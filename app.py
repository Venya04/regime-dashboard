import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os

def get_file_version(path):
    try:
        return os.path.getmtime(path)
    except:
        return None


# === SETTINGS ===
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
STABLECOIN_MONTHLY_YIELD = 0.05 / 12
CASH_DAILY_YIELD = 0.045 / 252

TICKERS = {
    "stocks": "SPY",
    "crypto": "BTC-USD",
    "commodities": "GLD",
    "cash": None,
    "stablecoins": None
}

st.set_page_config(page_title="Regime Report", layout="wide")

# === LOAD DATA ===
@st.cache_data
def load_csv_from_repo(path, version=None):
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

regime_df = load_csv_from_repo("regime_labels_expanded.csv", version=get_file_version("regime_labels_expanded.csv"))
opt_alloc_df = load_csv_from_repo("optimal_allocations.csv", version=get_file_version("optimal_allocations.csv"))


@st.cache_data
def load_prices():
    data = {}
    for asset, ticker in TICKERS.items():
        if ticker:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            # Extract price series
            if "Adj Close" in df.columns:
                series = df["Adj Close"]
            elif "Close" in df.columns:
                series = df["Close"]
            else:
                continue  # Skip if no usable price column

            # Ensure it's a DataFrame with the correct column name
            df_clean = pd.DataFrame(series)
            df_clean.columns = [asset]

            data[asset] = df_clean

    if not data:
        return pd.DataFrame()

    prices = pd.concat(data.values(), axis=1)
    return prices.dropna()

prices = load_prices()

# === VALIDATE DATA ===
if prices.empty:
    st.error("Price data failed to load.")
    st.stop()

if regime_df.empty or "regime" not in regime_df.columns:
    st.error("Regime data missing or malformed.")
    st.stop()

# === PREPARE DATA ===
regime_df.set_index("date", inplace=True)
regime_df = regime_df.asfreq("D").ffill().reindex(prices.index, method="ffill")
regime_df["regime"] = regime_df["regime"].str.capitalize()

allocations = opt_alloc_df.set_index("regime").to_dict(orient="index")
for alloc in allocations.values():
    if "cash" not in alloc:
        alloc["cash"] = 0.1
    total = sum(alloc.values())
    for k in alloc:
        alloc[k] /= total

# === RETURNS ===
returns = prices.pct_change().dropna()
returns["cash"] = CASH_DAILY_YIELD
returns["stablecoins"] = (1 + STABLECOIN_MONTHLY_YIELD)**(1/22) - 1

all_assets = set()
for alloc in allocations.values():
    all_assets.update(alloc.keys())

for asset in all_assets:
    if asset not in returns.columns:
        returns[asset] = 0.0

# === BACKTEST ===
def backtest(returns, regime_df, allocations):
    portfolio_returns = []
    current_weights = {asset: 0.25 for asset in TICKERS}
    prev_regime = None
    for date in returns.index:
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            portfolio_returns.append(np.nan)
            continue
        if regime != prev_regime and regime in allocations:
            current_weights = allocations[regime]
            prev_regime = regime
        ret = sum(returns.loc[date, asset] * current_weights.get(asset, 0) for asset in current_weights)
        portfolio_returns.append(ret)
    return pd.Series(portfolio_returns, index=returns.index)

portfolio_returns = backtest(returns, regime_df, allocations)

# === GET CURRENT REGIME ===
latest_date = regime_df.index[-1]
current_regime = regime_df.loc[latest_date, "regime"]
current_alloc = allocations.get(current_regime, {})

# === HEADER ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');
    .gothic-title {
        font-family: 'UnifrakturCook', serif;
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        padding: 0.5rem 0;
        letter-spacing: 1px;
    }
    .pub-info {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 0.8rem;
        margin-top: -18px;
        color: #ccc;
    }
    </style>
    <div class='gothic-title'>The Regime Report</div>
    <div class='pub-info'>No. 01 · Published biWeekly · Market Bulletin · June 2025</div>
    <h3 style='text-align: center; font-family: Georgia, serif; font-style: italic; margin-top: -10px;'>
        Asset Allocation in Current Market Conditions
    </h3>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container {
            padding-left: 5vw;
            padding-right: 5vw;
        }
        .section-title {
            font-family: Georgia, serif;
            font-size: 1.1rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 6px;
            color: #d4af37;
            border-bottom: 1px solid #555;
            padding-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# === LAYOUT ===
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.markdown("""
        <style>
            .left-section-title {
                font-family: Georgia, serif;
                font-size: 1.1rem;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 10px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    if current_alloc:
        # Filter out allocations smaller than 0.1%
        filtered_alloc = {k: v for k, v in current_alloc.items() if v > 0.001}

        if filtered_alloc:
            fig_pie = px.pie(
                names=list(filtered_alloc.keys()),
                values=list(filtered_alloc.values()),
                hole=0,
                color=list(filtered_alloc.keys()),
                color_discrete_map={
                    "stocks": "#102030",
                    "stablecoins": "#3A3A3A",
                    "cash": "#5C5149",
                    "crypto": "#2F4F4F",
                    "commodities": "#6B4E23",
                }
            )

            fig_pie.update_traces(
                textinfo='percent',
                textfont_size=16,
                pull=[0.03] * len(filtered_alloc),
                marker=dict(line=dict(color="#000000", width=2))
            )

            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div class='left-section-title'>Portfolio Holdings</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; margin-top: -5px;'>
            <ul style='padding-left: 10; list-style-position: inside; text-align: left; display: inline-block;'>
        """ + "".join([
            f"<li><strong>{asset.capitalize()}</strong>: {weight:.1%}</li>"
            for asset, weight in current_alloc.items()
        ]) + """
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


with right_col:
    st.markdown("""
        <style>
            .section-title {
                font-family: Georgia, serif;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 6px;
                text-align: left;
                color: white;
                border-bottom: 1px solid #555;
                padding-bottom: 4px;
            }
            .section-comment {
                font-family: Georgia, serif;
                font-size: 0.9rem;
                font-style: italic;
                color: #ccc;
                margin-top: 4px;
                margin-bottom: 8px;
            }
            @media (max-width: 768px) {
                .section-title {
                    font-size: 14px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    for title, placeholder in [
        ("Market Insight", "What are we seeing in the macro environment?"),
        ("Top Strategy Note", "Thoughts on the market (e.g., technical signals)"),
        ("Trader's Conclusion", "Summary and suggested action")
    ]:
        cols = st.columns([0.6, 0.1])
        with cols[0]:
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.text_area(placeholder, height=130)
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

def compute_metrics(returns):
    import numpy as np
    cumulative_return = (1 + returns).prod() - 1
    annualized_volatility = returns.std() * (252 ** 0.5)
    sharpe_ratio = (returns.mean() * 252) / annualized_volatility
    return {
        "Cumulative Return": cumulative_return,
        "Annual Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio
    }


metrics = compute_metrics(portfolio_returns.dropna())
st.subheader("📊 Performance Metrics")
st.write(pd.DataFrame(metrics, index=["Value"]).T.style.format("{:.2%}"))
