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
END_DATE = "2025-04-17"
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
# Normalize regime labels to prevent mismatch
regime_df["regime"] = regime_df["regime"].astype(str).str.strip().str.lower()
opt_alloc_df["regime"] = opt_alloc_df["regime"].astype(str).str.strip().str.lower()
# Group by regime, take mean if duplicates exist
allocations = (
    opt_alloc_df.groupby("regime")
    .mean(numeric_only=True)
    .apply(lambda row: row / row.sum(), axis=1)  # Normalize
    .to_dict(orient="index")
)

# st.write("✅ Keys in allocations dict:", list(allocations.keys()))

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
# === GET CURRENT REGIME ===
# Load raw regime file and normalize
original_regime_dates = pd.read_csv("regime_labels_expanded.csv", parse_dates=["date"])
original_regime_dates["regime"] = original_regime_dates["regime"].astype(str).str.strip().str.lower()

# Get latest labeled regime (not forward-filled)
latest_regime = original_regime_dates.dropna(subset=["regime"]).iloc[-1]["regime"]

# Debug
# st.write("🧠 Detected Regime:", latest_regime)
# st.write("📊 Available Allocation Regimes:", list(allocations.keys()))

# Use regime-specific allocation or fallback
if latest_regime not in allocations:
    st.warning(f"⚠️ No allocation found for regime: '{latest_regime}'. Falling back to 'recovery'.")
    current_alloc = allocations.get("recovery", {})
else:
    current_alloc = allocations[latest_regime]

# Confirm what allocation will be shown
# st.write("✅ Current allocation weights being used:")
# st.write(current_alloc)


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
                    "stocks": "#242039",
                    "stablecoins": "#5b5271",
                    "cash": "#484564",
                    "crypto": "#2a2b47",
                    "commodities": "#76570A",
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
    background-color: #262730;  /* 🎯 matches st.text_area theme */
    padding: 10px;
    border-radius: 5px;
    min-height: 160px;
    border: 1px solid #444; /* optional: matches input box border */
}
    @media (max-width: 768px) {
        .section-title {
            font-size: 14px;
        }
    }
</style>

    """, unsafe_allow_html=True)

    import json
    NOTES_FILE = "thoughts.txt"

    default_sections = {
        "Market Insight": "",
        "Top Strategy Note": "",
        "Trader's Conclusion": ""
    }

    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as f:
            json.dump(default_sections, f)

    try:
        with open(NOTES_FILE, "r") as f:
            commentary = json.load(f)
    except Exception:
        commentary = default_sections

    query_params = st.query_params
    is_admin_mode = query_params.get("admin", "false").lower() == "true"

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if is_admin_mode and not st.session_state.auth:
        with st.expander("🔒 Admin Login (edit mode)", expanded=False):
            pwd = st.text_input("Enter password", type="password")
            if pwd == st.secrets["auth"]["edit_password"]:
                st.session_state.auth = True
                st.success("Edit mode activated!")

    for section_title in commentary:
        cols = st.columns([0.6, 0.1])
        with cols[0]:
            st.markdown(f"<div class='section-title'>{section_title}</div>", unsafe_allow_html=True)
            if st.session_state.auth:
                commentary[section_title] = st.text_area(
                    f"{section_title} input",
                    value=commentary[section_title],
                    height=130,
                    key=section_title
                )
            else:
                content = commentary[section_title].strip() or "..."
                st.markdown(f"<div class='section-comment'>{content}</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # Save only if edited
    if st.session_state.auth:
        with open(NOTES_FILE, "w") as f:
            json.dump(commentary, f)
# Hide Streamlit menu and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

