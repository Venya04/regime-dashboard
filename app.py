import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import colorsys 


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

# === 1. Page Config & Session State
st.set_page_config(page_title="Regime Report", layout="wide")
query_params = st.query_params
admin_param = query_params.get("admin", "false")
if isinstance(admin_param, list):
    admin_value = admin_param[0]
else:
    admin_value = admin_param

is_admin_mode = admin_value.lower() == "true"
# st.write("query_params:", query_params)
# st.write("admin_value:", admin_value)
# st.write("is_admin_mode:", is_admin_mode)
if "show_guide" not in st.session_state:
    st.session_state["show_guide"] = False

# === 2. Floating "User Guide" Button (bottom-left)
if not st.session_state["show_guide"]:
    # Display button
    if st.button("📘 User Guide", key="guide_btn"):
        st.session_state["show_guide"] = True
    # Float with CSS
    st.markdown("""
    <style>
        button[kind="secondary"] {
            position: fixed !important;
            bottom: 15px;
            left: 15px;
            z-index: 9999;
            background-color: rgba(255,255,255,0.08);
            color: #eee;
            border: none;
            padding: 6px 12px;
            font-size: 13px;
            border-radius: 6px;
            cursor: pointer;
        }
        button[kind="secondary"]:hover {
            background-color: rgba(255,255,255,0.18);
        }
    </style>
    """, unsafe_allow_html=True)

# === 3. Hide the floating button when guide is open
if st.session_state["show_guide"]:
    st.markdown("""
    <style>
        button[kind="secondary"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

# === 4. Show the guide content when open
if st.session_state["show_guide"]:
    with st.container():
        st.markdown("""
        ### 📘 How to Read This Dashboard

        This dashboard is built around a core truth:

        > 💭 **You can’t control the market — but you can control your response.**

        Markets are shaped by forces beyond our control — macro shifts, geopolitical events, and investor behavior.  
        What we **can** do is manage risk, adjust wisely, and respond rationally.

        This dashboard helps you do exactly that.

        ---

        We analyze the current **economic regime** using macro indicators (like GDP growth, inflation, and interest rates) and deliver:
        - Optimized asset allocation  
        - Market and strategy insights
        - Data-informed actions to take (or avoid)  

        ---

        ### 🥧 Portfolio Allocation Pie Chart  
        A breakdown of how to allocate assets — **stocks, crypto, commodities, cash, and stablecoins** — based on the current regime.

        ### 🧭 Macro Outlook  
        Clear, up-to-date interpretation of economic trends: **growth, inflation and interest rates**.

        ### 🧮 Portfolio Positioning  
        Explains how the current macro backdrop shapes our **asset allocation** strategy — what we favor, avoid, and why.

        ### 🎯 Tactical Moves  
        A trader’s view on what we’re doing right now — **holding, hedging, rotating, or waiting** — with links to relevant charts.
        
        #### 📈 Portfolio Performance Chart  
        Shows how the strategy performed over time

        ---

        ### 🧪 How This Works

        This dashboard uses a combination of **macroeconomic data** and **machine learning clustering** to detect the current economic regime.

        We analyze:
        - **GDP Growth** – is the economy expanding or contracting?
        - **Inflation Trends** – are prices rising or stabilizing?
        - **Interest Rate & Liquidity Signals** – are conditions tightening or easing?
        - **PCA & K-Means Clustering** – we reduce dimensional noise and group macro patterns into clear regimes.

        From this, we identify 4 main regimes:
        - 🔥 **Overheating** – fast growth + rising inflation
        - 📈 **Recovery** – improving growth + low inflation
        - 🧊 **Contraction** – slowing growth + falling demand
        - ⚠️ **Stagflation** – weak growth + high inflation

        Each regime has a different impact on asset classes. For example:
        - **Overheating** → favors real assets (like commodities), reduces exposure to high-growth stocks.
        - **Contraction** → shifts toward cash and stable income (bonds/stablecoins).
        - **Recovery** → leans into equities and risk assets as optimism returns.
        - **Stagflation** → protects capital in inflation-proof stores like commodities, while trimming risk assets.

        We then calculate **optimal portfolio allocations** based on historical performance within each regime, using backtesting and machine learning optimization.

        **In short:** Macro signals define the regime → regime defines expected asset behavior → we align allocations accordingly.

        ### 💬 Still Learning?  
        No worries — this dashboard is designed to be educational and actionable.  
        Think of it as your **macro compass** — helping you navigate instead of guess.

        > **Discipline over desire always wins.**
        """, unsafe_allow_html=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        # ✅ Close guide form button
        with st.form("close_guide_form"):
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("❌ Close Guide")
            st.markdown("</div>", unsafe_allow_html=True)

            if submitted:
                st.session_state["show_guide"] = False
                st.rerun()

    # st.write("Query params at top:", query_params)
    # st.write("is_admin_mode at top:", is_admin_mode)

    # 🛑 Stop dashboard rendering when guide is shown
    st.stop()
    
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

@st.cache_data
def load_performance():
    try:
        perf_df = pd.read_csv("portfolio_performance.csv", parse_dates=["date"])
        perf_df.set_index("date", inplace=True)
        return perf_df
    except Exception as e:
        st.error(f"Failed to load performance data: {e}")
        return pd.DataFrame()


performance_df = load_performance()

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
returns["stablecoins"] = (1 + STABLECOIN_MONTHLY_YIELD) ** (1 / 22) - 1

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
        text-align: center;
        margin-bottom: 0.2rem;
        margin-top: -130px;
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

st.markdown("""
    <style>
        .portfolio-list, .portfolio-list li {
            font-family: Georgia, serif !important;
        font-size: 0.96rem;
        font-style: italic;
        color: #ccc;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 0.3em;
    }
    .portfolio-list li strong {
        font-style: normal;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# === LAYOUT ===
performance_df = load_performance()
perf_df = performance_df.reset_index() if not performance_df.empty else pd.DataFrame()

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

    # Limit left column content width
    st.markdown("<div style='max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)

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
                    "stocks": "#19212E",
                    "stablecoins": "#522D2D",
                    "cash": "#391514",
                    "crypto": "#212D40",
                    "commodities": "#6d5332",
                }
            )

            fig_pie.update_traces(
                textinfo='percent',
                textfont=dict(size=17, family="Georgia"),
                # insidetextorientation='radial',
                pull=[0.01] * len(filtered_alloc),
                marker=dict(line=dict(color="#000000", width=1))
            )

            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # 🔽 Portfolio Holdings
st.markdown("<div class='left-section-title'>💼 Portfolio Holdings</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; margin-top: -5px;'>
        <ul class='portfolio-list' style='padding-left: 10; list-style-position: inside; text-align: left; display: inline-block;'>
    """ + "".join([
        f"<li><strong>{asset.capitalize()}</strong>: {weight:.1%}</li>"
        for asset, weight in current_alloc.items()
    ]) + """
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

    # 🔽 Performance Summary (Matches Portfolio Holdings Header Style)
    if not performance_df.empty:
        start_val = performance_df["value"].iloc[0]
        end_val = performance_df["value"].iloc[-1]
        perf_pct = ((end_val / start_val) - 1) * 100

        st.markdown(
            f"""
            <div class='left-section-title' style='margin-bottom: 8px; margin-top: 8px;'>
                <span style='vertical-align: middle; font-size: 1.5rem;'>📈</span>
                <span style='vertical-align: middle;'>Performance: {perf_pct:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Graph
    if not perf_df.empty:
        import streamlit.components.v1 as components  # place at top of file ideally!
        perf_fig = px.line(
            perf_df,
            x="date",
            y="value",
            labels={"value": "Portfolio Value", "date": "Date"},
            template="plotly_dark",
            markers=True,
            color_discrete_sequence=["#e85d04"]
        )
        perf_fig.update_traces(line=dict(width=3), marker=dict(size=6))
        perf_fig.update_layout(
            height=510,
            width=600,
            margin=dict(l=20, r=20, t=10, b=20),
            autosize=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        html = perf_fig.to_html(include_plotlyjs='cdn', full_html=False)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)  # small spacer
        components.html(f"""
        <div style="display: flex; justify-content: center;">
            <div style="max-width: 600px; width: 100%;">
                {html}
            </div>
        </div>
        """, height=510)

with right_col:
    st.markdown("""
       <style>
    .section-title {
        font-family: Georgia, serif;
        font-size: 18px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 4px;
        text-align: left;
        color: white;
        border-bottom: 1px solid #555;
        padding-bottom: 6px;
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
    margin-bottom: 6px; /* <-- add this */
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
        "🧭 Macro Outlook": "",
        "🧮 Portfolio Positioning": "",
        "🎯 Tactical Moves": ""
    }

    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as f:
            json.dump(default_sections, f)

    try:
        with open(NOTES_FILE, "r") as f:
            commentary = json.load(f)
    except Exception:
        commentary = default_sections

    if "auth" not in st.session_state:
        st.session_state.auth = False

    # Login form only if needed
    if is_admin_mode and not st.session_state.auth:
        with st.expander("🔒 Admin Login (edit mode)", expanded=False):
            pwd = st.text_input("Enter password", type="password")
            if pwd == st.secrets["auth"]["edit_password"]:
                st.session_state.auth = True
                st.success("Edit mode activated!")

    # Commentary boxes, always rendered here
    for section_title in commentary:
        cols = st.columns([0.6, 0.1])
        with cols[0]:
            st.markdown(f"<div class='section-title'>{section_title}</div>", unsafe_allow_html=True)
            if is_admin_mode and st.session_state.auth:
                commentary[section_title] = st.text_area(
                    f"{section_title} input",
                    value=commentary[section_title],
                    height=100,
                    key=section_title
                )
            else:
                content = commentary[section_title].strip() or "..."
                st.markdown(f"<div class='section-comment'>{content}</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Save only if edited
    if is_admin_mode and st.session_state.auth:
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






