import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime

# Internal Modules
from src.data_loader import fetch_api_data, fetch_gameweek_history
from src.features import process_data
from src.predictor import load_model, make_predictions

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Catalan FPL Predictor", 
    layout="wide", 
    page_icon="⚽"
)

# --- HEADER  ---
st.title("⚽ Catalan FPL Predictor")
st.markdown("**Created by Subanta Poudel** | *Hybrid Intelligence Expected Points (xP) Engine*")
st.markdown("---")

# --- DATA INGESTION LAYER ---
@st.cache_data(ttl=3600)
def load_app_data():
    raw_json = fetch_api_data()
    if not raw_json: return None, None, None
    model = load_model()
    return raw_json, model

raw_json, model = load_app_data()

if not raw_json or not model:
    st.error("🚨 Critical Error: API or Model missing.")
    st.stop()

# Pre-calculate Live Data
df_live, CURRENT_GW, IS_PRESEASON = process_data(raw_json, target_gw=None)
df_live = make_predictions(df_live, model, is_preseason=IS_PRESEASON)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Control Panel")
if IS_PRESEASON:
    st.sidebar.info("System Status: **Season Has Not Started**")
else:
    st.sidebar.info(f"System Status: **GW {CURRENT_GW} Active**")

try:
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
        st.sidebar.divider()
        st.sidebar.markdown("### 🧠 Model Diagnostics")
        st.sidebar.metric("RMSE (Error)", f"{metrics.get('rmse', 'N/A')}", delta_color="inverse")
        st.sidebar.metric("R² (Accuracy)", f"{metrics.get('r2', 'N/A')}")
        st.sidebar.caption(f"Model Version: {metrics.get('note', 'v1.0')}")
except FileNotFoundError:
    pass

# --- TABS ---
tab_pred, tab_info = st.tabs(["🔮 Predictions", "ℹ️ Documentation"])

# === TAB 1: LIVE PREDICTIONS ===
with tab_pred:
    if IS_PRESEASON:
        st.info("ℹ️ **Pre-Season Active:** Gameweek 1 has not commenced yet. Predictions are calculated using prior season baselines (with positional median imputation for new transfers).")
        st.subheader("🚀 Predicted Points for GW 1 (Pre-Season Baselines)")
    else:
        st.subheader(f"🚀 Predicted Points for GW {CURRENT_GW + 1}")
    
    # 1. Controls
    unique_teams = ["All"] + sorted([t for t in df_live['team_name'].dropna().unique()])
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: filter_pos = st.selectbox("Filter Position", ["All", "GKP", "DEF", "MID", "FWD"])
    with c2: filter_team = st.selectbox("Filter Team", unique_teams)
    with c3: search_query = st.text_input("Search Player")
    
    # 2. Filtering
    view_df = df_live.copy()
    if filter_pos != "All": view_df = view_df[view_df['position'] == filter_pos]
    if filter_team != "All": view_df = view_df[view_df['team_name'] == filter_team]
    if search_query: view_df = view_df[view_df['web_name'].str.contains(search_query, case=False)]
    
    # 3. Preparation for Display
    display_cols = {
        'web_name': 'Player', 
        'team_name': 'Team', 
        'next_opponent': 'Opponent',
        'position': 'Pos',
        'final_xp': 'Predicted Points', 
        'reasoning': 'Reasoning',
        'value': 'Price (£m)', 
        'next_match_difficulty': 'Diff (1-5)'
    }
    
    final_table = view_df[display_cols.keys()].rename(columns=display_cols).sort_values(by='Predicted Points', ascending=False).reset_index(drop=True)
    
    # 4. HIGHLIGHT TOP 5 PLAYERS (The Visual Fix)
    def highlight_top5(s):
        is_top5 = s.name < 5 # Since we reset index, top 5 are 0,1,2,3,4
        return ['background-color: #3e3216' if is_top5 else '' for _ in s] # Gold/Dark highlight
    
    st.dataframe(
        final_table.style.apply(highlight_top5, axis=1).format({"Predicted Points": "{:.1f}", "Price (£m)": "£{:.1f}"}),
        use_container_width=True,
        height=600
    )

# === TAB 2: BACKTESTING (DISABLED DUE TO DATA LEAKAGE) ===
# DISABLED: Passing today's live API snapshot (`raw_json`) into past gameweek calculations (`target_gw`)
# introduces severe data leakage by using future accumulated season stats to predict past outcomes.
# Real backtesting requires point-in-time historical data snapshots for each target gameweek.
# To prevent misleading accuracy metrics in the UI, this tab rendering has been disabled.

# === TAB 2: DOCUMENTATION ===
with tab_info:
    st.header("📘 System Documentation")
    
    st.subheader("1. Architecture")
    st.markdown("""
    The **Catalan AI Predictor** uses a hybrid architecture:
    * **Data Layer:** Live API ingestion from FPL endpoints.
    * **Model Layer:** Linear Regression trained on 2021–2025 historical data.
    * **Logic Layer:** Rule-based adjustments for 'Pep Roulette' (Rotation) and Fixture Difficulty.
    """)
    
    st.subheader("2. System Scope")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Capabilities")
        st.success("""
        * **Hybrid AI:** Combines Linear Regression with Expert Rules.
        * **Live Context:** Adjusts for Injuries, Fixture Difficulty, and Rotation Risks.
        * **Explainable Math:** Outputs step-by-step reasoning breakdown for every player.
        * **Bias Correction:** Includes a "Big Club Tax" for non-premium assets.
        """)
    with c2:
        st.subheader("❌ Limitations")
        st.error("""
        * **No Price Forecasting:** Uses current market values only.
        * **No Cup Fatigue:** Ignores non-PL competitions.
        * **Expert Heuristic Layer:** Quantitative tests indicate hand-crafted rules optimize Top-20 ranking utility while introducing slight point variance bias.
        """)

    st.subheader("3. Credits")
    st.text(f"Developed by Subanta Poudel\nLast Updated: {datetime.now().strftime('%Y-%m-%d')}")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #808495;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #262730;
    }
    </style>
    <div class="footer">
        © 2026 Catalan AI Predictor | Created by <b>Subanta Poudel</b> | Powered by Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)