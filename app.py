import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
from datetime import datetime

# Internal Modules
from src.data_loader import fetch_api_data, fetch_gameweek_history, fetch_player_live_summary
from src.features import process_data, fetch_full_history
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
st.caption("Project Established: October 2025 | Current Engine: Catalan AI Predictor v1.2")
st.markdown("---")

from pathlib import Path

import re

def sanitize_search_input(query):
    """
    Sanitizes user search inputs by removing non-alphanumeric characters
    (except spaces) to prevent regex code injection or crashes.
    """
    if not isinstance(query, str) or not query:
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()

# --- DATA INGESTION LAYER ---
@st.cache_data(ttl=3600)
def load_app_data():
    raw_json = fetch_api_data()
    if not raw_json: return None, None, None
    model = load_model()
    return raw_json, model

@st.cache_data(ttl=604800)
def load_historical_data(_model):
    df_hist = fetch_full_history()
    if df_hist.empty:
        return pd.DataFrame()
    features = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ]
    X = df_hist[features].fillna(0)
    df_hist['raw_xp'] = _model.predict(X)
    return df_hist

raw_json, model = load_app_data()

if not raw_json or not model:
    st.error("🚨 Critical Error: API or Model missing.")
    st.stop()

# Pre-calculate Live Data
df_live, CURRENT_GW, IS_PRESEASON = process_data(raw_json, target_gw=None)
df_live = make_predictions(df_live, model, is_preseason=IS_PRESEASON)

# --- REAL-TIME MATCHDAY INTELLIGENCE ---
if raw_json and 'static' in raw_json and 'events' in raw_json['static']:
    events_list = raw_json['static']['events']
    next_ev = next((e for e in events_list if e.get('is_next')), None)
    curr_ev = next((e for e in events_list if e.get('is_current')), None)
    target_ev = next_ev or curr_ev or (events_list[0] if events_list else None)

    if target_ev:
        gw_id = target_ev.get('id', 1)
        deadline_raw = target_ev.get('deadline_time')
        
        deadline_text = f"⏰ **Gameweek {gw_id}**"
        countdown_text = ""
        if deadline_raw:
            try:
                deadline_dt = pd.to_datetime(deadline_raw)
                now_utc = pd.Timestamp.now(tz='UTC')
                diff = deadline_dt - now_utc
                days = diff.days
                hours = diff.seconds // 3600
                mins = (diff.seconds % 3600) // 60
                
                deadline_fmt = deadline_dt.strftime('%a %d %b %H:%M')
                if diff.total_seconds() > 0:
                    countdown_str = f"in {days}d {hours}h" if days > 0 else f"in {hours}h {mins}m"
                else:
                    countdown_str = "PASSED / IN PROGRESS"
                
                deadline_text = f"⏰ **Gameweek {gw_id} Deadline:** {deadline_fmt} UTC"
                countdown_text = f"⏳ **Countdown:** {countdown_str}"
            except Exception:
                deadline_text = f"⏰ **Gameweek {gw_id} Active**"
                countdown_text = ""
        
        # 1. Live Deadline Banner
        st.info(f"{deadline_text} | {countdown_text}")

        # 2. Live Gameweek Fixtures Ribbon (Expandable)
        gw_fixes = [f for f in raw_json.get('fixtures', []) if f.get('event') == gw_id]
        if gw_fixes:
            teams_dict = {t['id']: t['short_name'] for t in raw_json['static'].get('teams', [])}
            with st.expander(f"⚽ Gameweek {gw_id} Live Matchday Fixtures ({len(gw_fixes)} Matches Scheduled)", expanded=False):
                fix_cols = st.columns(min(len(gw_fixes), 5))
                for idx, f in enumerate(gw_fixes):
                    h_team = teams_dict.get(f.get('team_h'), str(f.get('team_h')))
                    a_team = teams_dict.get(f.get('team_a'), str(f.get('team_a')))
                    started = f.get('started', False)
                    finished = f.get('finished', False)
                    h_score = f.get('team_h_score', 0)
                    a_score = f.get('team_a_score', 0)
                    ko_raw = f.get('kickoff_time')
                    ko_str = pd.to_datetime(ko_raw).strftime('%a %H:%M') if ko_raw else ""

                    if finished:
                        status_str = f"✅ {h_team} {h_score}-{a_score} {a_team}"
                        sub_str = "FT"
                    elif started:
                        status_str = f"🔴 {h_team} {h_score}-{a_score} {a_team}"
                        sub_str = "LIVE"
                    else:
                        status_str = f"📅 {h_team} vs {a_team}"
                        sub_str = ko_str

                    col_target = fix_cols[idx % 5]
                    with col_target:
                        st.caption(f"**{status_str}** | `{sub_str}`")

st.markdown("---")

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
        st.sidebar.caption("Project Established: October 2025\nCurrent Engine: Catalan AI Predictor v1.2")
except FileNotFoundError:
    st.sidebar.caption("Project Established: October 2025\nCurrent Engine: Catalan AI Predictor v1.2")

# --- TABS ---
tab_pred, tab_hist, tab_hauls, tab_info = st.tabs(["🔮 Predictions", "📊 Player History", "🏆 Hall of Fame", "ℹ️ Documentation"])

# === TAB 1: LIVE PREDICTIONS ===
with tab_pred:
    if IS_PRESEASON:
        st.info("ℹ️ **Pre-Season Active:** Gameweek 1 has not commenced yet. Predictions are calculated using prior season baselines (with positional median imputation for new transfers).")
        st.subheader("🚀 Predicted Points for GW 1 (Pre-Season Baselines)")
    else:
        st.subheader(f"🚀 Predicted Points for GW {CURRENT_GW}")
    
    # 1. Controls
    unique_teams = ["All"] + sorted([t for t in df_live['team_name'].dropna().unique()])
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: filter_pos = st.selectbox("Filter Position", ["All", "GKP", "DEF", "MID", "FWD"])
    with c2: filter_team = st.selectbox("Filter Team", unique_teams)
    with c3: search_query = st.text_input("Search Player")
    
    # 2. Filtering & Input Protection
    view_df = df_live.copy()
    if filter_pos != "All": view_df = view_df[view_df['position'] == filter_pos]
    if filter_team != "All": view_df = view_df[view_df['team_name'] == filter_team]
    
    clean_search = sanitize_search_input(search_query)
    if clean_search: 
        view_df = view_df[view_df['web_name'].str.contains(re.escape(clean_search), case=False, na=False)]
    
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

# === TAB 2: PLAYER HISTORY (ZERO DATA LEAKAGE) ===
with tab_hist:
    st.subheader("📊 Historical Player Performance Analysis")
    st.caption("Inspect past gameweek predictions (raw xP) vs actual points scored across historical and current seasons without data leakage.")

    df_hist_all = load_historical_data(model)

    if df_hist_all.empty:
        st.info("No historical Premier League data on record for this player.")
    else:
        unique_hist_players = sorted(df_hist_all['name'].dropna().unique())
        
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            selected_hist_player = st.selectbox(
                "Select Player for Historical Inspection",
                options=unique_hist_players,
                index=unique_hist_players.index("Bukayo Saka") if "Bukayo Saka" in unique_hist_players else 0,
                key="hist_player_select"
            )

        player_df = df_hist_all[df_hist_all['name'] == selected_hist_player].copy()

        # Check for live in-season played gameweek actuals via FPL API
        if 'df_live' in locals() and not df_live.empty:
            live_match = df_live[
                (df_live['web_name'].str.contains(re.escape(selected_hist_player), case=False, na=False)) |
                (df_live['first_name'].str.contains(re.escape(selected_hist_player), case=False, na=False)) |
                (df_live['second_name'].str.contains(re.escape(selected_hist_player), case=False, na=False))
            ]
            if not live_match.empty:
                live_p = live_match.iloc[0]
                summary = fetch_player_live_summary(int(live_p['id']))
                if summary and summary.get('history'):
                    live_hist_df = pd.DataFrame(summary['history'])
                    if not live_hist_df.empty:
                        curr_season_name = "2025-26"
                        live_hist_df['name'] = selected_hist_player
                        live_hist_df['season'] = curr_season_name
                        live_hist_df['GW'] = live_hist_df['round']
                        feat_cols = ['rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity', 'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value']
                        for col in ['minutes', 'ict_index', 'creativity', 'influence', 'threat', 'total_points', 'value']:
                            if col in live_hist_df.columns:
                                live_hist_df[col] = pd.to_numeric(live_hist_df[col], errors='coerce').fillna(0)
                        
                        cols_to_roll = {
                            'minutes': 'rolling_3_minutes',
                            'ict_index': 'rolling_3_ict_index',
                            'creativity': 'rolling_3_creativity',
                            'influence': 'rolling_3_influence',
                            'threat': 'rolling_3_threat',
                            'total_points': 'rolling_3_total_points'
                        }
                        for src_col, target_col in cols_to_roll.items():
                            live_hist_df[target_col] = (
                                live_hist_df[src_col]
                                .shift(1).rolling(3, min_periods=1).mean()
                                .fillna(0)
                            )
                        X_live = live_hist_df[feat_cols].fillna(0)
                        live_hist_df['raw_xp'] = model.predict(X_live)
                        
                        # Merge live current season rows with historical data
                        player_df = pd.concat([player_df[player_df['season'] != curr_season_name], live_hist_df], ignore_index=True)

        if player_df.empty:
            st.info("No historical Premier League data on record for this player.")
        else:
            seasons_available = sorted(player_df['season'].unique(), reverse=True)
            with col_p2:
                selected_season = st.selectbox(
                    "Select Season", 
                    options=seasons_available,
                    key="hist_season_select"
                )

            season_player_df = player_df[player_df['season'] == selected_season].sort_values('GW').copy()

            if season_player_df.empty:
                st.info("No historical Premier League data on record for this player in the selected season.")
            else:
                # 1. Multi-Line Chart (Gameweek vs Predicted xP and Actual Points)
                chart_df = season_player_df[['GW', 'raw_xp', 'total_points']].copy()
                chart_df = chart_df.rename(columns={
                    'raw_xp': 'Predicted xP',
                    'total_points': 'Actual Points'
                })
                chart_df['Gameweek'] = chart_df['GW'].astype(str)

                st.markdown(f"#### 📈 {selected_hist_player} — {selected_season} Gameweek Performance")
                
                fig = px.line(chart_df, x='Gameweek', y=['Predicted xP', 'Actual Points'], markers=True)
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=[str(i) for i in range(1, 39)], fixedrange=True)
                fig.update_yaxes(fixedrange=True)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # 2. Summary Table
                teams_map = {t['id']: t['short_name'] for t in raw_json['static']['teams']} if raw_json and 'static' in raw_json else {}
                
                def format_opp(row):
                    opp_id = row.get('opponent_team')
                    t_name = teams_map.get(opp_id, str(opp_id))
                    home_away = "H" if row.get('was_home') else "A"
                    return f"vs {t_name} ({home_away})"

                summary_display = season_player_df.copy()
                summary_display['Opponent'] = summary_display.apply(format_opp, axis=1)

                display_hist_cols = {
                    'season': 'Season',
                    'GW': 'Gameweek',
                    'Opponent': 'Opponent',
                    'minutes': 'Minutes',
                    'raw_xp': 'Predicted xP',
                    'total_points': 'Actual Points'
                }

                summary_table = (
                    summary_display[display_hist_cols.keys()]
                    .rename(columns=display_hist_cols)
                    .sort_values(by='Gameweek')
                    .reset_index(drop=True)
                )

                st.dataframe(
                    summary_table.style.format({
                        "Predicted xP": "{:.1f}",
                        "Actual Points": "{:d}"
                    }),
                    use_container_width=True
                )

# === TAB 3: HALL OF FAME (ALL-TIME FPL LEGENDS) ===
with tab_hauls:
    st.subheader("🏆 All-Time FPL Hall of Fame")
    st.caption("Curated records of the greatest individual career totals and single-season point hauls in Fantasy Premier League history.")

    record_view = st.radio(
        "Select Record View",
        ["Career Records", "Single-Season Records"],
        horizontal=True,
        key="fame_record_view"
    )

    if record_view == "Career Records":
        career_data = [
            {"Rank": 1, "Player": "Wayne Rooney", "Career FPL Points": 2338, "Primary Club": "Manchester United", "Active Era": "2002–2018"},
            {"Rank": 2, "Player": "Frank Lampard", "Career FPL Points": 2318, "Primary Club": "Chelsea", "Active Era": "2002–2015"},
            {"Rank": 3, "Player": "Mohamed Salah", "Career FPL Points": 2100, "Primary Club": "Liverpool", "Active Era": "2017–Present"},
            {"Rank": 4, "Player": "Steven Gerrard", "Career FPL Points": 2044, "Primary Club": "Liverpool", "Active Era": "2002–2015"},
            {"Rank": 5, "Player": "Petr Čech", "Career FPL Points": 1908, "Primary Club": "Chelsea / Arsenal", "Active Era": "2004–2019"}
        ]
        df_career = pd.DataFrame(career_data)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("👑 All-Time Career Leader", "2,338 pts", "Wayne Rooney (Man Utd)")
        with m2:
            st.metric("⚡ Active Career Leader", "2,100 pts", "Mohamed Salah (Liverpool)")
        with m3:
            st.metric("⭐ 2,000 Points Club", "4 Legends", "Rooney, Lampard, Salah, Gerrard")

        st.markdown("---")
        st.dataframe(
            df_career.style.format({"Career FPL Points": "{:,d}"}),
            use_container_width=True,
            height=300
        )
    else:
        single_season_data = [
            {"Rank": 1, "Player": "Mohamed Salah", "Season": "2024-25", "Points": 306, "Club": "Liverpool"},
            {"Rank": 2, "Player": "Mohamed Salah", "Season": "2017-18", "Points": 303, "Club": "Liverpool"},
            {"Rank": 3, "Player": "Luis Suárez", "Season": "2013-14", "Points": 295, "Club": "Liverpool"},
            {"Rank": 4, "Player": "Frank Lampard", "Season": "2009-10", "Points": 284, "Club": "Chelsea"},
            {"Rank": 5, "Player": "Cristiano Ronaldo", "Season": "2007-08", "Points": 283, "Club": "Manchester United"},
            {"Rank": 6, "Player": "Erling Haaland", "Season": "2022-23", "Points": 272, "Club": "Manchester City"}
        ]
        df_single = pd.DataFrame(single_season_data)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🔥 Single-Season Record", "306 pts", "Mohamed Salah (2024-25)")
        with m2:
            st.metric("👑 300+ Points Club", "2 Seasons", "Mohamed Salah (17/18 & 24/25)")
        with m3:
            st.metric("⭐ Top 6 Record Average", "290.5 pts", "All-Time Elite Performances")

        st.markdown("---")
        st.dataframe(
            df_single.style.format({"Points": "{:d}"}),
            use_container_width=True,
            height=320
        )

    st.caption("ℹ️ Note: Historical records pre-dating 2016 are statically curated.")

# === TAB 4: DOCUMENTATION ===
with tab_info:
    st.header("📘 System Documentation")
    
    st.subheader("1. Architecture")
    st.markdown("""
    The **Catalan AI Predictor v1.2** uses a hybrid architecture:
    * **Data Layer:** Live API ingestion from FPL endpoints.
    * **Model Layer:** Linear Regression trained on 2021–2026 historical data.
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

    st.subheader("3. Credits & Timeline")
    st.text(f"Developed by Subanta Poudel\nProject Established: October 2025\nCurrent Engine: Catalan AI Predictor v1.2\nLast Updated: {datetime.now().strftime('%Y-%m-%d')}")

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
        © 2026 Catalan AI Predictor v1.2 | Created by <b>Subanta Poudel</b> | Project Established: October 2025
    </div>
    """,
    unsafe_allow_html=True
)