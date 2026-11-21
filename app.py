import streamlit as st
import pandas as pd
import json
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path

# Internal Modules
from src.data_loader import (
    TZ_MAP,
    fetch_api_data,
    fetch_league_squads,
    fetch_league_standings,
    fetch_player_live_summary,
    fetch_user_team,
)
from src.features import process_data, fetch_full_history
from src.gameweek_ui import (
    render_captaincy_card,
    render_gameweek_fixtures,
    render_gameweek_header,
    render_theme,
    render_wordmark,
)
from src.predictor import load_active_model, make_predictions
from src.projections import build_projection_csv
from src.projections_ui import render_player_projections

PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)

def render_html(html):
    """
    Renders a raw HTML/CSS block via st.markdown(unsafe_allow_html=True) safely.

    Streamlit's markdown parser runs CommonMark first: any line indented 4+
    spaces is treated as a preformatted code block and printed as literal
    text instead of being parsed as HTML. Python f-strings built inside
    indented functions inherit that source indentation, which is exactly
    what caused the raw CSS/HTML to leak onto the page as visible text.
    Stripping leading whitespace per line (line breaks don't affect HTML
    rendering) removes the whole class of bug at a single call site.
    """
    st.markdown("\n".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)

def sanitize_search_input(query):
    """
    Sanitizes user search inputs by removing non-alphanumeric characters
    (except spaces) to prevent regex code injection or crashes.
    """
    if not isinstance(query, str) or not query:
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()

# --- DATA INGESTION LAYER ---
# Not itself cached: fetch_api_data() combines fetch_static_data() (ttl=3600)
# and fetch_fixtures_data() (ttl=60) from src/data_loader.py, each cached
# independently. Wrapping this call in another st.cache_data would gate
# both behind whichever TTL was picked here, defeating the fixtures' 60s
# freshness -- see src/data_loader.py::fetch_api_data for the same note.
@st.cache_resource
def load_ml_model():
    return load_active_model()

def load_app_data():
    try:
        raw_json = fetch_api_data()
        if not raw_json or not raw_json.get("static", {}).get("elements"):
            return None
        model = load_ml_model()
        if model is None:
            return None
        frame, gameweek, is_preseason = process_data(raw_json, target_gw=None)
        predictions = make_predictions(frame, model, is_preseason=is_preseason)
    except Exception:
        LOGGER.exception("Failed to load the Gameweek forecast")
        return None
    return raw_json, model, predictions, gameweek, is_preseason

@st.cache_data(ttl=604800)
def load_historical_data(_model):
    """
    'Raw xP' only (no rule layer, no captaincy scoring) across the full
    multi-season historical pipeline for the Player History tab.
    Deliberately not make_predictions(): its rule layer needs live-only
    columns (chance_of_playing_next_round, etc.) this historical data
    doesn't carry, and computing final_xp/reasoning/ceiling_score for
    100k+ historical rows this tab never displays would be wasted work.

    `_model` is whichever bundle src/config.py::USE_COMPONENT_MODEL
    currently selects (see src/predictor.py::load_active_model) -- this
    used to assume a bare sklearn model with .predict(), a leftover from
    before the per-position-ensemble migration that broke silently the
    first time this tab was actually opened after that migration shipped.
    Routes by model shape the same way evaluation/backtest.py's own two
    backtest functions do, so this can't drift from what predictor.py
    actually serves.
    """
    df_hist = fetch_full_history()
    if df_hist.empty:
        return pd.DataFrame()
    df_hist['position'] = df_hist['position'].replace({'GK': 'GKP'})

    if 'positions' in _model:
        features = _model.get('features', [
            'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
            'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
        ])
        df_hist = df_hist.dropna(subset=features).reset_index(drop=True)
        df_hist['raw_xp'] = 0.0
        for position, ensemble in _model['positions'].items():
            mask = df_hist['position'] == position
            if not mask.any():
                continue
            X = df_hist.loc[mask, features].fillna(0)
            df_hist.loc[mask, 'raw_xp'] = ensemble.predict(X)
    else:
        from src.components import predict_all_components
        df_hist = df_hist.reset_index(drop=True)
        df_hist['raw_xp'] = predict_all_components(df_hist, _model)['raw_xp'].values

    return df_hist

# --- HELPER: SEASON POINTS EXPLORER (LIVE API ONLY, NO LOCAL STORAGE) ---
def render_season_points_explorer(df_live, raw_json):
    """
    Browse this season's top scorers and drill into any one player's
    gameweek-by-gameweek breakdown. Runs entirely on the live API --
    bootstrap-static/ for the leaderboard (already loaded into df_live),
    element-summary/{id}/ for the per-player history -- so unlike the local-
    snapshot recap this replaced, there is nothing cached to local disk and
    nothing lost on a Streamlit Cloud redeploy.
    """
    st.markdown("### Season points")
    st.caption("Top 50 players by total points this season. Search or filter, then pick one below for their full gameweek-by-gameweek breakdown.")

    c1, c2 = st.columns([2, 1])
    with c1:
        search_query = st.text_input("Search Player", key="spe_search")
    with c2:
        filter_pos = st.selectbox("Filter Position", ["All", "GKP", "DEF", "MID", "FWD"], key="spe_pos")

    top50 = df_live.sort_values(by='total_points', ascending=False).head(50).copy()

    if filter_pos != "All":
        top50 = top50[top50['position'] == filter_pos]

    clean_search = sanitize_search_input(search_query)
    if clean_search:
        top50 = top50[top50['web_name'].str.contains(re.escape(clean_search), case=False, na=False)]

    if top50.empty:
        st.info("No players in the top 50 match that search/filter.")
        return

    leaderboard_cols = {
        'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
        'total_points': 'Pts', 'minutes': 'Mins', 'goals_scored': 'G',
        'assists': 'A', 'bonus': 'Bonus', 'value': 'Price'
    }
    leaderboard = (
        top50[leaderboard_cols.keys()].rename(columns=leaderboard_cols)
        .sort_values(by='Pts', ascending=False).reset_index(drop=True)
    )

    st.dataframe(
        leaderboard.style.set_properties(**{'color': '#EFF4EE'}).format({"Price": "£{:.1f}"}),
        width="stretch",
        height=400,
        hide_index=True
    )

    st.markdown("#### Player gameweek breakdown")
    ranked_top50 = top50.sort_values(by='total_points', ascending=False)
    player_options = ranked_top50['web_name'].tolist()
    selected_player_name = st.selectbox(
        "Select a player from the table above", options=player_options, key="spe_player_select"
    )

    selected_row = ranked_top50[ranked_top50['web_name'] == selected_player_name].iloc[0]
    player_id = int(selected_row['id'])

    summary = fetch_player_live_summary(player_id)
    history = summary.get('history', []) if summary else []

    if not history:
        st.info(f"ℹ️ **No gameweek data yet for {selected_player_name}** — they haven't featured in a live gameweek this season.")
        return

    hist_df = pd.DataFrame(history).sort_values(by='round').reset_index(drop=True)
    total_minutes = int(pd.to_numeric(hist_df['minutes'], errors='coerce').fillna(0).sum())

    if total_minutes == 0:
        st.info(f"ℹ️ **{selected_player_name} hasn't played any minutes yet this season** — nothing to chart.")
        return

    season_points = int(selected_row['total_points'])
    pp90 = season_points / (total_minutes / 90.0)
    best_row = hist_df.sort_values(by='total_points', ascending=False).iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Season Points", f"{season_points}")
    with m2:
        st.metric("Total Minutes", f"{total_minutes}")
    with m3:
        st.metric("Points / 90", f"{pp90:.2f}")
    with m4:
        st.metric("Best Gameweek", f"GW {int(best_row['round'])} ({int(best_row['total_points'])} pts)")

    hist_df['GW'] = hist_df['round'].astype(str)
    fig = go.Figure(go.Bar(x=hist_df['GW'], y=hist_df['total_points'], marker_color='#DAF995'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#EFF4EE',
        xaxis_title='Gameweek',
        yaxis_title='Points',
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

    teams_map = {t['id']: t['short_name'] for t in raw_json['static'].get('teams', [])} if raw_json and 'static' in raw_json else {}

    def format_opp(row):
        t_name = teams_map.get(row.get('opponent_team'), str(row.get('opponent_team')))
        home_away = "H" if row.get('was_home') else "A"
        return f"{t_name} ({home_away})"

    breakdown = hist_df.copy()
    breakdown['Opponent'] = breakdown.apply(format_opp, axis=1)
    breakdown_cols = {
        'round': 'GW', 'Opponent': 'Opponent', 'minutes': 'Mins',
        'goals_scored': 'G', 'assists': 'A', 'clean_sheets': 'CS',
        'bonus': 'Bonus', 'total_points': 'Pts'
    }
    breakdown_table = breakdown[breakdown_cols.keys()].rename(columns=breakdown_cols)

    st.dataframe(
        breakdown_table.style.set_properties(**{'color': '#EFF4EE'}),
        width="stretch",
        height=400,
        hide_index=True
    )
    st.markdown("---")

# --- HELPER: AI TEAM OF THE WEEK (FOOTBALL PITCH VISUALIZER) ---
def select_team_of_the_week(df_candidates):
    """
    Selects optimal starting 11 for the upcoming gameweek obeying FPL formation rules:
    - Exactly 1 GKP
    - 3 to 5 DEF
    - 2 to 5 MID
    - 1 to 3 FWD
    - Total = 11, Max 3 players per team
    Assigns Captain (C) to highest xP player, Vice-Captain (V) to 2nd highest xP player.
    """
    if df_candidates is None or df_candidates.empty:
        return None, None
        
    candidates = df_candidates[df_candidates['final_xp'] > 0].sort_values(by='final_xp', ascending=False).reset_index(drop=True)
    if candidates.empty:
        return None, None
        
    gkp_cand = candidates[candidates['position'] == 'GKP']
    def_cand = candidates[candidates['position'] == 'DEF']
    mid_cand = candidates[candidates['position'] == 'MID']
    fwd_cand = candidates[candidates['position'] == 'FWD']
    
    if gkp_cand.empty:
        return None, None
        
    valid_formations = [
        (3, 5, 2), (3, 4, 3), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3)
    ]

    def pick_within_club_cap(pools, quotas, max_per_club=3):
        """
        Fills each position quota greedily by xP, but tracks a running
        per-club count *across all positions in this attempt* and skips
        any player who would push their club past the cap -- continuing
        down that position's candidate list instead. The previous version
        sliced a fixed top-N per position independently, so if one club
        (e.g. a favourable single-fixture gameweek) dominated the top of
        several positions at once, every formation ended up with 4+ of
        that club and got discarded outright with nothing to show.
        Returns (rows, club_counts) or (None, club_counts) if any quota
        can't be filled without breaching the cap.
        """
        rows = []
        club_counts = {}
        for pos, need in quotas.items():
            pool = pools[pos]
            taken = 0
            for _, row in pool.iterrows():
                if taken >= need:
                    break
                club = row['team_name']
                if club_counts.get(club, 0) >= max_per_club:
                    continue
                rows.append(row)
                club_counts[club] = club_counts.get(club, 0) + 1
                taken += 1
            if taken < need:
                return None, club_counts
        return rows, club_counts

    pools = {'GKP': gkp_cand, 'DEF': def_cand, 'MID': mid_cand, 'FWD': fwd_cand}

    best_squad = None
    best_xp = -1.0
    best_formation = None

    for (n_def, n_mid, n_fwd) in valid_formations:
        if len(gkp_cand) < 1 or len(def_cand) < n_def or len(mid_cand) < n_mid or len(fwd_cand) < n_fwd:
            continue

        quotas = {'GKP': 1, 'DEF': n_def, 'MID': n_mid, 'FWD': n_fwd}
        rows, _ = pick_within_club_cap(pools, quotas)
        if rows is None:
            continue

        squad = pd.DataFrame(rows).reset_index(drop=True)
        tot_xp = squad['final_xp'].sum()
        if tot_xp > best_xp:
            best_xp = tot_xp
            best_squad = squad
            best_formation = f"{n_def}-{n_mid}-{n_fwd}"

    if best_squad is None or best_squad.empty:
        return None, None
        
    sorted_squad = best_squad.sort_values(by='final_xp', ascending=False).reset_index(drop=True)
    captain_name = sorted_squad.iloc[0]['web_name'] if len(sorted_squad) > 0 else ""
    vice_name = sorted_squad.iloc[1]['web_name'] if len(sorted_squad) > 1 else ""
    
    return best_squad, {
        'formation': best_formation,
        'total_xp': best_xp,
        'captain': captain_name,
        'vice': vice_name
    }

def render_pitch_visualizer(totw_df, totw_info, gameweek):
    """
    Renders an interactive green HTML/CSS Football Pitch visualizer displaying the starting XI.
    """
    if totw_df is None or totw_df.empty:
        return
        
    formation = totw_info.get('formation', '3-5-2')
    total_xp = totw_info.get('total_xp', 0.0)
    captain_name = totw_info.get('captain', '')
    vice_name = totw_info.get('vice', '')
    
    fwds = totw_df[totw_df['position'] == 'FWD']
    mids = totw_df[totw_df['position'] == 'MID']
    defs = totw_df[totw_df['position'] == 'DEF']
    gkps = totw_df[totw_df['position'] == 'GKP']

    def player_card_html(row):
        name = row['web_name']
        team = row['team_name']
        opp = row.get('next_opponent', '-')
        xp = row['final_xp']
        
        badge_html = ""
        if name == captain_name:
            badge_html = '<span style="background:#EDCD94; color:#1A1408; font-weight:bold; padding:1px 4px; border-radius:3px; font-size:10px; margin-left:3px;">(C)</span>'
        elif name == vice_name:
            badge_html = '<span style="background:#C0C0C0; color:#1A1408; font-weight:bold; padding:1px 4px; border-radius:3px; font-size:10px; margin-left:3px;">(V)</span>'

        return f'''
        <div style="background: #1D2821; border: 1px solid #303C33; border-radius: 8px; padding: 6px 4px; text-align: center; flex: 1 1 70px; max-width: 95px; min-width: 65px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); box-sizing: border-box;">
            <div style="font-weight: 700; font-size: 12px; color: #EFF4EE; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{name}{badge_html}</div>
            <div style="font-size: 10px; color: #A9B9AA; margin-top: 2px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{team} · {opp}</div>
            <div style="font-family: var(--font-mono); font-size: 12px; font-weight: 700; color: #DAF995; margin-top: 4px;">{xp:.1f} xP</div>
        </div>
        '''

    def row_html(df_group):
        cards = "".join([player_card_html(r) for _, r in df_group.iterrows()])
        return f'<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px; margin: 10px 0;">{cards}</div>'

    fwd_row = row_html(fwds)
    mid_row = row_html(mids)
    def_row = row_html(defs)
    gkp_row = row_html(gkps)

    pitch_html = f'''
    <div style="background: linear-gradient(180deg, #1D2821 0%, #101613 100%); border: 2px solid #DAF995; border-radius: 12px; padding: 18px 12px; margin-bottom: 25px; box-shadow: inset 0 0 40px rgba(0,0,0,0.6); position: relative;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 8px; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;">
            <div style="font-family: var(--font-display); font-weight: 700; font-size: 18px; color: #EDCD94;">⭐ AI Team of the Week — GW {gameweek}</div>
            <div style="font-size: 13px; color: #EFF4EE;">Formation: <strong>{formation}</strong> | Squad xP: <strong style="font-family: var(--font-mono); color:#EDCD94;">{total_xp:.1f} pts</strong></div>
        </div>
        {fwd_row}
        {mid_row}
        {def_row}
        {gkp_row}
    </div>
    '''
    render_html(pitch_html)

# --- HELPER: 5-GAMEWEEK FIXTURE TICKER ---
def render_fixture_ticker(raw_json, current_gw):
    """
    Renders an interactive matrix table for all 20 Premier League teams showing their
    next 5 opponents (GW+1 to GW+5) with FPL FDR color coding.
    """
    if not raw_json or 'fixtures' not in raw_json or 'static' not in raw_json:
        return
        
    fixtures = raw_json.get('fixtures', [])
    teams = raw_json['static'].get('teams', [])
    teams_map = {t['id']: t['short_name'] for t in teams}

    gws = [current_gw + i for i in range(5)]
    gw_cols = [f"GW {gw}" for gw in gws]
    
    ticker_data = []
    
    for t_id, t_name in teams_map.items():
        row = {'Team': t_name}
        total_fdr = 0
        cnt = 0
        
        for i, gw in enumerate(gws):
            col_name = f"GW {gw}"
            fix = next((f for f in fixtures if f.get('event') == gw and (f.get('team_h') == t_id or f.get('team_a') == t_id)), None)
            if fix:
                is_home = (fix.get('team_h') == t_id)
                opp_id = fix.get('team_a') if is_home else fix.get('team_h')
                fdr = fix.get('team_h_difficulty') if is_home else fix.get('team_a_difficulty')
                opp_name = teams_map.get(opp_id, str(opp_id))
                venue = "H" if is_home else "A"
                cell_text = f"{opp_name} ({venue})"
                total_fdr += fdr
                cnt += 1
            else:
                cell_text = "BLANK"
                total_fdr += 3
                cnt += 1
                
            row[col_name] = cell_text
            
        row['Avg FDR'] = round(total_fdr / max(cnt, 1), 2)
        ticker_data.append(row)

    df_ticker = pd.DataFrame(ticker_data).sort_values(by='Avg FDR', ascending=True).reset_index(drop=True)
    
    fdr_lookup = {}
    for t_id, t_name in teams_map.items():
        for gw in gws:
            fix = next((f for f in fixtures if f.get('event') == gw and (f.get('team_h') == t_id or f.get('team_a') == t_id)), None)
            if fix:
                is_home = (fix.get('team_h') == t_id)
                fdr = fix.get('team_h_difficulty') if is_home else fix.get('team_a_difficulty')
            else:
                fdr = 3
            fdr_lookup[(t_name, f"GW {gw}")] = fdr

    def style_fdr_cells(val, team_name, col_name):
        fdr = fdr_lookup.get((team_name, col_name), 3)
        if fdr <= 2:
            return 'background-color: #006437; color: #FFFFFF; font-weight: bold;'
        elif fdr == 3:
            return 'background-color: #37003C; color: #FFFFFF; font-weight: bold;'
        elif fdr == 4:
            return 'background-color: #7F0038; color: #FFFFFF; font-weight: bold;'
        else:
            return 'background-color: #FF005A; color: #FFFFFF; font-weight: bold;'

    def apply_ticker_styling(df_in):
        style_df = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
        for idx, r in df_in.iterrows():
            t_name = r['Team']
            style_df.loc[idx, 'Team'] = 'color: #EFF4EE; font-weight: bold;'
            style_df.loc[idx, 'Avg FDR'] = 'color: #EDCD94; font-weight: bold;'
            for col in gw_cols:
                val = r[col]
                style_df.loc[idx, col] = style_fdr_cells(val, t_name, col)
        return style_df

    st.markdown("### Fixture difficulty by club")
    st.caption("Interactive 5-week schedule matrix for all 20 Premier League clubs color-coded by Fixture Difficulty Rating (FDR). Sorted by overall easiest schedule.")
    
    st.dataframe(
        df_ticker.style.apply(apply_ticker_styling, axis=None).format({"Avg FDR": "{:.2f}"}),
        width="stretch",
        height=400,
        hide_index=True
    )

# --- HELPER: BENCH OPTIMIZATION & TRANSFER RECOMMENDATION ---
def find_best_valid_bench_swap(starters, bench):
    """
    Finds bench-to-starter swap with maximum positive xP gain complying with FPL rules:
    - Bench GKP only compared against Starting GKP.
    - Bench Outfield players (DEF, MID, FWD) only compared against Starting Outfield players.
    - Resulting formation must be valid (at least 3 DEF, at least 2 MID, at least 1 FWD).
    """
    best_swap = None
    best_gain = 0.0

    if starters.empty or bench.empty:
        return None

    # 1. Evaluate Goalkeeper Swap
    bench_gkp = bench[bench['position_pred'] == 'GKP']
    starting_gkp = starters[starters['position_pred'] == 'GKP']
    if not bench_gkp.empty and not starting_gkp.empty:
        b_gkp = bench_gkp.iloc[0]
        s_gkp = starting_gkp.iloc[0]
        gkp_gain = float(b_gkp.get('final_xp', 0.0)) - float(s_gkp.get('final_xp', 0.0))
        if gkp_gain > 0:
            best_gain = gkp_gain
            best_swap = (b_gkp, s_gkp, gkp_gain)

    # 2. Evaluate Outfield Swaps
    bench_outfield = bench[bench['position_pred'] != 'GKP']
    starting_outfield = starters[starters['position_pred'] != 'GKP']

    def_cnt = len(starters[starters['position_pred'] == 'DEF'])
    mid_cnt = len(starters[starters['position_pred'] == 'MID'])
    fwd_cnt = len(starters[starters['position_pred'] == 'FWD'])

    for _, b_player in bench_outfield.iterrows():
        b_pos = b_player.get('position_pred')
        b_xp = float(b_player.get('final_xp', 0.0))
        if b_pos not in ['DEF', 'MID', 'FWD']:
            continue

        for _, s_player in starting_outfield.iterrows():
            s_pos = s_player.get('position_pred')
            s_xp = float(s_player.get('final_xp', 0.0))
            if s_pos not in ['DEF', 'MID', 'FWD']:
                continue

            gain = b_xp - s_xp
            if gain <= best_gain:
                continue

            # Calculate formation after swap
            new_def = def_cnt - (1 if s_pos == 'DEF' else 0) + (1 if b_pos == 'DEF' else 0)
            new_mid = mid_cnt - (1 if s_pos == 'MID' else 0) + (1 if b_pos == 'MID' else 0)
            new_fwd = fwd_cnt - (1 if s_pos == 'FWD' else 0) + (1 if b_pos == 'FWD' else 0)

            if new_def >= 3 and new_mid >= 2 and new_fwd >= 1:
                best_gain = gain
                best_swap = (b_player, s_player, gain)

    return best_swap


def recommend_gameweek_transfer(starters, squad_df, df_live):
    """
    Identifies primary sell candidate (injured starting outfielder or lowest xP starting outfielder),
    searches league for highest xP buy target in same position within Sell Price + 0.5m budget.
    Filters out players already in squad and enforces 3-players-per-club limit.
    """
    outfield_starters = starters[starters['position_pred'] != 'GKP']
    if outfield_starters.empty:
        return None

    def is_injured(row):
        st = str(row.get('status', 'a')).lower()
        ch_raw = row.get('chance_of_playing_next_round', row.get('chance_of_playing', 100))
        try:
            ch = float(ch_raw) if pd.notna(ch_raw) else 100.0
        except (ValueError, TypeError):
            ch = 100.0
        return st in ['i', 's', 'u'] or ch == 0 or float(row.get('final_xp', 0.0)) == 0.0

    injured_starters = outfield_starters[outfield_starters.apply(is_injured, axis=1)]
    if not injured_starters.empty:
        sell_candidate = injured_starters.sort_values(by='final_xp', ascending=True).iloc[0]
    else:
        sell_candidate = outfield_starters.sort_values(by='final_xp', ascending=True).iloc[0]

    sell_pos = sell_candidate.get('position_pred')
    sell_xp = float(sell_candidate.get('final_xp', 0.0))
    sell_val_raw = float(sell_candidate.get('val_m', sell_candidate.get('value', 0.0)))
    sell_price = sell_val_raw / 10.0 if sell_val_raw > 20 else sell_val_raw
    sell_club = sell_candidate.get('team_name')

    max_budget = sell_price + 0.5

    squad_ids = set(squad_df['element'].dropna().astype(int).tolist()) if 'element' in squad_df.columns else set()
    squad_club_counts = squad_df['team_name'].value_counts().to_dict()

    candidates = df_live[
        (df_live['position'] == sell_pos) &
        (~df_live['id'].isin(squad_ids)) &
        (df_live['final_xp'] > sell_xp)
    ].copy()

    if candidates.empty:
        return {
            'sell_candidate': sell_candidate,
            'buy_target': None,
            'blocked_info': None
        }

    def parse_price(row):
        v = float(row.get('val_m', row.get('value', 0.0)))
        return v / 10.0 if v > 20 else v

    candidates['price_m'] = candidates.apply(parse_price, axis=1)
    valid_budget_candidates = candidates[candidates['price_m'] <= max_budget].sort_values(by='final_xp', ascending=False)

    buy_target = None
    blocked_info = None

    for _, cand in valid_budget_candidates.iterrows():
        cand_club = cand.get('team_name')
        curr_count = squad_club_counts.get(cand_club, 0)
        effective_count = curr_count - (1 if sell_club == cand_club else 0)

        if effective_count >= 3:
            if blocked_info is None:
                blocked_info = {
                    'blocked_player': cand.get('web_name'),
                    'club': cand_club
                }
            continue
        else:
            buy_target = cand
            break

    return {
        'sell_candidate': sell_candidate,
        'buy_target': buy_target,
        'blocked_info': blocked_info
    }


MAX_FPL_ID = 2**53 - 1  # Largest integer Streamlit can round-trip through the browser.


def render_fpl_id_form(param, label, button_label):
    query_id = st.query_params.get(param, "")
    default_id = None
    if query_id.isascii() and query_id.isdecimal() and len(query_id) <= len(str(MAX_FPL_ID)):
        parsed_id = int(query_id)
        if 1 <= parsed_id <= MAX_FPL_ID:
            default_id = parsed_id

    with st.container(key="deadline_entry"):
        with st.form(f"{param}_form", border=False):
            entered_id = st.number_input(
                label, min_value=1, max_value=MAX_FPL_ID, value=default_id,
                step=1, key=f"{param}_input",
            )
            submitted = st.form_submit_button(button_label, type="primary")

    if submitted and entered_id is not None:
        st.query_params[param] = str(entered_id)
        return entered_id
    if submitted or (query_id and default_id is None):
        st.warning(f"Enter a valid {label} to continue.")
        return None
    return default_id


def render_rate_my_team_section(df_live, current_gw):
    """
    Renders the interactive "Rate My Team" squad evaluation card.
    Pulls user's 15 FPL picks via Team ID, calculates total starting XI xP,
    weakest starter link, bench optimization alert, recommended transfer plan card, and displays a full squad table.
    """
    st.caption("Find your Team ID in your FPL URL (`/entry/123456/event/...`). The loaded ID is saved in this page’s URL.")
    team_id = render_fpl_id_form("team_id", "FPL Team ID", "Evaluate squad")

    if team_id:
        team_data = fetch_user_team(int(team_id), current_gw)
        if not team_data or not team_data.get('picks'):
            st.warning(f"Could not load squad picks for Team ID `{team_id}` in Gameweek {current_gw}. Check the ID in your FPL URL and try again.")
            return

        picks_df = pd.DataFrame(team_data['picks'])
        gw_used = team_data.get('gw', current_gw)
        if gw_used != current_gw:
            st.caption(f"Published picks source: Gameweek {gw_used}; forecasts shown for Gameweek {current_gw}.")
        
        merged = pd.merge(picks_df, df_live, left_on='element', right_on='id', how='left', suffixes=('_pick', '_pred'))
        
        if merged.empty:
            st.error("Could not match squad players with model prediction dataset.")
            return

        starters = merged[merged['position_pick'] <= 11].copy()
        bench = merged[merged['position_pick'] > 11].copy()

        starters['calc_xp'] = starters['final_xp'] * starters['multiplier']
        total_starting_xp = starters['calc_xp'].sum()

        weakest_starter = starters.sort_values(by='final_xp', ascending=True).iloc[0]
        w_name = weakest_starter.get('web_name', 'Unknown')
        w_xp = float(weakest_starter.get('final_xp', 0.0))
        w_opp = weakest_starter.get('next_opponent', '-')

        best_swap = find_best_valid_bench_swap(starters, bench)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label=f"🚀 Total Projected Starting XI (GW {current_gw})",
                value=f"{total_starting_xp:.1f} pts",
                delta=f"GW {gw_used} Squad Import"
            )
        with m2:
            st.metric(
                label="⚠️ Weakest Link (Starter)",
                value=f"{w_name} ({w_xp:.1f} xP)",
                delta=f"vs {w_opp}"
            )
        with m3:
            if best_swap is not None:
                rec_bench, rec_starter, rec_gain = best_swap
                b_name = rec_bench.get('web_name', '')
                s_name = rec_starter.get('web_name', '')
                s_xp = float(rec_starter.get('final_xp', 0.0))

                st.metric(
                    label="💡 Bench Optimization Alert",
                    value=f"{b_name} (+{rec_gain:.1f} xP)",
                    delta=f"Bench {s_name} ({s_xp:.1f} xP)",
                    delta_color="normal"
                )
            else:
                st.metric(
                    label="✅ Bench Optimization",
                    value="Starting XI Optimal",
                    delta="No Bench Swaps Needed"
                )

        if best_swap is not None:
            rec_bench, rec_starter, rec_gain = best_swap
            b_name = rec_bench.get('web_name', '')
            b_xp = float(rec_bench.get('final_xp', 0.0))
            b_pos = rec_bench.get('position_pred', '')
            s_name = rec_starter.get('web_name', '')
            s_xp = float(rec_starter.get('final_xp', 0.0))
            s_pos = rec_starter.get('position_pred', '')
            st.warning(f"⚠️ **Bench Swap Recommended:** Bench player **{b_name}** ({b_pos}, {b_xp:.1f} xP) has higher predicted points than starting **{s_name}** ({s_pos}, {s_xp:.1f} xP)! Consider starting {b_name} instead.")

        # --- RECOMMENDATION CARD: 🎯 Recommended Gameweek Transfer ---
        transfer_res = recommend_gameweek_transfer(starters, merged, df_live)
        if transfer_res and transfer_res.get('sell_candidate') is not None:
            sell_c = transfer_res['sell_candidate']
            buy_t = transfer_res['buy_target']
            blocked_inf = transfer_res['blocked_info']
            
            sell_n = sell_c.get('web_name', '')
            sell_xp_val = float(sell_c.get('final_xp', 0.0))
            
            if buy_t is not None:
                buy_n = buy_t.get('web_name', '')
                buy_xp_val = float(buy_t.get('final_xp', 0.0))
                net_gain = buy_xp_val - sell_xp_val
                
                if blocked_inf is not None:
                    st.info(f"ℹ️ Club Limit: 3 players already owned from {blocked_inf['club']}. Targeting alternate option: {buy_n}")
                    
                render_html(f"""
                <div style="background: #1D2821; border: 1px solid #303C33; border-radius: 10px; padding: 14px 18px; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 17px; color: #EDCD94; margin-bottom: 8px;">🎯 Recommended Gameweek Transfer</div>
                    <div style="font-size: 14px; color: #EFF4EE; display: flex; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #7f1d1d; color: #f87171; font-weight: bold; padding: 3px 8px; border-radius: 4px;">[SELL]</span>
                        <strong>{sell_n}</strong> <span style="font-family: var(--font-mono);">({sell_xp_val:.1f} xP)</span>
                        <span style="font-size: 16px; color: #EDCD94;">➔</span>
                        <span style="background: #064e3b; color: #34d399; font-weight: bold; padding: 3px 8px; border-radius: 4px;">[BUY]</span>
                        <strong>{buy_n}</strong> <span style="font-family: var(--font-mono);">({buy_xp_val:.1f} xP)</span>
                        <span style="color: #A9B9AA; margin: 0 4px;">|</span>
                        <strong>Net Projected Gain: <span style="font-family: var(--font-mono); color: #4CBE86; font-size: 15px;">+{net_gain:.1f} xP</span></strong>
                    </div>
                </div>
                """)
            else:
                st.info("✅ **Squad Transfer Status:** No transfer recommended this week. Your starting players are already optimal for their budget range.")

        st.markdown("#### 📋 Squad Breakdown & Captain Multipliers")
        
        def format_role(row):
            mult = row.get('multiplier', 1)
            is_c = row.get('is_captain', False)
            is_vc = row.get('is_vice_captain', False)
            pos = row.get('position_pick', 1)
            
            if is_c or mult >= 2:
                return f"Captain ({mult}×)"
            elif is_vc:
                return "Vice-captain"
            elif pos <= 11:
                return "Starter"
            else:
                return "Bench"

        merged['Role'] = merged.apply(format_role, axis=1)
        merged['Projected Points'] = merged['final_xp'] * merged['multiplier']

        display_rmt_cols = {
            'position_pick': 'Slot',
            'web_name': 'Player',
            'team_name': 'Team',
            'position_pred': 'Pos',
            'next_opponent': 'Opponent',
            'Role': 'Role',
            'final_xp': 'Adjusted xP',
            'Projected Points': 'Effective xP'
        }

        rmt_table = merged[display_rmt_cols.keys()].rename(columns=display_rmt_cols).sort_values(by='Slot').reset_index(drop=True)

        def style_rmt_rows(s):
            is_starter = s['Slot'] <= 11
            is_c = s['Role'].startswith("Captain")
            if is_c:
                return ['background-color: #2b2410; color: #EFF4EE; font-weight: bold;' for _ in s]
            elif is_starter:
                return ['color: #EFF4EE;' for _ in s]
            else:
                return ['background-color: #171F1B; color: #EFF4EE;' for _ in s]

        st.dataframe(
            rmt_table.style.apply(style_rmt_rows, axis=1).format({"Adjusted xP": "{:.1f}", "Effective xP": "{:.1f}"}),
            width="stretch",
            height=420,
            hide_index=True
        )

# --- HELPER: ACCURACY (PUBLIC PREDICTED-VS-ACTUAL RECORD) ---
# Reuses evaluation/backtest.py's own scoring function directly rather than
# re-implementing RMSE/bucket logic here -- this page can never silently
# report a flattered number the CLI backtest wouldn't also produce, because
# it's the exact same function. Recomputed fresh from the historical CSV
# pipeline (source B in CLAUDE.md) on every cache refresh -- no local
# persistence, per the project's deployment constraint (Streamlit Community
# Cloud's free tier wipes the filesystem on every redeploy).
ACCURACY_SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22"]


@st.cache_data(ttl=3600)
def _compute_accuracy_backtest(season):
    from evaluation.backtest import run_position_ensemble_backtest
    return run_position_ensemble_backtest(season)


def render_accuracy_page():
    """
    The public accuracy record: predicted vs. actual points, tracked across
    every gameweek of a completed season, shown openly -- including the
    weeks the model got it wrong. Deliberately evaluates raw_xp (the
    model's own prediction), not final_xp (the rule-adjusted number shown
    elsewhere in the app) -- the same scoping evaluation/backtest.py's
    module docstring documents: final_xp's historical rule-layer context
    (what fixture difficulty/rotation risk/DEFCON rate genuinely were for
    a specific past gameweek) isn't reconstructable, so this reports the
    number that actually can be honestly measured, labeled as exactly
    that, rather than a flattering number this project can't verify.
    """
    st.markdown("### Season backtest")
    st.caption(
        "Predicted vs. actual points, tracked across every gameweek of a "
        "completed season — shown openly, including the weeks the model "
        "got it wrong. This evaluates **raw_xp**, the model's own raw "
        "prediction before the rule layer adjusts it to the `final_xp` "
        "shown elsewhere in the app — the same honest scope "
        "`evaluation/backtest.py` uses, so this page can never show a "
        "number the project's own CLI backtest wouldn't also report."
    )

    season = st.selectbox("Season", ACCURACY_SEASONS, index=0, key="accuracy_season")

    try:
        with st.spinner(f"Backtesting the production model against every gameweek of {season}..."):
            results, season_df = _compute_accuracy_backtest(season)
    except Exception as e:
        st.warning(f"⚠️ Could not run the accuracy backtest for {season} right now: {e}")
        return

    overall = results["overall"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Gameweeks", results["n_gameweeks"])
    with m2:
        st.metric("RMSE", f"{overall['rmse']:.2f}")
    with m3:
        st.metric("MAE", f"{overall['mae']:.2f}")
    with m4:
        sp = results.get("spearman_rank_correlation_mean_per_gw")
        st.metric("Rank Correlation", f"{sp:.2f}" if sp is not None else "n/a")

    # --- Per-gameweek RMSE/MAE trend -- the actual "tracked across
    # gameweeks" record, not just one season-wide number. ---
    gw_stats = (
        season_df.groupby("GW")
        .apply(lambda g: pd.Series({
            "RMSE": float(np.sqrt(((g["predicted"] - g["actual"]) ** 2).mean())),
            "MAE": float((g["predicted"] - g["actual"]).abs().mean()),
        }))
        .reset_index()
        .sort_values("GW")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gw_stats["GW"], y=gw_stats["RMSE"], mode="lines+markers", name="RMSE", line=dict(color="#DAF995")))
    fig.add_trace(go.Scatter(x=gw_stats["GW"], y=gw_stats["MAE"], mode="lines+markers", name="MAE", line=dict(color="#EDCD94")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EFF4EE",
        xaxis_title="Gameweek",
        yaxis_title="Error (points)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # --- By outcome bucket -- the same Zeros/Blanks/Tickers/Haulers split
    # evaluation/backtest.py reports, so a visitor sees the model is judged
    # on more than one flattering global number. ---
    st.markdown("#### By outcome bucket")
    st.caption("Zeros: didn't play. Blanks: played, ≤2 pts. Tickers: 3–4 pts. Haulers: 5+ pts.")
    bucket_df = pd.DataFrame(results["by_bucket"]).rename(
        columns={"bucket": "Bucket", "n": "N", "rmse": "RMSE", "mae": "MAE"}
    )
    st.dataframe(
        bucket_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        width="stretch", hide_index=True,
    )

    # --- Biggest misses, most recent gameweek in this season -- the
    # "including the weeks it got things wrong" part, named explicitly
    # rather than left implicit in an aggregate error number. ---
    st.markdown("#### Biggest misses — most recent gameweek in this season")
    latest_gw = int(season_df["GW"].max())
    latest = season_df[season_df["GW"] == latest_gw].copy()
    latest["miss"] = latest["predicted"] - latest["actual"]
    top_misses = latest.reindex(latest["miss"].abs().sort_values(ascending=False).index).head(10)
    disp = top_misses[["name", "position", "predicted", "actual", "miss"]].rename(
        columns={"name": "Player", "position": "Pos", "predicted": "Predicted", "actual": "Actual", "miss": "Predicted − Actual"}
    )
    st.caption(f"Gameweek {latest_gw}")
    st.dataframe(
        disp.style.format({"Predicted": "{:.1f}", "Actual": "{:.0f}", "Predicted − Actual": "{:+.1f}"}),
        width="stretch", hide_index=True,
    )


# --- HELPER: MINI-LEAGUE (GROUP VIEW) ---
# Squad-level detail (projected points, captains, ownership) is capped to
# this many entries by rank -- fetch_league_standings can return up to 500
# (10 pages), but fetching every one of those managers' picks one at a
# time (politely, ~0.15s apart -- see fetch_entry_picks) would take
# minutes for a league that size. A mini-league among friends is almost
# always well under this; the standings table itself still shows everyone
# retrieved.
MAX_LEAGUE_ENTRIES_FOR_SQUADS = 60


def render_mini_league_section(df_live, current_gw):
    """
    Renders the Mini-League tab: standings, each manager's projected
    points and captain for the upcoming gameweek, effective ownership
    within the league (weighted by captaincy, not global "selected by"),
    and a differentials view.
    """
    st.caption(
        "Compare standings, projected points, captain picks and effective ownership in a classic FPL league. "
        "Find the league ID in its FPL URL (`/leagues/123456/standings/c`)."
    )
    league_id = render_fpl_id_form("league_id", "FPL League ID", "Load league")
    if not league_id:
        return

    with st.spinner(f"Fetching standings for league {int(league_id)}..."):
        league_data = fetch_league_standings(int(league_id))

    if league_data is None:
        st.warning(
            f"⚠️ Could not load league `{int(league_id)}`. It may be **private** "
            "(FPL's public API can only read leagues that allow it), the ID may be "
            "wrong, or the FPL API may be temporarily unavailable. Double-check the "
            "ID from the league's own page URL and try again."
        )
        return

    standings = league_data['standings']
    if not standings:
        st.info("This league has no entries yet.")
        return

    league_name = league_data['league'].get('name', 'Mini-League')
    st.markdown(f"#### {league_name}")
    if league_data.get('truncated'):
        st.info(f"League has more than {len(standings)} entries -- showing the first {len(standings)} by rank.")

    entry_ids = [e['entry'] for e in standings]
    fetch_entries = entry_ids[:MAX_LEAGUE_ENTRIES_FOR_SQUADS]
    if len(entry_ids) > MAX_LEAGUE_ENTRIES_FOR_SQUADS:
        st.caption(
            f"Squad-level detail (projected points, captains, ownership) limited to "
            f"the top {MAX_LEAGUE_ENTRIES_FOR_SQUADS} entries by rank to keep load "
            f"times reasonable -- the standings table below still covers everyone "
            f"retrieved."
        )

    with st.spinner(f"Fetching {len(fetch_entries)} managers' squads for GW {current_gw} (paced to respect FPL's rate limits, may take a moment)..."):
        squads = fetch_league_squads(fetch_entries, current_gw)

    id_to_web_name = df_live.set_index('id')['web_name'].to_dict()
    id_to_xp = df_live.set_index('id')['final_xp'].to_dict()

    manager_rows = []
    ownership_count = {}
    ownership_weight = {}
    n_managers_with_data = 0

    for entry in standings:
        eid = entry['entry']
        picks_data = squads.get(eid)
        row = {
            'Rank': entry.get('rank'),
            'Manager': entry.get('player_name'),
            'Team': entry.get('entry_name'),
            'Total Pts': entry.get('total'),
        }
        if picks_data is None:
            row['Projected GW Pts'] = None
            row['Captain'] = 'Unavailable'
            manager_rows.append(row)
            continue

        n_managers_with_data += 1
        proj = 0.0
        captain_name = 'N/A'
        for p in picks_data.get('picks', []):
            el = p.get('element')
            mult = p.get('multiplier', 0) or 0
            proj += id_to_xp.get(el, 0.0) * mult
            ownership_count[el] = ownership_count.get(el, 0) + 1
            ownership_weight[el] = ownership_weight.get(el, 0) + mult
            if p.get('is_captain'):
                captain_name = id_to_web_name.get(el, f"#{el}")

        row['Projected GW Pts'] = round(proj, 1)
        row['Captain'] = captain_name
        manager_rows.append(row)

    standings_df = pd.DataFrame(manager_rows)
    st.dataframe(standings_df, width="stretch", hide_index=True, height=420)

    if n_managers_with_data == 0:
        st.info(
            "Could not retrieve squad picks for any manager in this league for "
            f"GW {current_gw} yet -- picks are usually visible once a manager has "
            "set their team for the gameweek; try again closer to the deadline."
        )
        return

    # Effective ownership across the FULL predicted player pool, not just
    # players someone in the league happens to own -- a player 0% of the
    # league has is exactly the differential case that matters most, and
    # it would silently disappear from a table built only from owned
    # players.
    eo_df = df_live[['id', 'web_name', 'team_name', 'position', 'final_xp']].copy()
    eo_df['Ownership %'] = (eo_df['id'].map(ownership_count).fillna(0) * 100 / n_managers_with_data).round(1)
    eo_df['Effective Ownership %'] = (eo_df['id'].map(ownership_weight).fillna(0) * 100 / n_managers_with_data).round(1)
    eo_df['Projected xP'] = eo_df['final_xp'].round(1)
    eo_df = eo_df.rename(columns={'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos'})

    st.markdown("#### 👥 Effective Ownership")
    st.caption(
        "Weights each owning manager by their multiplier this gameweek (captain = "
        "x2, triple captain = x3, benched = x0) -- this is what actually moves you "
        "up or down **this league**, not FPL's global \"selected by\" percentage. A "
        "player can exceed 100% effective ownership if enough of the league has "
        "them captained."
    )
    owned = eo_df[eo_df['Ownership %'] > 0].sort_values('Effective Ownership %', ascending=False)
    st.dataframe(
        owned[['Player', 'Team', 'Pos', 'Ownership %', 'Effective Ownership %', 'Projected xP']].head(25),
        width="stretch", hide_index=True
    )

    st.markdown("#### 💎 Differentials")
    st.caption("Players projected well for the upcoming gameweek that fewer than 30% of this league owns.")
    diffs = eo_df[eo_df['Ownership %'] < 30].sort_values('Projected xP', ascending=False)
    if diffs.empty or diffs['Projected xP'].max() <= 0:
        st.info("No standout differentials found for this gameweek.")
    else:
        st.dataframe(
            diffs[['Player', 'Team', 'Pos', 'Ownership %', 'Projected xP']].head(20),
            width="stretch", hide_index=True
        )

def render_player_history(df_live, raw_json, model):
    st.subheader("Historical performance")
    st.caption("Compare past base xP forecasts with actual points. Forecasts use data from earlier gameweeks.")

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

        # Add published current-season results using the same positional model.
        season_start = next(
            (event.get("deadline_time") for event in raw_json["static"].get("events", []) if event["id"] == 1),
            None,
        )
        first_deadline = pd.to_datetime(season_start, utc=True, errors="coerce")
        if pd.notna(first_deadline) and "positions" in model:
            full_names = df_live['first_name'].fillna('') + ' ' + df_live['second_name'].fillna('')
            player_name = selected_hist_player.casefold()
            live_match = df_live[
                full_names.str.casefold().eq(player_name) | df_live['web_name'].str.casefold().eq(player_name)
            ]
            if len(live_match) == 1:
                live_p = live_match.iloc[0]
                ensemble = model['positions'].get(live_p['position'])
                summary = fetch_player_live_summary(int(live_p['id']))
                if ensemble is not None and summary and summary.get('history'):
                    live_history = pd.DataFrame(summary['history']).rename(columns={'round': 'GW'})
                    rolling_stats = ('minutes', 'ict_index', 'creativity', 'influence', 'threat', 'total_points')
                    for col in (*rolling_stats, 'value'):
                        live_history[col] = pd.to_numeric(live_history[col], errors='coerce').fillna(0)
                    aggregation = dict.fromkeys(rolling_stats, 'sum')
                    aggregation.update(value='first', opponent_team='first', was_home='first')
                    live_history = live_history.groupby('GW', as_index=False).agg(aggregation)
                    for col in rolling_stats:
                        live_history[f'rolling_3_{col}'] = live_history[col].shift(1).rolling(3, min_periods=1).mean().fillna(0)
                    live_history['raw_xp'] = ensemble.predict(live_history[model['features']])
                    year = first_deadline.year
                    season = f"{year}-{(year + 1) % 100:02d}"
                    live_history['season'] = season
                    player_df = pd.concat([player_df[player_df['season'] != season], live_history], ignore_index=True)

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
                    'raw_xp': 'Base xP',
                    'total_points': 'Actual Points'
                })
                chart_df['Gameweek'] = chart_df['GW'].astype(str)

                st.markdown(f"#### 📈 {selected_hist_player} — {selected_season} Gameweek Performance")
                
                fig = px.line(chart_df, x='Gameweek', y=['Base xP', 'Actual Points'], markers=True,
                              color_discrete_sequence=['#DAF995', '#EDCD94'])
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=[str(i) for i in range(1, 39)], fixedrange=True)
                fig.update_yaxes(fixedrange=True)
                
                st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

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
                    'raw_xp': 'Base xP',
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
                        "Base xP": "{:.1f}",
                        "Actual Points": "{:d}"
                    }),
                    width="stretch",
                    hide_index=True
                )

def render_hall_of_fame():
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
            df_career.style.set_properties(**{'color': '#EFF4EE'}).format({"Career FPL Points": "{:,d}"}),
            width="stretch",
            height=300,
            hide_index=True
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
            df_single.style.set_properties(**{'color': '#EFF4EE'}).format({"Points": "{:d}"}),
            width="stretch",
            height=320,
            hide_index=True
        )

    st.caption("ℹ️ Note: Historical records pre-dating 2016 are statically curated.")


def render_model_report():
    st.title("Model Report")
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    ensemble = json.loads((results_dir / "2025-26_position_models_v1.json").read_text())
    baseline = json.loads((results_dir / "2025-26_linear_reg_v1.json").read_text())
    st.caption(
        f"Saved {ensemble['season']} base-forecast backtest · "
        f"{ensemble['n_gameweeks']} gameweeks · {ensemble['n_player_gameweeks']:,} player-gameweeks"
    )
    comparison = pd.DataFrame(
        [
            {
                "Model": label,
                "MAE": result["overall"]["mae"],
                "RMSE": result["overall"]["rmse"],
                "Mean rank correlation": result["spearman_rank_correlation_mean_per_gw"],
            }
            for label, result in (("Position ensemble", ensemble), ("Linear regression", baseline))
        ]
    )
    st.subheader("Forecast comparison")
    st.caption("MAE and RMSE are in FPL points. Rank correlation is averaged across gameweeks.")
    st.dataframe(
        comparison, hide_index=True,
        column_config={metric: st.column_config.NumberColumn(format="%.3f") for metric in comparison.columns[1:]},
    )

    st.subheader("Errors by actual outcome")
    outcomes = {
        "Zeros": "Did not play",
        "Blanks": "Played, 2 pts or fewer",
        "Tickers": "Scored 3–4 pts",
        "Haulers": "Scored 5+ pts",
    }
    buckets = pd.DataFrame(ensemble["by_bucket"])
    buckets["bucket"] = buckets["bucket"].map(outcomes)
    buckets = buckets.rename(columns={"bucket": "Actual outcome", "n": "Rows", "mae": "MAE", "rmse": "RMSE"})
    st.dataframe(
        buckets, hide_index=True, column_order=("Actual outcome", "Rows", "MAE", "RMSE"),
        column_config={
            "Rows": st.column_config.NumberColumn(format="localized"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("Methodology"):
        st.markdown(
            "The position ensemble blends XGBoost and Random Forest forecasts for each FPL position. "
            "The saved backtest uses walk-forward features from earlier gameweeks and evaluates **base xP** "
            "against actual FPL points.\n\n"
            "**Adjusted xP** is the forecast shown on Gameweek and Players after adjustments for factors "
            "such as availability, rotation and fixture difficulty. **Captain score** ranks captain options "
            "using a separate heuristic. The model uses current market prices and Premier League fixtures."
        )
        legacy = json.loads((PROJECT_ROOT / "metrics.json").read_text())
        st.caption(f"Legacy linear regression metrics from `metrics.json` · RMSE {legacy['rmse']:.2f} · R² {legacy['r2']:.3f}. {legacy['notes']}")
        st.caption("Original model and applied ML work by Subanta Poudel.")

    with st.expander("Backtest by season"):
        if st.toggle("Show season charts and forecast misses"):
            render_accuracy_page()

    with st.expander("Hall of Fame"):
        render_hall_of_fame()


def render_solver_export(raw_json, model, current_gw):
    horizon = st.number_input(
        "Horizon (gameweeks)", min_value=1, max_value=10, value=5, step=1,
        help="The export starts at the gameweek expected by the solver, which can be the next round during live play.",
    )
    st.caption(
        "The CSV works with Open FPL Solver. Fixture context changes across gameweeks while player form "
        "uses the current snapshot. Double gameweeks use a single representative fixture per player."
    )
    export_scope = (current_gw, horizon)
    if st.button("Generate projections", key="gen_projections"):
        with st.spinner("Generating projections…"):
            projections, gameweeks = build_projection_csv(raw_json, model, horizon=horizon)
        st.session_state["solver_export"] = (export_scope, projections.to_csv(index=False), gameweeks)

    export = st.session_state.get("solver_export")
    if export and export[0] == export_scope:
        _, csv_text, gameweeks = export
        if gameweeks:
            st.caption(f"Covers Gameweek {gameweeks[0]}–{gameweeks[-1]}.")
        st.download_button("Download projections.csv", csv_text, "projections.csv", "text/csv")


ROUTES = {
    "gameweek": "Gameweek",
    "my-team": "My Team",
    "players": "Players",
    "mini-league": "Mini-League",
    "model-report": "Model Report",
}
NAVIGATION_KEY = "deadline_navigation"
TIMEZONE_LABELS = {
    "UTC": "UTC",
    "Europe/London": "United Kingdom",
    "Europe/Paris": "Central Europe",
    "Asia/Kathmandu": "Nepal",
    "Asia/Kolkata": "India",
    "America/New_York": "US Eastern",
    "America/Los_Angeles": "US Pacific",
}


def timezone_label(option):
    zone = getattr(TZ_MAP[option], "key", None)
    return TIMEZONE_LABELS.get(zone, "Server local time")


def initial_view():
    view = st.query_params.get("view")
    if view in ROUTES:
        return view
    # Older league bookmarks can also carry the original app's default team ID.
    if "league_id" in st.query_params:
        return "mini-league"
    if "team_id" in st.query_params:
        return "my-team"
    return "gameweek"


def sync_view():
    st.query_params["view"] = st.session_state[NAVIGATION_KEY]


def open_my_team():
    st.session_state[NAVIGATION_KEY] = "my-team"
    sync_view()


def main():
    st.set_page_config(page_title="Deadline Sheet", page_icon="⚽", layout="wide")
    render_theme()

    with st.container(key="deadline_masthead"):
        wordmark, settings = st.columns([6, 1], vertical_alignment="center")
        with wordmark:
            render_wordmark()
        with settings, st.popover("Timezone"):
            selected_tz = st.selectbox("Display timezone", list(TZ_MAP), format_func=timezone_label)

    if NAVIGATION_KEY not in st.session_state:
        st.session_state[NAVIGATION_KEY] = initial_view()
    view = st.radio(
        "Primary navigation", list(ROUTES),
        format_func=ROUTES.get, key=NAVIGATION_KEY, on_change=sync_view,
        horizontal=True, label_visibility="collapsed", width="stretch",
    )

    if view == "model-report":
        render_model_report()
    else:
        forecast = load_app_data()
        if forecast is None:
            st.title(ROUTES[view])
            st.error("The forecast is temporarily unavailable. Try refreshing the page in a moment.")
        else:
            render_forecast_view(view, forecast, selected_tz)

    render_html('''
        <footer class="deadline-footer">
            <span>Deadline Sheet · Subanta Poudel</span>
            <span>Independent Fantasy Premier League project</span>
        </footer>
    ''')


def render_forecast_view(view, forecast, selected_tz):
    raw_json, model, df_live, current_gw, is_preseason = forecast
    season_finished = df_live["season_finished"].fillna(False).any()

    if view == "gameweek":
        event = next((item for item in raw_json["static"].get("events", []) if item["id"] == current_gw), None)
        render_gameweek_header(
            current_gw, event, selected_tz,
            is_preseason=is_preseason, season_finished=season_finished, on_rate_team=open_my_team,
        )
        render_captaincy_card(df_live)
        with st.container(key="deadline_dashboard"):
            projections, fixtures = st.columns([2, 1], gap="medium")
            with projections:
                render_player_projections(df_live, key_prefix="gameweek", sanitize_query=sanitize_search_input)
            with fixtures:
                render_gameweek_fixtures(raw_json, current_gw, selected_tz)

        with st.expander("Five-gameweek fixture ticker"):
            render_fixture_ticker(raw_json, current_gw)
        with st.expander("Team of the Week"):
            squad, squad_info = select_team_of_the_week(df_live)
            if squad is not None:
                render_pitch_visualizer(squad, squad_info, current_gw)
            else:
                st.caption("A full XI could not be selected from this gameweek’s forecast.")
        return

    st.title(ROUTES[view])
    if season_finished:
        st.caption(f"Season complete · Showing the final gameweek (GW {current_gw}).")
    else:
        st.caption(f"Gameweek {current_gw}")

    if view == "my-team":
        render_rate_my_team_section(df_live, current_gw)
        with st.expander("Solver export"):
            render_solver_export(raw_json, model, current_gw)
    elif view == "players":
        render_player_projections(df_live, key_prefix="players", sanitize_query=sanitize_search_input)
        with st.expander("Season points explorer"):
            render_season_points_explorer(df_live, raw_json)
        with st.expander("Player history"):
            if st.toggle("Show player history"):
                render_player_history(df_live, raw_json, model)
    elif view == "mini-league":
        render_mini_league_section(df_live, current_gw)


if __name__ == "__main__":
    main()
