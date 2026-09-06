import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Internal Modules
from src.data_loader import fetch_api_data, fetch_player_live_summary, fetch_user_team, TZ_MAP, format_timestamp_tz
from src.features import process_data, fetch_full_history
from src.predictor import load_model, make_predictions

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

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="CatalanPlays",
    layout="wide",
    page_icon="⚽"
)

# --- TYPOGRAPHY ---
# Big Shoulders Display: condensed sporting/scoreboard energy, used sparingly
# on headings only. Source Sans 3: body/UI text. IBM Plex Mono: anything
# numeric (xP, BPS, price, RMSE/R²) so stat columns line up and read as
# data, not prose. Streamlit's own emotion-generated classes carry higher
# specificity than a plain selector, so the font-family overrides need
# !important to reliably win — this is fighting a third-party framework's
# scoped CSS, not our own cascade.
# Known limitation: st.dataframe renders cells to an HTML canvas (glide-data-
# grid), so these rules cannot reach text *inside* a dataframe grid — only
# Streamlit's own UI chrome (headings, sidebar, metrics, buttons, captions).
render_html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {
    --font-display: 'Big Shoulders Display', 'Arial Narrow', sans-serif;
    --font-body: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, p, div, label, li, button {
    font-family: var(--font-body) !important;
}
/* Deliberately NOT touching `span`: Streamlit renders its icon glyphs
   (expander chevrons, toolbar icons) as ligature text inside spans keyed
   to a dedicated icon font — a blanket font-family override there replaces
   the icon font and the glyph shows as its literal name ("arrow_right")
   instead of rendering. Body text still gets Source Sans 3 through normal
   CSS inheritance from `body` above; only the icon spans need to opt out. */
h1 {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    letter-spacing: .2px;
}
h2 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
}
h3, h4 {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
}
code, pre, .stCodeBlock {
    font-family: var(--font-mono) !important;
}
</style>
""")

# --- HEADER  ---
st.title("⚽ CatalanPlays")
st.markdown("**Created by Subanta Poudel** | *Hybrid Intelligence Expected Points (xP) Engine*")
st.caption("Project Established: August 2026 | Current Engine: CatalanPlays Engine v1.2")
st.markdown("---")

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

# --- LAST-GAMEWEEK RECAP TARGET ---
_events_list = raw_json.get('static', {}).get('events', []) if raw_json else []
_current_gw_event = next((e for e in _events_list if e.get('id') == CURRENT_GW), None)

_finished_gw_ids = [e['id'] for e in _events_list if e.get('finished')]
RECAP_GW = max(_finished_gw_ids) if _finished_gw_ids else None

CURRENT_SEASON_LABEL = "2025-26"

# --- HELPER: LAST GAMEWEEK RECAP (PREDICTED VS. ACTUAL, NO LEAKAGE, NO STORAGE) ---
def render_last_gameweek_recap(gw_id):
    """
    Shows predicted-vs-actual for the most recently finished gameweek.

    Reconstructed entirely from fetch_full_history() -- the same historical
    per-gameweek pipeline the Player History tab already uses. Each row
    there is a true point-in-time record: rolling features are computed
    with .shift(1).rolling(3), grouped by player and season, so a
    gameweek's `raw_xp` only ever depends on gameweeks *before* it. That
    means this is safe to recompute fresh on every page load -- no local
    snapshot file, nothing to lose on a Streamlit Cloud redeploy.

    Trade-off, stated plainly: this compares the base statistical model's
    raw_xp, not the full rule-adjusted final_xp shown in the live
    Predictions table. Reconstructing the expert-rule layer (fixture
    difficulty, rotation risk, DEFCON, etc. as they stood *for that
    gameweek specifically*) historically is a larger undertaking than
    reusing data this app already fetches for another tab.
    """
    if gw_id is None:
        return

    df_hist_all = load_historical_data(model)
    if df_hist_all.empty:
        return

    gw_rows = df_hist_all[
        (df_hist_all['season'] == CURRENT_SEASON_LABEL) & (df_hist_all['GW'] == gw_id)
    ].copy()
    if gw_rows.empty:
        st.info(f"ℹ️ **GW {gw_id} Recap unavailable:** the historical data source hasn't published this gameweek yet. Check back shortly.")
        return

    gw_rows['Diff'] = gw_rows['raw_xp'] - gw_rows['total_points']
    rmse = float(np.sqrt((gw_rows['Diff'] ** 2).mean()))
    top_performer = gw_rows.sort_values('total_points', ascending=False).iloc[0]

    st.markdown(f"### 📋 Last Gameweek Recap — GW {gw_id}")
    st.caption("Reconstructed from historical per-gameweek data using only stats available before that gameweek -- no snapshot storage, no hindsight.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("RMSE (This Gameweek)", f"{rmse:.2f}")
    with m2:
        st.metric("Players Tracked", f"{len(gw_rows)}")
    with m3:
        st.metric("Top Performer", f"{top_performer['name']} ({int(top_performer['total_points'])} pts)")

    display_recap = gw_rows.rename(columns={
        'name': 'Player', 'team': 'Team', 'position': 'Pos',
        'raw_xp': 'Predicted', 'total_points': 'Actual'
    })[['Player', 'Team', 'Pos', 'Predicted', 'Actual', 'Diff']].sort_values(by='Actual', ascending=False).head(20).reset_index(drop=True)

    st.dataframe(
        display_recap.style
            .set_properties(**{'color': '#F2F4F8'})
            .format({"Predicted": "{:.1f}", "Diff": "{:+.1f}"}),
        use_container_width=True,
        height=420,
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
            badge_html = '<span style="background:#FFC72C; color:#1A1408; font-weight:bold; padding:1px 4px; border-radius:3px; font-size:10px; margin-left:3px;">(C)</span>'
        elif name == vice_name:
            badge_html = '<span style="background:#C0C0C0; color:#1A1408; font-weight:bold; padding:1px 4px; border-radius:3px; font-size:10px; margin-left:3px;">(V)</span>'

        return f'''
        <div style="background: rgba(26, 34, 53, 0.95); border: 1px solid #2A3450; border-radius: 8px; padding: 6px 4px; text-align: center; flex: 1 1 70px; max-width: 95px; min-width: 65px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); box-sizing: border-box;">
            <div style="font-weight: 700; font-size: 12px; color: #F2F4F8; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{name}{badge_html}</div>
            <div style="font-size: 10px; color: #8891A6; margin-top: 2px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{team} · {opp}</div>
            <div style="font-family: var(--font-mono); font-size: 12px; font-weight: 700; color: #D64B5E; margin-top: 4px;">{xp:.1f} xP</div>
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
    <div style="background: linear-gradient(180deg, #16233B 0%, #0F1420 100%); border: 2px solid #D64B5E; border-radius: 12px; padding: 18px 12px; margin-bottom: 25px; box-shadow: inset 0 0 40px rgba(0,0,0,0.6); position: relative;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 8px; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;">
            <div style="font-family: var(--font-display); font-weight: 700; font-size: 18px; color: #FFC72C;">⭐ AI Team of the Week — GW {gameweek}</div>
            <div style="font-size: 13px; color: #F2F4F8;">Formation: <strong>{formation}</strong> | Squad xP: <strong style="font-family: var(--font-mono); color:#FFC72C;">{total_xp:.1f} pts</strong></div>
        </div>
        {fwd_row}
        {mid_row}
        {def_row}
        {gkp_row}
    </div>
    '''
    render_html(pitch_html)

# --- HELPER: CAPTAINCY RATING CARD ---
def render_captaincy_card(df_candidates):
    """
    Renders top 5 captaincy options for the upcoming gameweek with 5-star evaluation
    and key rationale.
    """
    if df_candidates is None or df_candidates.empty:
        return

    # Captaincy eligibility: attackers, or defenders with real attacking
    # threat (own xGI/90 > 0.30 -- see src/predictor.py). Ranked by Ceiling
    # Score rather than plain final_xp, so a nailed budget defender with an
    # easy fixture doesn't crowd out genuine explosive-upside attackers.
    eligible = df_candidates[df_candidates.get('captain_eligible', False) == True]
    if eligible.empty:
        eligible = df_candidates  # Degrade gracefully rather than show nothing.

    caps = eligible.sort_values(by='ceiling_score', ascending=False).head(5).reset_index(drop=True)
    if caps.empty:
        return

    st.markdown("### 👑 Top Captain Picks")

    cap_rows = []
    for idx, r in caps.iterrows():
        xp = float(r.get('final_xp', 0.0))
        name = r.get('web_name', '')
        team = r.get('team_name', '')
        opp = r.get('next_opponent', '-')
        diff = float(r.get('next_match_difficulty', 3))
        starter_prob = float(r.get('starter_prob', 100))

        if xp >= 6.0:
            stars = "⭐⭐⭐⭐⭐"
        elif xp >= 4.5:
            stars = "⭐⭐⭐⭐"
        elif xp >= 3.5:
            stars = "⭐⭐⭐"
        elif xp >= 2.5:
            stars = "⭐⭐"
        else:
            stars = "⭐"

        rationale_parts = []
        if "(H)" in opp:
            rationale_parts.append("Home Fixture")
        elif "(A)" in opp:
            rationale_parts.append("Away Match")

        if diff <= 2:
            rationale_parts.append("Favorable Opponent (FDR 1-2)")
        elif diff >= 4:
            rationale_parts.append("Tough Matchup")

        val_m = float(r.get('val_m', r.get('value', 0)))
        if bool(r.get('is_penalty_taker', False)):
            rationale_parts.append("Penalty Taker")
        if val_m >= 9.5:
            rationale_parts.append("Premium Captain Potential")
        elif starter_prob >= 90:
            rationale_parts.append("Nailed Starter (100%)")
        if r.get('position') == 'DEF':
            rationale_parts.append(f"Attacking Threat (xGI/90 {float(r.get('xgi_per_90', 0)):.2f})")

        rationale_str = " · ".join(rationale_parts) if rationale_parts else "Strong Gameweek Baseline"

        cap_rows.append({
            "Rank": idx + 1,
            "Player": f"{name} ({team})",
            "Opponent": opp,
            "Predicted Points": f"{xp:.1f} pts",
            "Star Rating": stars,
            "Key Rationale": rationale_str
        })

    # Note: st.dataframe never interprets markdown syntax inside cell text
    # (that's a st.markdown/st.write-only behavior) — "**bold**" would show
    # as literal asterisks. Real emphasis on this column has to come from
    # the Styler's CSS properties instead.
    df_cap_display = pd.DataFrame(cap_rows)
    st.dataframe(
        df_cap_display.style
            .set_properties(**{'color': '#F2F4F8'})
            .set_properties(subset=['Player'], **{'font-weight': '700'}),
        use_container_width=True,
        height=210,
        hide_index=True
    )

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
            style_df.loc[idx, 'Team'] = 'color: #F2F4F8; font-weight: bold;'
            style_df.loc[idx, 'Avg FDR'] = 'color: #FFC72C; font-weight: bold;'
            for col in gw_cols:
                val = r[col]
                style_df.loc[idx, col] = style_fdr_cells(val, t_name, col)
        return style_df

    st.markdown("### 🗓️ 5-Gameweek FDR Schedule Ticker")
    st.caption("Interactive 5-week schedule matrix for all 20 Premier League clubs color-coded by Fixture Difficulty Rating (FDR). Sorted by overall easiest schedule.")
    
    st.dataframe(
        df_ticker.style.apply(apply_ticker_styling, axis=None).format({"Avg FDR": "{:.2f}"}),
        use_container_width=True,
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


# --- HELPER: RATE MY TEAM (LIVE SQUAD EVALUATOR) ---
def render_rate_my_team_section(df_live, raw_json, current_gw):
    """
    Renders the interactive "Rate My Team" squad evaluation card.
    Pulls user's 15 FPL picks via Team ID, calculates total starting XI xP,
    weakest starter link, bench optimization alert, recommended transfer plan card, and displays a full squad table.
    """
    st.markdown("### 📊 Rate My Team (Live Squad Evaluator)")
    st.caption("Enter your official FPL Team ID (from your FPL web URL: `entry/{team_id}/event/...`) to evaluate your starting XI projected points and bench optimization.")

    c1, c2 = st.columns([1, 2])
    with c1:
        team_id = st.number_input("Enter FPL Team ID", min_value=1, value=3753, step=1, key="rmt_team_id_input")
        fetch_btn = st.button("Evaluate Squad 🚀", key="rmt_fetch_btn")

    if team_id:
        team_data = fetch_user_team(int(team_id), current_gw)
        if not team_data or 'picks' not in team_data:
            st.warning(f"⚠️ Could not retrieve squad picks for Team ID `{team_id}` in Gameweek {current_gw}. Verify your Team ID.")
            return

        picks_df = pd.DataFrame(team_data['picks'])
        gw_used = team_data.get('gw', current_gw)
        
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
                <div style="background: rgba(26, 34, 53, 0.95); border: 1px solid #2A3450; border-radius: 10px; padding: 14px 18px; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 17px; color: #FFC72C; margin-bottom: 8px;">🎯 Recommended Gameweek Transfer</div>
                    <div style="font-size: 14px; color: #F2F4F8; display: flex; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #7f1d1d; color: #f87171; font-weight: bold; padding: 3px 8px; border-radius: 4px;">[SELL]</span>
                        <strong>{sell_n}</strong> <span style="font-family: var(--font-mono);">({sell_xp_val:.1f} xP)</span>
                        <span style="font-size: 16px; color: #FFC72C;">➔</span>
                        <span style="background: #064e3b; color: #34d399; font-weight: bold; padding: 3px 8px; border-radius: 4px;">[BUY]</span>
                        <strong>{buy_n}</strong> <span style="font-family: var(--font-mono);">({buy_xp_val:.1f} xP)</span>
                        <span style="color: #8891A6; margin: 0 4px;">|</span>
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
                return "👑 Captain (2x)"
            elif is_vc:
                return "🛡️ Vice-Captain"
            elif pos <= 11:
                return "⚡ Starter"
            else:
                return "🪑 Bench"

        merged['Role'] = merged.apply(format_role, axis=1)
        merged['Projected Points'] = merged['final_xp'] * merged['multiplier']

        display_rmt_cols = {
            'position_pick': 'Slot',
            'web_name': 'Player',
            'team_name': 'Team',
            'position_pred': 'Pos',
            'next_opponent': 'Opponent',
            'Role': 'Role',
            'final_xp': 'Base xP',
            'Projected Points': 'Effective xP'
        }

        rmt_table = merged[display_rmt_cols.keys()].rename(columns=display_rmt_cols).sort_values(by='Slot').reset_index(drop=True)

        def style_rmt_rows(s):
            is_starter = s['Slot'] <= 11
            is_c = "Captain" in str(s['Role'])
            if is_c:
                return ['background-color: #2b2410; color: #F2F4F8; font-weight: bold;' for _ in s]
            elif is_starter:
                return ['color: #F2F4F8;' for _ in s]
            else:
                return ['background-color: #1A2235; color: #F2F4F8;' for _ in s]

        st.dataframe(
            rmt_table.style.apply(style_rmt_rows, axis=1).format({"Base xP": "{:.1f}", "Effective xP": "{:.1f}"}),
            use_container_width=True,
            height=420,
            hide_index=True
        )

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Control Panel")
if IS_PRESEASON:
    st.sidebar.info("System Status: **Season Has Not Started**")
else:
    st.sidebar.info(f"System Status: **GW {CURRENT_GW} Active**")

selected_tz = st.sidebar.selectbox(
    "🌐 Timezone",
    ["Local / Regional", "UTC", "BST (UK / UTC+1)", "CET (Europe / UTC+2)", "NPT (Nepal / UTC+5:45)", "IST (India / UTC+5:30)", "EST (US East / UTC-5)", "PST (US West / UTC-8)"],
    index=0,
    help="Select your timezone to convert FPL Gameweek deadlines and match kickoff times."
)

try:
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
        st.sidebar.divider()
        st.sidebar.markdown("### 🧠 Model Diagnostics")
        st.sidebar.metric("RMSE (Error)", f"{metrics.get('rmse', 'N/A')}", delta_color="inverse")
        st.sidebar.metric("R² (Accuracy)", f"{metrics.get('r2', 'N/A')}")
        st.sidebar.caption("Project Established: August 2026\nCurrent Engine: CatalanPlays Engine v1.2")
except FileNotFoundError:
    st.sidebar.caption("Project Established: August 2026\nCurrent Engine: CatalanPlays Engine v1.2")

# --- REAL-TIME MATCHDAY INTELLIGENCE ---
if raw_json and 'static' in raw_json and 'events' in raw_json['static']:
    events_list = raw_json['static']['events']
    # Reuse the same event process_data() already resolved for CURRENT_GW
    # (is_current-priority -- see src/features.py) so this banner can never
    # disagree with the sidebar/predictions about which gameweek "now" is.
    target_ev = _current_gw_event or (events_list[0] if events_list else None)

    if target_ev:
        gw_id = target_ev.get('id', 1)
        deadline_raw = target_ev.get('deadline_time')
        
        deadline_text = f"⏰ **Gameweek {gw_id}**"
        countdown_text = ""
        if deadline_raw:
            try:
                deadline_dt = pd.to_datetime(deadline_raw)
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.tz_localize('UTC')
                
                target_tz = TZ_MAP.get(selected_tz, datetime.now().astimezone().tzinfo)
                converted_dt = deadline_dt.tz_convert(target_tz)
                
                now_utc = pd.Timestamp.now(tz='UTC')
                diff = deadline_dt - now_utc
                days = diff.days
                hours = diff.seconds // 3600
                mins = (diff.seconds % 3600) // 60
                
                tz_label = selected_tz
                deadline_fmt = converted_dt.strftime('%a %d %b %H:%M')
                if diff.total_seconds() > 0:
                    countdown_str = f"in {days}d {hours}h" if days > 0 else f"in {hours}h {mins}m"
                else:
                    countdown_str = "PASSED / IN PROGRESS"
                
                deadline_text = f"⏰ **Gameweek {gw_id} Deadline:** {deadline_fmt} ({tz_label})"
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
                    ko_str = format_timestamp_tz(ko_raw, selected_tz, fmt="%a %H:%M Local") if ko_raw else ""

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

# --- TABS ---
tab_pred, tab_hist, tab_hauls, tab_info = st.tabs(["🔮 Predictions", "📊 Player History", "🏆 Hall of Fame", "ℹ️ Documentation"])

# === TAB 1: LIVE PREDICTIONS ===
with tab_pred:
    render_last_gameweek_recap(RECAP_GW)

    totw_squad, totw_info = select_team_of_the_week(df_live)
    if totw_squad is not None:
        render_pitch_visualizer(totw_squad, totw_info, CURRENT_GW)

    # Render Captaincy Rating Card
    render_captaincy_card(df_live)
    st.markdown("---")

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
    # Core columns are what a manager scans first: who, for whom, against
    # whom, at what price in xP, and why. Everything else (DEFCON, bonus/BPS,
    # price movement, ownership) is real signal but not first-glance signal,
    # so it lives in a collapsed expander instead of stretching the table
    # wide on mobile.
    core_cols = {
        'web_name': 'Player',
        'team_name': 'Team',
        'next_opponent': 'Opponent',
        'position': 'Pos',
        'final_xp': 'Predicted Points',
        'reasoning': 'Reasoning'
    }
    secondary_cols = {
        'web_name': 'Player',
        'defensive_contribution_per_90': 'DEFCON/90',
        'bonus': 'Bonus',
        'bps': 'BPS',
        'value': 'Price (£m)',
        'price_change_today': 'Price Δ (Today)',
        'ownership_pct': 'Owned %',
        'next_match_difficulty': 'Diff (1-5)'
    }

    view_df_sorted = view_df.sort_values(by='final_xp', ascending=False).reset_index(drop=True)
    core_table = view_df_sorted[core_cols.keys()].rename(columns=core_cols)
    secondary_table = view_df_sorted[secondary_cols.keys()].rename(columns=secondary_cols)

    # HIGHLIGHT TOP 5 PLAYERS (The Visual Fix with High-Contrast Text)
    def highlight_top5(s):
        is_top5 = s.name < 5 # Row order matches view_df_sorted, so top 5 are 0,1,2,3,4
        return ['background-color: #2e2612; color: #F2F4F8;' if is_top5 else 'color: #F2F4F8;' for _ in s]

    def format_price_delta(v):
        if v > 0:
            return f"▲ +£{v:.1f}m"
        elif v < 0:
            return f"▼ -£{abs(v):.1f}m"
        return "— £0.0m"

    st.dataframe(
        core_table.style.apply(highlight_top5, axis=1).format({"Predicted Points": "{:.1f}"}),
        use_container_width=True,
        height=600,
        hide_index=True
    )

    with st.expander("📊 View Underlying Defensive & Bonus Metrics"):
        st.dataframe(
            secondary_table.style.apply(highlight_top5, axis=1).format({
                "Price (£m)": "£{:.1f}",
                "DEFCON/90": "{:.1f}",
                "Bonus": "{:d}",
                "BPS": "{:d}",
                "Price Δ (Today)": format_price_delta,
                "Owned %": "{:.1f}%"
            }),
            use_container_width=True,
            height=500,
            hide_index=True
        )

    st.markdown("---")
    # Render 5-Gameweek Fixture Ticker
    render_fixture_ticker(raw_json, CURRENT_GW)

    st.markdown("---")
    # Render Rate My Team Evaluator
    render_rate_my_team_section(df_live, raw_json, CURRENT_GW)

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
                    use_container_width=True,
                    hide_index=True
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
            df_career.style.set_properties(**{'color': '#F2F4F8'}).format({"Career FPL Points": "{:,d}"}),
            use_container_width=True,
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
            df_single.style.set_properties(**{'color': '#F2F4F8'}).format({"Points": "{:d}"}),
            use_container_width=True,
            height=320,
            hide_index=True
        )

    st.caption("ℹ️ Note: Historical records pre-dating 2016 are statically curated.")

# === TAB 4: DOCUMENTATION ===
with tab_info:
    st.header("📘 System Documentation")
    
    st.subheader("1. Architecture")
    st.markdown("""
    The **CatalanPlays Engine v1.2** uses a hybrid architecture:
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
    st.text(f"Developed by Subanta Poudel\nProject Established: August 2026\nCurrent Engine: CatalanPlays Engine v1.2\nLast Updated: {datetime.now().strftime('%Y-%m-%d')}")

# --- FOOTER ---
st.markdown("---")
render_html("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0F1420;
        color: #8891A6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #2A3450;
    }
    </style>
    <div class="footer">
        © 2026 CatalanPlays Engine v1.2 | Created by <b>Subanta Poudel</b> | Project Established: August 2026
    </div>
    """)