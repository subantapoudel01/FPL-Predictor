from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import pandas as pd
import numpy as np
import requests
import unicodedata
import re
import streamlit as st

def _normalize_name(name):
    """
    Normalizes player names across seasons by stripping accents/diacritics,
    converting to lowercase, and removing spaces & non-alphanumeric characters.
    e.g. 'Bruno Borges Fernandes' -> 'brunoborgesfernandes'
         'Bukayo Saka' -> 'bukayosaka'
         'Cole Palmer' -> 'colepalmer'
         'Gonçalo' -> 'goncalo'
    """
    if not isinstance(name, str) or not name:
        return ""
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_str = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r'[^a-z0-9]', '', ascii_str.lower())


def match_player_name(first_name, second_name, web_name, code, code_map, name_index):
    """
    Shared 5-tier player-identity matching cascade, used wherever this app
    has to line up a player from one source against another that doesn't
    share a common ID -- originally built for pre-season baseline matching
    (see _load_historical_baselines below) and reused as-is by
    src/understat.py, which has exactly the same problem (Understat has no
    FPL id, no Opta code) against a different data source.

    Tiers, in order:
      1. Opta `code` -- permanent across seasons, if the caller has one for
         this source (Understat doesn't, so this tier simply never fires
         for that caller -- which is expected, not a bug).
      2. Normalized "first second" full name.
      3. Normalized "first web_name" (catches players commonly known by a
         shortened first name on their web_name).
      4. Normalized web_name alone.
      5. Substring match: both normalized first and second name appear
         somewhere in a candidate key (catches suffix/prefix differences
         and most transliteration variants that survive normalization).

    Returns the matching key into `name_index` (or `code_map`), or None if
    no tier matched -- callers are expected to log unmatched names rather
    than silently dropping them.
    """
    if code_map and code in code_map:
        return code

    nf = _normalize_name(f"{first_name} {second_name}")
    nfw = _normalize_name(f"{first_name} {web_name}")
    nw = _normalize_name(web_name)

    if nf in name_index:
        return nf
    if nfw in name_index:
        return nfw
    if nw in name_index:
        return nw

    fir_norm = _normalize_name(first_name)
    sec_norm = _normalize_name(second_name)
    if fir_norm and sec_norm:
        for candidate in name_index:
            if fir_norm in candidate and sec_norm in candidate:
                return candidate

    # Mononym fallback: a player known by one name only (common for
    # Brazilian/Portuguese players -- Alisson, Casemiro, Rodri, Willian)
    # has no second_name at all, so the two-part substring check above
    # never even runs for them -- this was the single largest cause of
    # unmatched Understat players before this tier was added. Only
    # accepted when it resolves to exactly one candidate, so a short,
    # common single name can't silently grab the wrong player out of a
    # roster of hundreds.
    single = nw or fir_norm
    if single and not sec_norm:
        hits = [c for c in name_index if single in c]
        if len(hits) == 1:
            return hits[0]

    return None

def _load_historical_baselines():
    """
    Loads historical dataset (Data/Cleaned/model_ready_data.csv) to extract
    active rolling statistics for returning players and calculate
    positional medians for new transfers / promoted players.
    """
    data_path = Path(__file__).resolve().parent.parent / "Data" / "Cleaned" / "model_ready_data.csv"
    if not data_path.exists():
        return None, None, None, None, None
        
    try:
        hist_df = pd.read_csv(data_path)
        hist_df['position'] = hist_df['position'].replace({'GK': 'GKP'})
        
        if 'minutes_per_game' not in hist_df.columns:
            hist_df['minutes_per_game'] = hist_df['rolling_3_minutes']
            
        feature_cols = [
            'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
            'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'minutes_per_game'
        ]
        
        hist_df['norm_name'] = hist_df['name'].apply(_normalize_name)
        
        # Filter for rows with valid rolling minutes > 0 (active starter history)
        active_hist = hist_df[hist_df['rolling_3_minutes'] > 0].copy()
        if active_hist.empty:
            active_hist = hist_df.dropna(subset=['rolling_3_minutes']).copy()
            
        # Group active history by normalized name (take average of recent active appearances)
        active_name_baselines = (
            active_hist.groupby('norm_name')[feature_cols]
            .tail(10)
            .groupby(active_hist['norm_name'])[feature_cols]
            .mean()
            .to_dict(orient='index')
        )
        
        # Build Opta code map if 'code' column exists
        code_map = {}
        if 'code' in hist_df.columns and not hist_df['code'].dropna().empty:
            code_hist = hist_df.dropna(subset=['code', 'rolling_3_minutes'])
            code_map = (
                code_hist.groupby('code')[feature_cols]
                .tail(10)
                .groupby(code_hist['code'])[feature_cols]
                .mean()
                .to_dict(orient='index')
            )
            
        positional_medians = active_hist.groupby('position')[feature_cols].median().to_dict(orient='index')
        global_medians = active_hist[feature_cols].median().to_dict()
        
        return code_map, active_name_baselines, positional_medians, global_medians, feature_cols
    except Exception:
        return None, None, None, None, None


def _live_team_rolling_goals(fixtures, teams_map, windows=(5, 10)):
    """
    Team goals-for/against averaged over each team's last up-to-`window`
    finished fixtures *this season* -- the live equivalent of
    fetch_full_history()'s team_rolling_{w}_team_goals_for/against, built
    from api_data['fixtures'] (every fixture in the season, not just
    target_gw -- see data_loader.py::fetch_fixtures_data) instead of
    vaastav's historical CSV. Same semantics, same column names, different
    source data, so the clean-sheet/goals-conceded component models can use
    one set of column names both live and in the backtest.

    Returns {'team_rolling_{w}_team_goals_for': {team_name: value}, ...} --
    one dict per (window, for/against) pair, ready to .map() onto a
    team-name column. 0.0 for a team with no finished fixtures yet
    (pre-season), the same cold-start default every other rolling feature
    in this codebase falls back to.
    """
    finished = [f for f in fixtures if f.get('finished') or f.get('finished_provisional')]
    finished = sorted(finished, key=lambda f: (f.get('event') or 0, f.get('kickoff_time') or ''))

    goals_for, goals_against = {}, {}
    for f in finished:
        home_id, away_id = f.get('team_h'), f.get('team_a')
        hs, aws = f.get('team_h_score'), f.get('team_a_score')
        if hs is None or aws is None:
            continue
        goals_for.setdefault(home_id, []).append(hs)
        goals_against.setdefault(home_id, []).append(aws)
        goals_for.setdefault(away_id, []).append(aws)
        goals_against.setdefault(away_id, []).append(hs)

    result = {}
    for w in windows:
        gf_map, ga_map = {}, {}
        for team_id, name in teams_map.items():
            gf_list = goals_for.get(team_id, [])[-w:]
            ga_list = goals_against.get(team_id, [])[-w:]
            gf_map[name] = float(np.mean(gf_list)) if gf_list else 0.0
            ga_map[name] = float(np.mean(ga_list)) if ga_list else 0.0
        result[f'team_rolling_{w}_team_goals_for'] = gf_map
        result[f'team_rolling_{w}_team_goals_against'] = ga_map
    return result


def process_data(api_data, target_gw=None):
    """
    Converts raw API JSON into a Model-Ready DataFrame.
    """
    static = api_data['static']
    fixtures = api_data['fixtures']
    events = static.get('events', [])
    
    # 1. Identify Target Gameweek & Pre-Season Status
    # `is_next` flips to the FOLLOWING gameweek the moment the current
    # gameweek's deadline passes -- while its own matches are still being
    # played (is_current=True, finished=False). Checking is_next first
    # meant the app jumped to predicting next week's gameweek the instant
    # a deadline passed, before this week had even kicked off. `is_current`
    # must win whenever it's set; only fall through to is_next once the
    # current gameweek has actually finished.
    active_event = next((e for e in events if e.get('is_current')), None)
    if active_event is None:
        active_event = next((e for e in events if e.get('is_next')), None)
    if active_event is None:
        active_event = next((e for e in events if not e.get('finished')), None)

    detected_gw = active_event['id'] if active_event else 1
    if target_gw is None:
        target_gw = detected_gw
        
    gw1_fixtures = [f for f in fixtures if f.get('event') == 1]
    gw1_started = any(f.get('started', False) or f.get('finished', False) for f in gw1_fixtures)
    is_preseason = (target_gw == 1) and (not gw1_started)
    completed_gws = sum(1 for e in events if e.get('finished', False))
        
    df = pd.DataFrame(static['elements'])
    
    # Map Position & Value early for fallback matching
    df['position'] = df['element_type'].map({1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    df['value'] = df['now_cost'] / 10.0
    
    # 2. Map Team Names (Always use live API team ID)
    teams_map = {t['id']: t['short_name'] for t in static['teams']}
    df['team_name'] = df['team'].map(teams_map)

    # ==========================================
    # 3. FIXTURE DIFFICULTY & OPPONENT MAPPING
    # ==========================================
    team_difficulty_map = {}
    team_opponent_map = {} # Stores "vs ARS (H)"
    team_cs_prob_map = {}

    # Team strength lookup for dynamic clean sheet odds. `strength_attack_*`
    # / `strength_defence_*` exist on the API but read 0 for every team this
    # early in a season, so this uses `strength_overall_home/away` (a
    # coarser but always-populated 2-5 scale) as the proxy for both a
    # team's own defensive solidity and its opponent's attacking threat --
    # falls back automatically to the granular fields once FPL populates
    # them, since the lookup just prefers whichever is non-zero.
    def _team_strength(team_obj, venue):
        attack_key, defence_key, overall_key = f'strength_attack_{venue}', f'strength_defence_{venue}', f'strength_overall_{venue}'
        attack = team_obj.get(attack_key) or 0
        defence = team_obj.get(defence_key) or 0
        overall = team_obj.get(overall_key) or 0
        return {
            'attack': attack if attack else overall,
            'defence': defence if defence else overall,
        }

    teams_by_id = {t['id']: t for t in static['teams']}

    relevant_fixtures = [f for f in fixtures if f.get('event') == target_gw]

    for f in relevant_fixtures:
        home_id = f['team_h']
        away_id = f['team_a']

        team_difficulty_map[home_id] = f['team_h_difficulty']
        team_difficulty_map[away_id] = f['team_a_difficulty']

        team_opponent_map[home_id] = f"{teams_map.get(away_id, '')} (H)"
        team_opponent_map[away_id] = f"{teams_map.get(home_id, '')} (A)"

        home_team_obj = teams_by_id.get(home_id, {})
        away_team_obj = teams_by_id.get(away_id, {})
        home_strength = _team_strength(home_team_obj, 'home')
        away_strength = _team_strength(away_team_obj, 'away')

        # Clean sheet odds: own defensive strength vs the specific opponent's
        # attacking strength (not the generic 1-5 FDR band), scaled to a
        # 0.05-0.55 probability band around a 0.30 league-average baseline.
        home_delta = home_strength['defence'] - away_strength['attack']
        away_delta = away_strength['defence'] - home_strength['attack']
        team_cs_prob_map[home_id] = float(np.clip(0.30 + home_delta * 0.08, 0.05, 0.55))
        team_cs_prob_map[away_id] = float(np.clip(0.30 + away_delta * 0.08, 0.05, 0.55))

    df['next_match_difficulty'] = df['team'].map(team_difficulty_map).fillna(3)
    df['next_opponent'] = df['team'].map(team_opponent_map).fillna("-")
    df['cs_probability'] = df['team'].map(team_cs_prob_map).fillna(0.30)
    
    # 4. Normalization / Pre-Season Baseline Fallback
    feature_cols = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'minutes_per_game'
    ]
    
    if is_preseason or completed_gws == 0:
        code_map, name_baselines, pos_medians, glob_medians, hist_cols = _load_historical_baselines()
        
        if name_baselines is not None and len(name_baselines) > 0:
            def get_baseline_row(row):
                fn = row.get('first_name', '')
                sn = row.get('second_name', '')
                wn = row.get('web_name', '')
                pos = row.get('position', 'MID')
                code = row.get('code')

                # Tiers 1-5: shared cascade (Opta code -> normalized full
                # name -> first+web name -> web name -> substring) -- see
                # match_player_name(). Also reused as-is by
                # src/understat.py for matching Understat's own players.
                matched_key = match_player_name(fn, sn, wn, code, code_map, name_baselines)
                if matched_key is not None:
                    hist_entry = code_map[matched_key] if (code_map and matched_key in code_map) else name_baselines[matched_key]
                    return pd.Series([float(hist_entry.get(col, 0)) for col in feature_cols], index=feature_cols)

                # 6. New Transfer / Promoted Player -> Impute Positional Median
                pos_dict = pos_medians.get(pos, glob_medians) if pos_medians else glob_medians
                if not pos_dict:
                    pos_dict = {col: 0.0 for col in feature_cols}
                return pd.Series([
                    float(pos_dict.get('rolling_3_minutes', 60.0)),
                    float(pos_dict.get('rolling_3_ict_index', 1.0)),
                    float(pos_dict.get('rolling_3_creativity', 1.0)),
                    float(pos_dict.get('rolling_3_influence', 1.0)),
                    float(pos_dict.get('rolling_3_threat', 1.0)),
                    float(pos_dict.get('rolling_3_total_points', 1.5)),
                    float(pos_dict.get('minutes_per_game', pos_dict.get('rolling_3_minutes', 60.0)))
                ], index=feature_cols)

            baseline_df = df.apply(get_baseline_row, axis=1)
            for col in feature_cols:
                df[col] = baseline_df[col].fillna(0.0)
        else:
            df['minutes_per_game'] = 0.0
            df['rolling_3_minutes'] = 0.0
            df['rolling_3_total_points'] = 0.0
            for col in ['ict_index', 'creativity', 'influence', 'threat']:
                df[col] = 0.0
                df[f'rolling_3_{col}'] = 0.0
    else:
        normalization_gw = max(1, min(target_gw, completed_gws))
        df['minutes_per_game'] = (df['minutes'] / normalization_gw).fillna(0)
        df['rolling_3_minutes'] = df['minutes_per_game']
        df['rolling_3_total_points'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        
        for col in ['ict_index', 'creativity', 'influence', 'threat']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[f'rolling_3_{col}'] = (df[col] / normalization_gw).fillna(0)

    # 4b. Defensive Contribution (DEFCON) — 2025-26 scoring rule.
    # FPL's `defensive_contribution` field is the season-cumulative combined
    # CBIT count (DEF/GKP) or CBIRT count (MID/FWD) that the rule is scored on;
    # `defensive_contribution_per_90` is that season rate normalized to a full match.
    # Verified directly against the raw components (clearances_blocks_interceptions +
    # tackles [+ recoveries for MID/FWD]) — this is not a derived/approximate stat.
    if 'defensive_contribution_per_90' not in df.columns:
        df['defensive_contribution_per_90'] = 0.0
    df['defensive_contribution_per_90'] = pd.to_numeric(
        df['defensive_contribution_per_90'], errors='coerce'
    ).fillna(0.0)

    # 4c. Live matchday intelligence: bonus points, BPS, price movement, ownership.
    # All straight from bootstrap-static — no new data source, just fields the
    # pipeline wasn't reading yet.
    if 'bonus' not in df.columns:
        df['bonus'] = 0
    df['bonus'] = pd.to_numeric(df['bonus'], errors='coerce').fillna(0).astype(int)

    if 'bps' not in df.columns:
        df['bps'] = 0
    df['bps'] = pd.to_numeric(df['bps'], errors='coerce').fillna(0).astype(int)

    # cost_change_event is FPL's own signed price movement for the current
    # event, in tenths of a million (e.g. 1 -> +£0.1m). Convert to real £m here
    # so the UI never has to remember the /10 scale factor.
    if 'cost_change_event' not in df.columns:
        df['cost_change_event'] = 0
    df['price_change_today'] = (
        pd.to_numeric(df['cost_change_event'], errors='coerce').fillna(0) / 10.0
    )

    if 'selected_by_percent' not in df.columns:
        df['selected_by_percent'] = 0.0
    df['ownership_pct'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0.0)

    # 4d. Season totals for the Season Points Explorer (Predictions tab).
    # Straight season-cumulative counts from bootstrap-static, not part of
    # any prediction -- display-only.
    for col in ['total_points', 'goals_scored', 'assists']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 5. Availability
    df['chance_of_playing'] = df['chance_of_playing_next_round'].fillna(100)

    # (c) Availability as a MODEL feature, not just a post-hoc rule-layer
    # multiplier. `chance_of_playing_next_round` was already whitelisted
    # in data_loader.py (the injury multiplier in predictor.py already
    # reads it) -- nothing new needed there. `availability_pct` is the
    # same 0/25/50/75/100 value, explicitly named and numeric-coerced, so
    # it's ready to hand to a model as an input column. The rule-layer
    # multiplier in predictor.py stays as-is for now; the plan is to
    # retrain the model with this feature and only then remove the
    # multiplier, so availability isn't double-counted.
    #
    # NOT backtestable via evaluation/backtest.py: vaastav's historical
    # per-gameweek CSVs carry no injury/availability field at all (checked
    # directly -- zero columns matching chance/avail/injur in any season).
    # Only the live app can use this column; there is no historical ground
    # truth to walk-forward test it against.
    df['availability_pct'] = pd.to_numeric(
        df['chance_of_playing_next_round'], errors='coerce'
    ).fillna(100.0)

    # 7. Component-model live inputs (src/components.py,
    # training/train_component_models.py). Everything here is a live
    # equivalent of a walk-forward feature fetch_full_history() computes
    # historically -- same naming/semantics wherever the live API makes an
    # identical computation possible, a season-aggregate proxy where it
    # doesn't (documented per field below), following the same precedent
    # already set by rolling_3_minutes = minutes_per_game above.
    for col in ['starts', 'starts_per_90', 'expected_goals_per_90', 'expected_assists_per_90',
                'expected_goals_conceded_per_90', 'saves_per_90', 'clean_sheets_per_90',
                'clearances_blocks_interceptions', 'tackles', 'recoveries',
                'defensive_contribution', 'yellow_cards', 'red_cards']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Squad depth: how many teammates in the same position are themselves
    # regular starters (rolling_3_minutes >= 60) -- a genuine input to the
    # minutes model (rotation risk rises with depth at a position), not the
    # old rule layer's hand-picked "5+ attackers over £5.5m" cutoff. Counts
    # the player themselves too (so "no depth competition" reads as 1, not
    # 0) -- the minutes model sees this as a feature and learns its own
    # relationship, no threshold chosen by hand.
    is_starter = df['rolling_3_minutes'] >= 60
    df['squad_depth_count'] = (
        df.assign(_is_starter=is_starter)
        .groupby(['team_name', 'position'])['_is_starter']
        .transform('sum')
    ).fillna(0)

    # Team rolling goals-for/against, computed from this season's already-
    # finished fixtures (api_data['fixtures'] carries every fixture in the
    # season, not just target_gw -- see data_loader.py::fetch_fixtures_data)
    # -- the live equivalent of fetch_full_history()'s
    # team_rolling_{w}_team_goals_for/against, same semantics (average
    # goals for/against over a team's last up-to-w finished matches this
    # season), just computed from this season's live fixture results
    # instead of vaastav's historical CSV. Feeds the clean-sheet and
    # goals-conceded components the same way the historical rolling
    # columns do; early season this is naturally sparse (fewer finished
    # matches than the window), same cold-start behavior every other
    # rolling feature already has.
    team_strength = _live_team_rolling_goals(fixtures, teams_map, windows=(5, 10))
    for col, mapping in team_strength.items():
        df[col] = df['team_name'].map(mapping).fillna(0.0)

    opp_short = df['next_opponent'].str.replace(r'\s*\((H|A)\)$', '', regex=True)
    for col, mapping in team_strength.items():
        df[f'opp_{col}'] = opp_short.map(mapping).fillna(0.0)

    # 6. Starter Probability (Strict Curve)
    def calculate_starter_prob(mins_avg):
        if mins_avg >= 60: return 100
        elif mins_avg >= 45: return 75
        elif mins_avg >= 30: return 25
        else: return 5
            
    df['starter_prob'] = df['minutes_per_game'].apply(calculate_starter_prob)
    
    # Fill remaining NaNs defensively
    for col in feature_cols + ['starter_prob', 'chance_of_playing', 'availability_pct', 'next_match_difficulty', 'defensive_contribution_per_90', 'bonus', 'bps', 'price_change_today', 'ownership_pct']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    return df, target_gw, is_preseason

# Multi-horizon rolling windows (matches per-position). OpenFPL
# (arXiv 2508.09992) uses ~200 features per position against our 7 --
# this is the first step toward closing that gap: every base stat gets
# rolled at each of these window sizes instead of just 3. Generated
# programmatically in fetch_full_history() below, not hand-copied per
# horizon, so adding/removing a horizon is a one-line change.
ROLLING_HORIZONS = [1, 3, 5, 10, 38]

# Base per-player stats rolled at every horizon above. Unchanged from the
# original single-horizon set -- this expands *how far back* each stat
# looks, not *which* stats are used.
ROLLED_PLAYER_STATS = ['minutes', 'ict_index', 'creativity', 'influence', 'threat', 'total_points']

# Team-level stats rolled at every horizon, attached twice per player row:
# once for their own team, once for their opponent (via TEAM_ID_KEYS
# resolution below). Opponent context barely existed before this (just
# FPL's 1-5 FDR integer) despite being where a lot of the signal lives.
ROLLED_TEAM_STATS = ['team_goals_for', 'team_goals_against']

# Raw per-match scoring-event stats the component prediction pipeline
# (src/components.py, training/train_component_models.py) is built on --
# each one is rolled walk-forward at the same horizons as ROLLED_PLAYER_STATS
# via the same _rolling_walk_forward mechanism, just a different source
# stat. 'starts' is 0/1 per match (nailed-on-starter signal for the minutes
# model); the rest map directly onto FPL's 2025-26+ scoring components.
COMPONENT_PLAYER_STATS = [
    'expected_goals', 'expected_assists', 'expected_goals_conceded',
    'saves', 'bps', 'bonus', 'yellow_cards', 'red_cards', 'starts',
]


def _rolling_walk_forward(df, group_cols, col, window):
    """
    shift(1) before rolling(window) so the value for any row only ever
    depends on strictly earlier gameweeks within that group -- the one
    mechanism every walk-forward feature in this file relies on to stay
    leak-free. min_periods=1 means an early gameweek (fewer than `window`
    prior matches to look back on) still gets a mean over however many
    exist rather than NaN; the .fillna(0) below covers a group's very
    first gameweek, where shift(1) has nothing at all to look back on.
    """
    return (
        df.groupby(group_cols)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        .fillna(0)
    )


def _build_team_id_to_name_map(df):
    """
    vaastav's per-gameweek CSV has no direct team-id -> team-name table:
    `team` is the player's own club as a short name string, but
    `opponent_team` is FPL's internal numeric team id -- a different
    namespace, needed to look up the *opponent's* own rolling stats (which
    are keyed by the name string like the player's own team is).

    Reconstructed from the data itself: for any fixture, the home side's
    `team` name is exactly the team the away side's rows list as their
    `opponent_team` id, and vice versa. Built per season (team ids are
    generally stable across seasons in FPL, but this makes no assumption
    either way).
    """
    id_to_names = {}
    home = df[df['was_home'] == True][['fixture', 'team', 'opponent_team']].drop_duplicates('fixture')
    away = df[df['was_home'] == False][['fixture', 'team', 'opponent_team']].drop_duplicates('fixture')
    merged = home.merge(away, on='fixture', suffixes=('_home', '_away'))
    for _, r in merged.iterrows():
        id_to_names.setdefault(r['opponent_team_away'], []).append(r['team_home'])
        id_to_names.setdefault(r['opponent_team_home'], []).append(r['team_away'])
    # A team id should resolve to exactly one name within a season; take
    # the most common in case of any single noisy row rather than crash.
    return {tid: pd.Series(names).mode().iloc[0] for tid, names in id_to_names.items()}


def _defcon_walk_forward_hit_rate(df, prior_strength=8.0):
    """
    Per-player, walk-forward P(hits the real DEFCON threshold this match) --
    Beta-Binomial shrinkage toward a position-level prior, not a plain
    rolling mean. A player with 2 games of history and 2 hits shouldn't
    read as a certain 100% the way an unweighted mean would show it; this
    blends their own observed hit/miss record with a prior that dominates
    early and fades as real matches accumulate.

    Prior: the position-level (DEF/GKP vs MID/FWD) empirical hit rate,
    pooled once across every row where `defcon_hit` is known. 2025-26
    onward is the only real source for this -- DEFCON didn't exist as a
    scoring rule before, so `defcon_hit` is NaN for every earlier season
    (see the caller). This prior is fit on that same 2025-26+ window; it
    is not being reused as extra unrelated training data, just the
    necessary source for a quantity that has no earlier history to draw on.

    `prior_strength` is the prior's weight in "virtual matches" -- roughly
    how many real matches of a player's own history it takes to move the
    shrunk estimate halfway from the prior to their own observed rate.
    """
    known = df['defcon_hit'].notna()
    is_def_gk = df['position'].isin(['DEF', 'GKP', 'GK'])
    group_key = pd.Series(np.where(is_def_gk, 'DEF_GKP', 'MID_FWD'), index=df.index)

    priors = df.loc[known, 'defcon_hit'].groupby(group_key[known]).mean().to_dict()
    default_prior = float(df.loc[known, 'defcon_hit'].mean()) if known.any() else 0.30
    prior = group_key.map(priors).fillna(default_prior).astype(float)

    alpha0 = prior * prior_strength
    beta0 = (1.0 - prior) * prior_strength

    # Cumulative hits/attempts strictly before this row, per player+season:
    # shift(1) first (this gameweek's own outcome must never leak into its
    # own estimate), then an expanding sum. Rows where defcon_hit is
    # unknown contribute to neither the hit count nor the attempt count --
    # a missing gameweek is excluded, not scored as a miss.
    hit_for_sum = df['defcon_hit'].where(known, 0.0)
    attempt_for_sum = known.astype(float)
    keys = [df['name'], df['season']]

    cum_hits = hit_for_sum.groupby(keys).transform(lambda x: x.shift(1).cumsum()).fillna(0)
    cum_attempts = attempt_for_sum.groupby(keys).transform(lambda x: x.shift(1).cumsum()).fillna(0)

    df['defcon_hit_prob'] = (alpha0 + cum_hits) / (alpha0 + beta0 + cum_attempts)
    return df


@st.cache_data(ttl=3600)
def fetch_full_history():
    """
    Fetches complete multi-season historical player gameweek datasets
    dynamically from Vaastav GitHub repository using PyArrow engine and column trimming.
    Cached for 1 hour -- short enough that the Season Points Explorer and
    Player History tab (see app.py) pick up a newly-published gameweek
    reasonably promptly, since this is the only place that data comes from.

    Every rolling_*/team_*/opp_* column produced here is walk-forward by
    construction (see _rolling_walk_forward) -- there is no feature
    engineering anywhere else in this codebase that touches historical
    data, so this is the one place a leakage bug could be introduced, and
    the one place it needs checking when this function changes.
    """
    seasons = {
        "2025-26": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv",
        "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
        "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
        "2022-23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
        "2021-22": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
    }

    needed_cols = [
        'name', 'GW', 'total_points', 'minutes', 'fixture', 'opponent_team',
        'was_home', 'ict_index', 'creativity', 'influence', 'threat', 'value',
        'team', 'position', 'team_h_score', 'team_a_score',
        # Component-model raw stats (see COMPONENT_PLAYER_STATS below) --
        # vaastav started publishing expected_goals/expected_assists/
        # expected_goals_conceded/starts from ~2024-25 and
        # clearances_blocks_interceptions/tackles/recoveries only from
        # 2025-26 (DEFCON didn't exist as a scoring rule before then).
        # usecols tolerates a season's file not having a given column --
        # missing ones just come back all-NaN for that season after concat,
        # handled explicitly below rather than silently zero-filled.
        'expected_goals', 'expected_assists', 'expected_goals_conceded',
        'saves', 'bps', 'bonus', 'yellow_cards', 'red_cards', 'starts',
        'clearances_blocks_interceptions', 'tackles', 'recoveries',
        # Real outcomes the attack calibration (fit_attack_calibration in
        # training/train_component_models.py) regresses against --
        # expected_goals/expected_assists are the *input* signal, these are
        # what actually happened, available every season back to 2021-22.
        'goals_scored', 'assists',
    ]

    def _fetch_one_season(season_name, url):
        # pd.read_csv(url, ...) hands the URL straight to urllib under the
        # hood, which has no timeout of its own -- a stalled or throttled
        # connection to raw.githubusercontent.com can hang this
        # indefinitely (observed directly: multi-minute hangs with no
        # error, no progress, while a fresh single request to the same
        # host succeeds in under 2s). Fetch the bytes via requests with an
        # explicit timeout first, then hand the in-memory content to
        # read_csv, so a stalled connection fails fast instead of hanging
        # the whole pipeline.
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content = io.BytesIO(resp.content)
        try:
            df_season = pd.read_csv(content, usecols=needed_cols, engine='pyarrow')
        except Exception:
            content.seek(0)
            df_season = pd.read_csv(content, usecols=lambda c: c in needed_cols)
        df_season['season'] = season_name
        return df_season

    # Fetched concurrently, not one season at a time: a single slow or
    # throttled season (this data comes from raw.githubusercontent.com,
    # which has no SLA) used to serialize the whole function -- one slow
    # season made every season after it wait its turn too. Concurrency
    # bounds total wall-clock to roughly the slowest single season instead
    # of the sum of all of them; a season that still fails is just skipped,
    # same as the previous sequential try/except did.
    dfs = []
    with ThreadPoolExecutor(max_workers=len(seasons)) as pool:
        futures = {pool.submit(_fetch_one_season, name, url): name for name, url in seasons.items()}
        for future in as_completed(futures):
            try:
                dfs.append(future.result())
            except Exception:
                continue

    if not dfs:
        local_path = Path(__file__).resolve().parent.parent / "Data" / "Cleaned" / "model_ready_data.csv"
        if local_path.exists():
            return pd.read_csv(local_path)
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    num_cols = [
        'minutes', 'ict_index', 'creativity', 'influence', 'threat', 'total_points', 'value',
        'team_h_score', 'team_a_score',
        'expected_goals', 'expected_assists', 'expected_goals_conceded',
        'saves', 'bps', 'bonus', 'yellow_cards', 'red_cards', 'starts',
        'goals_scored', 'assists',
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # DEFCON's raw CBIT/CBIRT inputs deliberately NOT coerced/fillna(0) here
    # like num_cols above -- a season that never had this column (pre-
    # 2025-26, when DEFCON didn't exist as a scoring rule) must stay NaN,
    # not become a false "0 defensive actions this match", or the hit-rate
    # estimator below would count real professional footballers as having
    # recorded zero tackles/clearances/blocks/interceptions for four whole
    # seasons -- systematically dragging every pre-2025-26 walk-forward
    # estimate toward "never hits the threshold" for reasons that have
    # nothing to do with how the player actually played.
    defcon_raw_cols = ['clearances_blocks_interceptions', 'tackles', 'recoveries']
    for col in defcon_raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Team goals for/against *this fixture*, derived before the double-
    # gameweek aggregation below so both legs of a double gameweek get
    # summed into one gameweek total, same as points/minutes already are.
    df['team_goals_for'] = np.where(df['was_home'], df['team_h_score'], df['team_a_score'])
    df['team_goals_against'] = np.where(df['was_home'], df['team_a_score'], df['team_h_score'])

    # Double gameweeks give a player two fixture rows sharing the same
    # (name, season, GW) -- a team plays twice in one FPL gameweek. The old
    # drop_duplicates(keep='first') silently discarded the second fixture,
    # undercounting that gameweek's points/minutes by up to half. FPL's own
    # scoring sums both legs, so aggregate the same way: additive stats are
    # summed across legs, identity/display fields (team, position, value,
    # which fixture/opponent) just take the first leg -- a double gameweek
    # shows one representative fixture/opponent in the UI rather than two,
    # which is a display simplification, not a points error. The same
    # applies to team_goals_for/against now: both legs' goals are summed,
    # so a team's rolling goal stats still reflect everything they actually
    # played, even though the opponent-lookup below only resolves one of
    # the two opponents for that gameweek.
    additive_cols = [c for c in [
        'total_points', 'minutes', 'ict_index', 'creativity', 'influence', 'threat',
        'team_goals_for', 'team_goals_against',
        'expected_goals', 'expected_assists', 'expected_goals_conceded',
        'saves', 'bps', 'bonus', 'yellow_cards', 'red_cards', 'starts',
        'goals_scored', 'assists',
    ] if c in df.columns]
    # DEFCON raw cols summed separately with min_count=1: plain .sum() on an
    # all-NaN group (a pre-2025-26 season) silently returns 0.0, which is
    # exactly the "no data" -> "false 0 actions" bug this NaN-preservation
    # is for -- min_count=1 makes an all-NaN group sum to NaN instead.
    defcon_additive_cols = [c for c in defcon_raw_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in additive_cols + defcon_additive_cols + ['name', 'season', 'GW']]

    agg_map = {c: 'sum' for c in additive_cols}
    agg_map.update({c: (lambda s: s.sum(min_count=1)) for c in defcon_additive_cols})
    agg_map.update({c: 'first' for c in other_cols})

    df = df.groupby(['name', 'season', 'GW'], as_index=False).agg(agg_map)
    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

    # --- (a) Multi-horizon player rolling features ---
    # rolling_3_* keeps its exact original name (the shipped model was
    # trained on those 7 column names specifically) -- every other horizon
    # is a new, additional column, generated from ROLLING_HORIZONS rather
    # than hand-written per window.
    for stat in ROLLED_PLAYER_STATS:
        for window in ROLLING_HORIZONS:
            df[f'rolling_{window}_{stat}'] = _rolling_walk_forward(df, ['name', 'season'], stat, window)

    # --- (a2) Component-model raw stats, same mechanism, same horizons.
    # expected_goals/expected_assists/expected_goals_conceded/starts don't
    # exist before 2022-23 (bps/bonus/yellow_cards/red_cards/saves go back
    # to 2021-22) -- _rolling_walk_forward's fillna(0) treats an all-NaN
    # early window as "0 rate so far", the same cold-start behavior every
    # other rolling feature already has for a player's first few
    # gameweeks. That's an acceptable approximation here (it self-corrects
    # within a few gameweeks of real data); it is deliberately NOT used for
    # DEFCON's hit-rate below, where the missing-data span is 4 of 5 whole
    # seasons, not a few early gameweeks -- see the dedicated walk-forward
    # Beta-Binomial estimator further down.
    for stat in COMPONENT_PLAYER_STATS:
        for window in ROLLING_HORIZONS:
            df[f'rolling_{window}_{stat}'] = _rolling_walk_forward(df, ['name', 'season'], stat, window)

    # --- (a3) DEFCON hit-rate: walk-forward Beta-Binomial shrinkage, not a
    # rolling mean. "Historical threshold hit rate, not his mean rate" --
    # each match is a binary hit/miss against the real 2025-26+ rule
    # (DEF/GKP: clearances+blocks+interceptions+tackles >= 10; MID/FWD:
    # + recoveries >= 12), and a player's own cumulative hit/miss record
    # (shifted by 1, so a gameweek's estimate never sees its own outcome)
    # is blended with a position-level prior via Beta-Binomial shrinkage --
    # a player with 2 games of history shouldn't be treated as certain to
    # repeat a 100% or 0% hit rate the way a plain rolling mean would.
    # NaN wherever the raw CBIT/CBIRT columns are NaN (pre-2025-26, or this
    # gameweek didn't map -- see defcon_raw_cols above): those rows are
    # excluded from both the numerator and denominator, not counted as a
    # miss.
    cbit = df['clearances_blocks_interceptions'] + df['tackles']
    cbirt = cbit + df['recoveries']
    is_def_gk = df['position'].isin(['DEF', 'GKP', 'GK'])
    threshold_stat = np.where(is_def_gk, cbit, cbirt)
    threshold = np.where(is_def_gk, 10, 12)
    # NaN whenever the underlying raw stat itself is NaN (propagates
    # automatically through the arithmetic above) -- explicit here so the
    # comparison below doesn't silently coerce NaN to a boolean.
    df['defcon_hit'] = np.where(np.isnan(threshold_stat), np.nan, (threshold_stat >= threshold).astype(float))

    df = _defcon_walk_forward_hit_rate(df)

    # --- (b) Team and opponent rolling features + league rank ---
    # One row per (team, season, GW): every player on the same team in the
    # same gameweek carries an identical team_goals_for/against value
    # (it's a team-level fact), so 'first' here is exact, not an
    # approximation the way it is for display-only fields above.
    team_gw = (
        df.groupby(['team', 'season', 'GW'], as_index=False)[['team_goals_for', 'team_goals_against']]
        .first()
    )
    for stat in ROLLED_TEAM_STATS:
        for window in ROLLING_HORIZONS:
            team_gw[f'team_rolling_{window}_{stat}'] = _rolling_walk_forward(team_gw, ['team', 'season'], stat, window)

    # League rank from goal difference *so far this season* -- cumulative,
    # not windowed (a table position isn't a fixed-window average), shifted
    # by one gameweek so a team's rank going into GW g never includes GW g
    # itself. Rank 1 = best goal difference to date.
    team_gw['cum_gf'] = team_gw.groupby(['team', 'season'])['team_goals_for'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    team_gw['cum_ga'] = team_gw.groupby(['team', 'season'])['team_goals_against'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    team_gw['goal_diff_so_far'] = team_gw['cum_gf'] - team_gw['cum_ga']
    team_gw['league_rank'] = team_gw.groupby(['season', 'GW'])['goal_diff_so_far'].rank(ascending=False, method='min')

    team_feature_cols = [f'team_rolling_{w}_{s}' for s in ROLLED_TEAM_STATS for w in ROLLING_HORIZONS] + ['league_rank']

    # Own-team features: direct merge on (team, season, GW).
    df = df.merge(
        team_gw[['team', 'season', 'GW'] + team_feature_cols],
        on=['team', 'season', 'GW'], how='left'
    )

    # Opponent features: resolve opponent_team (numeric id) to the same
    # name-string namespace `team` uses, per season, then merge the same
    # team-level table again keyed by that resolved name -- i.e. the
    # opponent's *own* rolling stats, computed exactly like the player's
    # team's were.
    opp_frames = []
    for season_name, season_df in df.groupby('season'):
        id_to_name = _build_team_id_to_name_map(season_df)
        season_df = season_df.copy()
        season_df['opponent_name'] = season_df['opponent_team'].map(id_to_name)
        opp_frames.append(season_df)
    df = pd.concat(opp_frames, ignore_index=True).sort_values(['name', 'season', 'GW']).reset_index(drop=True)

    opp_lookup = team_gw[['team', 'season', 'GW'] + team_feature_cols].rename(
        columns={'team': 'opponent_name', **{c: f'opp_{c}' for c in team_feature_cols}}
    )
    df = df.merge(opp_lookup, on=['opponent_name', 'season', 'GW'], how='left')

    fill_cols = team_feature_cols + [f'opp_{c}' for c in team_feature_cols]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Squad depth: how many same-team, same-position players were
    # themselves regular starters (rolling_3_minutes >= 60) THAT
    # gameweek -- the historical counterpart of process_data()'s live
    # squad_depth_count, same definition (includes the player themselves,
    # so "no competition" reads as 1, not 0), just computed per
    # (season, GW, team, position) instead of from the live squad list.
    is_starter = df['rolling_3_minutes'] >= 60
    df['squad_depth_count'] = (
        df.assign(_is_starter=is_starter)
        .groupby(['season', 'GW', 'team', 'position'])['_is_starter']
        .transform('sum')
    ).fillna(0)

    return df


def fetch_full_history_with_understat():
    """
    fetch_full_history() enriched with Understat's xG/xA/xGChain/xGBuildup/
    key passes/shots (player) and xG/xGA/PPDA/PPDA-allowed/Deep/Deep-allowed
    (team + opponent) -- see src/understat.py.

    Deliberately a separate function rather than folding this into
    fetch_full_history() itself: that function is also used by the live
    Player History tab and Season Points Explorer (see app.py), neither of
    which need Understat data, and fetch_full_history() must keep working
    exactly as before -- unslowed, and with no new failure mode -- for
    those. Understat is a scraped site with no SLA; isolating its use to
    callers that actually asked for it (currently just
    evaluation/backtest.py) is what makes "the app still runs on FPL data
    alone if Understat is unreachable" actually true rather than aspirational.

    Returns (enriched_df, match_report) -- match_report is
    {season: {'player_match_rate': float, 'team_match_rate': float}},
    for the caller to verify against before trusting the new columns.
    """
    from src.understat import attach_understat_features

    df = fetch_full_history()
    if df.empty:
        return df, {}

    fpl_names_by_season = {}
    for season, season_df in df.groupby('season'):
        names = season_df['name'].dropna().unique()
        split = pd.Series(names).str.split(' ', n=1, expand=True)
        names_df = pd.DataFrame({
            'first_name': split[0],
            'second_name': split[1] if split.shape[1] > 1 else '',
        }).fillna('')
        # fetch_full_history()'s own "name" column (the vaastav join key)
        # is a full "First Last" name -- there's no real FPL web_name (the
        # short display name, usually the surname) in this historical
        # source to give match_player_name(). The surname (second_name) is
        # at least a semantically honest stand-in, unlike the previous
        # `= first_name` (a nonsensical duplicate of a field already used
        # elsewhere in the same call). NOTE: as of the code review that
        # caught this, src/understat.py::match_understat_players doesn't
        # actually read this column at all -- wiring it in was tried and
        # reverted after it measurably regressed the real match rate (see
        # that function's own comment) -- so this line is presently
        # cosmetic, not a match-rate fix, and is being kept only because
        # `= first_name` was actively misleading to read.
        names_df['web_name'] = names_df['second_name']
        fpl_names_by_season[season] = names_df

    return attach_understat_features(df, fpl_names_by_season)