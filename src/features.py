from pathlib import Path
import pandas as pd
import numpy as np
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
    
    relevant_fixtures = [f for f in fixtures if f.get('event') == target_gw]
    
    for f in relevant_fixtures:
        home_id = f['team_h']
        away_id = f['team_a']
        
        team_difficulty_map[home_id] = f['team_h_difficulty']
        team_difficulty_map[away_id] = f['team_a_difficulty']
        
        team_opponent_map[home_id] = f"{teams_map.get(away_id, '')} (H)"
        team_opponent_map[away_id] = f"{teams_map.get(home_id, '')} (A)"
    
    df['next_match_difficulty'] = df['team'].map(team_difficulty_map).fillna(3)
    df['next_opponent'] = df['team'].map(team_opponent_map).fillna("-") 
    
    # 4. Normalization / Pre-Season Baseline Fallback
    feature_cols = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'minutes_per_game'
    ]
    
    if is_preseason or completed_gws == 0:
        code_map, name_baselines, pos_medians, glob_medians, hist_cols = _load_historical_baselines()
        
        if name_baselines is not None and len(name_baselines) > 0:
            all_hist_norms = list(name_baselines.keys())
            
            def get_baseline_row(row):
                # 1. Permanent Opta Code match if present in both datasets
                code = row.get('code')
                if code_map and code in code_map:
                    hist_entry = code_map[code]
                    return pd.Series([float(hist_entry.get(col, 0)) for col in feature_cols], index=feature_cols)
                    
                # Robust normalized string matching
                fn = row.get('first_name', '')
                sn = row.get('second_name', '')
                wn = row.get('web_name', '')
                pos = row.get('position', 'MID')
                
                nf = _normalize_name(f"{fn} {sn}")
                nfw = _normalize_name(f"{fn} {wn}")
                nw = _normalize_name(wn)
                
                # 2. Normalized Full Name match
                if nf in name_baselines:
                    hist_entry = name_baselines[nf]
                    return pd.Series([float(hist_entry.get(col, 0)) for col in feature_cols], index=feature_cols)
                    
                # 3. Normalized First + Web Name match
                elif nfw in name_baselines:
                    hist_entry = name_baselines[nfw]
                    return pd.Series([float(hist_entry.get(col, 0)) for col in feature_cols], index=feature_cols)
                    
                # 4. Normalized Web Name match
                elif nw in name_baselines:
                    hist_entry = name_baselines[nw]
                    return pd.Series([float(hist_entry.get(col, 0)) for col in feature_cols], index=feature_cols)
                    
                # 5. Substring Name match
                else:
                    sec_norm = _normalize_name(sn)
                    fir_norm = _normalize_name(fn)
                    sub_m = [h for h in all_hist_norms if (fir_norm and fir_norm in h) and (sec_norm and sec_norm in h)]
                    if sub_m:
                        hist_entry = name_baselines[sub_m[0]]
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

    # 5. Availability
    df['chance_of_playing'] = df['chance_of_playing_next_round'].fillna(100)
    
    # 6. Starter Probability (Strict Curve)
    def calculate_starter_prob(mins_avg):
        if mins_avg >= 60: return 100
        elif mins_avg >= 45: return 75
        elif mins_avg >= 30: return 25
        else: return 5
            
    df['starter_prob'] = df['minutes_per_game'].apply(calculate_starter_prob)
    
    # Fill remaining NaNs defensively
    for col in feature_cols + ['starter_prob', 'chance_of_playing', 'next_match_difficulty', 'defensive_contribution_per_90', 'bonus', 'bps', 'price_change_today', 'ownership_pct']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    return df, target_gw, is_preseason

@st.cache_data(ttl=3600)
def fetch_full_history():
    """
    Fetches complete multi-season historical player gameweek datasets
    dynamically from Vaastav GitHub repository using PyArrow engine and column trimming.
    Cached for 1 hour -- short enough that the Last Gameweek Recap (see app.py)
    picks up a newly-published gameweek reasonably promptly, since this is the
    only place that data comes from now.
    """
    seasons = {
        "2025-26": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv",
        "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
        "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
        "2022-23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
    }
    
    needed_cols = [
        'name', 'GW', 'total_points', 'minutes', 'fixture', 'opponent_team',
        'was_home', 'ict_index', 'creativity', 'influence', 'threat', 'value',
        'team', 'position'
    ]
    
    dfs = []
    for season_name, url in seasons.items():
        try:
            try:
                df_season = pd.read_csv(url, usecols=needed_cols, engine='pyarrow')
            except Exception:
                df_season = pd.read_csv(url, usecols=lambda c: c in needed_cols)
                
            df_season['season'] = season_name
            dfs.append(df_season)
        except Exception:
            continue
            
    if not dfs:
        local_path = Path(__file__).resolve().parent.parent / "Data" / "Cleaned" / "model_ready_data.csv"
        if local_path.exists():
            return pd.read_csv(local_path)
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['name', 'season', 'GW'], keep='first')
    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)
    
    num_cols = ['minutes', 'ict_index', 'creativity', 'influence', 'threat', 'total_points', 'value']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    cols_to_roll = {
        'minutes': 'rolling_3_minutes',
        'ict_index': 'rolling_3_ict_index',
        'creativity': 'rolling_3_creativity',
        'influence': 'rolling_3_influence',
        'threat': 'rolling_3_threat',
        'total_points': 'rolling_3_total_points'
    }
    
    for src_col, target_col in cols_to_roll.items():
        df[target_col] = (
            df.groupby(['name', 'season'])[src_col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            .fillna(0)
        )
        
    return df