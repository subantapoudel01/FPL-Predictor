from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re

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
    
    # Detect if season has not started yet (all events finished == False)
    is_preseason = not any(e.get('finished', False) for e in events)
    completed_gws = sum(1 for e in events if e.get('finished', False))

    # 1. Identify Time
    current_real_gw = next((e['id'] for e in events if e.get('is_current')), None)
    if current_real_gw is None:
        current_real_gw = next((e['id'] for e in events if e.get('is_next')), 1)
        
    if target_gw is None:
        target_gw = current_real_gw
        
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
    for col in feature_cols + ['starter_prob', 'chance_of_playing', 'next_match_difficulty']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    return df, current_real_gw, is_preseason