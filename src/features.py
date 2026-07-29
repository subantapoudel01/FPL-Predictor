import pandas as pd
import numpy as np

def process_data(api_data, target_gw=None):
    """
    Converts raw API JSON into a Model-Ready DataFrame.
    """
    static = api_data['static']
    fixtures = api_data['fixtures']
    
    # 1. Identify Time
    # Try to find the current gameweek, but return None if it doesn't exist
    current_real_gw = next((e['id'] for e in static['events'] if e['is_current']), None)

    # Off-season fallback: If no current GW, look for the 'next' GW, or default to 1
    if current_real_gw is None:
        current_real_gw = next((e['id'] for e in static['events'] if e['is_next']), 1)
        
    # Set target_gw if none is provided
    if target_gw is None:
        target_gw = current_real_gw
        
    # Define normalization_gw to prevent division errors later
    normalization_gw = max(1, target_gw)
    
    df = pd.DataFrame(static['elements'])
    
    # 2. Map Team Names
    teams_map = {t['id']: t['short_name'] for t in static['teams']}
    df['team_name'] = df['team'].map(teams_map)

    # ==========================================
    # 3. FIXTURE DIFFICULTY & OPPONENT MAPPING
    # ==========================================
    team_difficulty_map = {}
    team_opponent_map = {} # Stores "vs ARS (H)"
    
    relevant_fixtures = [f for f in fixtures if f['event'] == target_gw]
    
    for f in relevant_fixtures:
        home_id = f['team_h']
        away_id = f['team_a']
        
        # Difficulty Map
        team_difficulty_map[home_id] = f['team_h_difficulty']
        team_difficulty_map[away_id] = f['team_a_difficulty']
        
        # Opponent Name Map
        # If I am Home Team, my opponent is Away Team Name
        team_opponent_map[home_id] = f"{teams_map[away_id]} (H)"
        # If I am Away Team, my opponent is Home Team Name
        team_opponent_map[away_id] = f"{teams_map[home_id]} (A)"
    
    df['next_match_difficulty'] = df['team'].map(team_difficulty_map).fillna(3)
    # Fill missing opponents (Blank GW) with "-"
    df['next_opponent'] = df['team'].map(team_opponent_map).fillna("-") 
    
    # 4. Normalization
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
    
    # Mappings
    df['position'] = df['element_type'].map({1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    df['value'] = df['now_cost'] / 10
    
    return df, current_real_gw