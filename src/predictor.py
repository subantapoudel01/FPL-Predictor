import joblib
import pandas as pd
import numpy as np
import os

def load_model():
    """Safely loads the .pkl model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "linear_reg_v1.pkl")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def make_predictions(df, model):
    """
    Hybrid Prediction System:
    1. Linear Regression (Base Performance)
    2. Expert Rules (Context Awareness)
    """
    features = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ]
    
    X = df[features].fillna(0)
    
    # 1. AI Prediction (Raw Potential)
    df['raw_xp'] = model.predict(X)
    
    def apply_expert_logic(row):
        # A. Base Appearance Points
        # (Only if they are likely to start)
        base_points = 0
        if row['starter_prob'] > 50: base_points = 2.0
        elif row['starter_prob'] > 10: base_points = 1.0
            
        # B. AI Performance Points
        ai_points = max(0, row['raw_xp'])
        
        # C. Fixture Difficulty Adjustments
        diff = row['next_match_difficulty']
        fixture_bonus = 0
        if diff == 1: fixture_bonus = 2.5     # vs Relegation fodder
        elif diff == 2: fixture_bonus = 1.5   # vs Weak team
        elif diff == 4: fixture_bonus = -0.5  # vs Strong team
        elif diff == 5: fixture_bonus = -1.0  # vs Title contender
        
        # D. The "Premium Captain" Boost
        star_bonus = 0
        if row['value'] > 10.0 and diff <= 3:
            star_bonus = 1.5
            
        # Combine Basic Score
        total_xp = base_points + ai_points + fixture_bonus + star_bonus
        
        # E. Position-Specific Logic
        # Defenders/Keepers rely on Clean Sheets, not just ICT Index
        if row['position'] in ['DEF', 'GKP']:
            # Boost if playing an easy team (Clean Sheet likelihood)
            if diff <= 2: 
                total_xp += 1.0
        
        # --- F. THE "PEP ROULETTE" TAX (REFINED) ---
        # Logic: Only tax CHEAP ATTACKERS at Big Clubs.
        # Do NOT tax Defenders/Keepers (they usually play 90 mins).
        
        super_teams = ["MCI", "ARS", "LIV", "CHE"] 
        
        # Check 1: Are they in a Super Team?
        if row['team_name'] in super_teams:
            # Check 2: Are they an ATTACKER? (MID or FWD)
            if row['position'] in ['MID', 'FWD']:
                # Check 3: Are they Cheap/Non-Premium? (< 9.0m)
                if row['value'] < 9.0:
                    # Apply Tax: Assume rotation risk
                    total_xp = total_xp * 0.75
        
        # G. Injury & Selection Filter
        # If API says "Injured" (0% chance), this kills the score to 0.
        injury_factor = row['chance_of_playing'] / 100.0
        
        # If they average 10 mins/game, this kills the score (benchwarmer).
        rotation_factor = row['starter_prob'] / 100.0
        
        return total_xp * injury_factor * rotation_factor

    df['final_xp'] = df.apply(apply_expert_logic, axis=1)
    
    # Final Cleanup
    df.loc[df['chance_of_playing'] == 0, 'final_xp'] = 0
    df['final_xp'] = df['final_xp'].clip(0, 18)
    
    return df