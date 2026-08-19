from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def load_model():
    """Safely loads the .pkl model."""
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "Models" / "linear_reg_v1.pkl"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def make_predictions(df, model, is_preseason=False):
    """
    Hybrid Prediction System:
    1. Linear Regression (Base Performance)
    2. Expert Rules (Context Awareness)
    3. Mathematical Reasoning Tracking
    """
    if is_preseason:
        df['raw_xp'] = 0.0
        df['final_xp'] = 0.0
        df['reasoning'] = "0.0 pts (Season has not started)"
        return df

    features = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ]
    
    X = df[features].fillna(0)
    
    # 1. AI Prediction (Raw Potential)
    df['raw_xp'] = model.predict(X)
    
    def apply_expert_logic(row):
        # A. Base Appearance Points
        base_points = 0.0
        if row['starter_prob'] > 50:
            base_points = 2.0
        elif row['starter_prob'] > 10:
            base_points = 1.0
            
        # B. AI Performance Points
        raw_val = float(row['raw_xp'])
        ai_points = max(0.0, raw_val)
        
        # C. Fixture Difficulty Adjustments
        diff = row.get('next_match_difficulty', 3)
        fixture_bonus = 0.0
        if diff == 1:
            fixture_bonus = 2.5     # vs Relegation fodder
        elif diff == 2:
            fixture_bonus = 1.5   # vs Weak team
        elif diff == 4:
            fixture_bonus = -0.5  # vs Strong team
        elif diff == 5:
            fixture_bonus = -1.0  # vs Title contender
        
        # D. The "Premium Captain" Boost
        val_m = row['value'] / 10.0 if row['value'] > 20 else row['value']
        star_bonus = 0.0
        if val_m > 10.0 and diff <= 3:
            star_bonus = 1.5
            
        subtotal = base_points + ai_points + fixture_bonus + star_bonus
        
        # E. Position-Specific Logic (Clean Sheet Boost)
        cs_bonus = 0.0
        if row['position'] in ['DEF', 'GKP'] and diff <= 2:
            cs_bonus = 1.0
            subtotal += cs_bonus
        
        # F. Pep Roulette Tax (Rotation Risk)
        super_teams = ["MCI", "ARS", "LIV", "CHE", "Man City", "Arsenal", "Liverpool", "Chelsea"]
        rotation_tax = 0.0
        if row['team_name'] in super_teams and row['position'] in ['MID', 'FWD'] and val_m < 9.0:
            post_tax = subtotal * 0.75
            rotation_tax = subtotal - post_tax
            subtotal = post_tax
        
        # G. Injury & Selection Multipliers
        chance = row.get('chance_of_playing', 100)
        injury_factor = chance / 100.0
        rotation_factor = row['starter_prob'] / 100.0
        
        final_xp = subtotal * injury_factor * rotation_factor
        if chance == 0:
            final_xp = 0.0
            
        final_xp = float(np.clip(final_xp, 0.0, 18.0))
        
        # Build Reasoning String
        if chance == 0:
            reasoning = "0.0 pts (Injured / 0% chance of playing)"
        else:
            parts = []
            if base_points > 0:
                parts.append(f"{base_points:.1f} (appearance)")
            if ai_points > 0:
                parts.append(f"{ai_points:.1f} (base xP)")
            if fixture_bonus > 0:
                parts.append(f"+ {fixture_bonus:.1f} (easy fixture)")
            elif fixture_bonus < 0:
                parts.append(f"- {abs(fixture_bonus):.1f} (tough fixture)")
            if star_bonus > 0:
                parts.append(f"+ {star_bonus:.1f} (star captain)")
            if cs_bonus > 0:
                parts.append(f"+ {cs_bonus:.1f} (clean sheet)")
            if rotation_tax > 0:
                parts.append(f"- {rotation_tax:.1f} (rotation tax)")
            if rotation_factor < 1.0 and rotation_factor > 0:
                parts.append(f"x {rotation_factor:.2f} (starter prob)")

            if not parts:
                reasoning = f"{final_xp:.1f} pts"
            else:
                first = parts[0]
                rest = " ".join([p if p.startswith(("+", "-", "x")) else f"+ {p}" for p in parts[1:]])
                expr = f"{first} {rest}".strip()
                reasoning = f"{final_xp:.1f} pts = {expr}"

        return pd.Series([final_xp, reasoning], index=['final_xp', 'reasoning'])

    results = df.apply(apply_expert_logic, axis=1)
    df['final_xp'] = results['final_xp']
    df['reasoning'] = results['reasoning']
    
    return df