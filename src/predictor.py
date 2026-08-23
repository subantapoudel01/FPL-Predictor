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
    3. Edge Case Guards & Mathematical Reasoning Tracking
    """
    features = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ]
    
    X = df[features].fillna(0)
    
    # 1. AI Prediction (Raw Potential)
    df['raw_xp'] = model.predict(X)
    
    def apply_expert_logic(row):
        prefix = "(Prior season baseline) " if is_preseason else ""
        
        # Check availability & API status flags
        status = str(row.get('status', 'a')).lower() if pd.notna(row.get('status')) else 'a'
        chance_val = row.get('chance_of_playing_next_round', row.get('chance_of_playing', 100))
        try:
            chance = float(chance_val) if pd.notna(chance_val) else 100.0
        except (ValueError, TypeError):
            chance = 100.0
            
        # Hard Unavailable Guard: 0% chance or status in ['i', 's', 'u'] (injured, suspended, unavailable)
        if chance == 0 or status in ['i', 's', 'u']:
            reasoning = f"{prefix}0.0 pts (Injured / Suspended / Unavailable)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])
            
        # A. Base Appearance Points
        base_points = 0.0
        starter_prob = float(row.get('starter_prob', 0)) if pd.notna(row.get('starter_prob')) else 0.0
        if starter_prob > 50:
            base_points = 2.0
        elif starter_prob > 10:
            base_points = 1.0
            
        # B. AI Performance Points
        raw_val = float(row.get('raw_xp', 0)) if pd.notna(row.get('raw_xp')) else 0.0
        ai_points = max(0.0, raw_val)
        
        # C. Fixture Difficulty Adjustments
        diff = float(row.get('next_match_difficulty', 3)) if pd.notna(row.get('next_match_difficulty')) else 3.0
        fixture_bonus = 0.0
        if diff == 1:
            fixture_bonus = 2.5     # vs Relegation fodder
        elif diff == 2:
            fixture_bonus = 1.5   # vs Weak team
        elif diff == 4:
            fixture_bonus = -0.5  # vs Strong team
        elif diff == 5:
            fixture_bonus = -1.0  # vs Title contender
        
        # D. Premium Captain Boost
        val = float(row.get('value', 0)) if pd.notna(row.get('value')) else 0.0
        val_m = val / 10.0 if val > 20 else val
        star_bonus = 0.0
        if val_m > 10.0 and diff <= 3:
            star_bonus = 1.5
            
        subtotal = base_points + ai_points + fixture_bonus + star_bonus
        
        # E. Clean Sheet Boost
        cs_bonus = 0.0
        if row.get('position') in ['DEF', 'GKP'] and diff <= 2:
            cs_bonus = 1.0
            subtotal += cs_bonus
            
        # F. Pep Tax (Rotation Risk)
        super_teams = ["MCI", "ARS", "LIV", "CHE", "Man City", "Arsenal", "Liverpool", "Chelsea"]
        rotation_tax = 0.0
        if row.get('team_name') in super_teams and row.get('position') in ['MID', 'FWD'] and val_m < 9.0:
            post_tax = subtotal * 0.75
            rotation_tax = subtotal - post_tax
            subtotal = post_tax
            
        # G. Selection & Partial Injury Doubt Multipliers
        rotation_factor = starter_prob / 100.0
        has_partial_injury = (0 < chance < 100)
        injury_factor = (chance / 100.0) if has_partial_injury else 1.0
        
        final_xp = subtotal * rotation_factor * injury_factor
        final_xp = float(np.clip(final_xp, 0.0, 18.0))
        
        # Build Reasoning String
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
        if has_partial_injury:
            parts.append(f"x {injury_factor:.2f} ({int(chance)}% injury doubt)")

        if not parts:
            reasoning = f"{prefix}{final_xp:.1f} pts"
        else:
            first = parts[0]
            rest = " ".join([p if p.startswith(("+", "-", "x")) else f"+ {p}" for p in parts[1:]])
            expr = f"{first} {rest}".strip()
            reasoning = f"{prefix}{final_xp:.1f} pts = {expr}"

        return pd.Series([final_xp, reasoning], index=['final_xp', 'reasoning'])

    results = df.apply(apply_expert_logic, axis=1)
    df['final_xp'] = results['final_xp'].fillna(0.0)
    df['reasoning'] = results['reasoning'].fillna(f"{'(Prior season baseline) ' if is_preseason else ''}0.0 pts")
    
    return df