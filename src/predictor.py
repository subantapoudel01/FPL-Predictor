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
    1. Linear Regression (Base Performance without appearance double-counting)
    2. Expert Rules (Context Awareness & Dynamic Squad Depth Rotation)
    3. Edge Case Guards & Mathematical Reasoning Tracking
    """
    features = [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ]
    
    X = df[features].fillna(0)
    
    # 1. AI Prediction (Raw Potential - inherently models baseline appearance)
    df['raw_xp'] = model.predict(X)
    
    # 2. Dynamic Squad Depth Calculation (Sprint 3)
    df['val_m'] = df['value'].apply(lambda v: (v / 10.0 if v > 20 else v) if pd.notna(v) else 0.0)
    status_series = df['status'] if 'status' in df.columns else pd.Series(['a'] * len(df), index=df.index)
    
    # Calculate active senior midfielders/attackers above average squad cost (val_m >= 5.5m)
    active_mask = (df['val_m'] >= 5.5) & (status_series.isin(['a', 'd']))
    attacker_counts = df[active_mask & df['position'].isin(['MID', 'FWD'])].groupby('team_name')['team_name'].count()
    # Clubs with dense positional competition (5+ active senior attackers above average cost) are flagged
    deep_squad_teams = set(attacker_counts[attacker_counts >= 5].index)
    
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
            
        # A. AI Performance Points (Raw prediction directly from LR without appearance double-counting)
        raw_val = float(row.get('raw_xp', 0)) if pd.notna(row.get('raw_xp')) else 0.0
        ai_points = max(0.0, raw_val)
        
        # B. Fixture Difficulty Adjustments (Rebalanced)
        diff = float(row.get('next_match_difficulty', 3)) if pd.notna(row.get('next_match_difficulty')) else 3.0
        fixture_bonus = 0.0
        if diff == 1:
            fixture_bonus = 1.25    # vs Relegation fodder
        elif diff == 2:
            fixture_bonus = 0.75    # vs Weak team
        elif diff == 4:
            fixture_bonus = -0.5    # vs Strong team
        elif diff == 5:
            fixture_bonus = -0.75   # vs Title contender
        
        # C. Premium Attacker & Captain Boost
        val_m = float(row.get('val_m', 0)) if pd.notna(row.get('val_m')) else 0.0
        premium_mult = 1.0
        if val_m >= 10.0 and row.get('position') in ['MID', 'FWD']:
            premium_mult = 1.15
            
        star_bonus = 0.0
        if val_m >= 10.0 and diff <= 3:
            star_bonus = 1.0
            
        subtotal = (ai_points + fixture_bonus) * premium_mult + star_bonus
        
        # D. Clean Sheet Boost (Reduced to +0.5 for DEF/GKP on easy fixtures FDR <= 2)
        cs_bonus = 0.0
        if row.get('position') in ['DEF', 'GKP'] and diff <= 2:
            cs_bonus = 0.5
            subtotal += cs_bonus

        # D2. Defensive Contribution (DEFCON) — 2025-26 scoring rule.
        # DEF/GKP earn 2 pts for combined CBIT (clearances+blocks+interceptions+tackles) >= 10/match.
        # MID/FWD earn 2 pts for combined CBIRT (+ recoveries) >= 12/match.
        # `defensive_contribution_per_90` is FPL's own season rate for that exact combined
        # count — verified directly against its raw components, not an approximation.
        # No match-by-match variance is available, so this is a stepped heuristic against
        # the season rate vs. threshold, consistent with the rest of this rule layer.
        defcon_bonus = 0.0
        dc_rate = float(row.get('defensive_contribution_per_90', 0.0)) if pd.notna(row.get('defensive_contribution_per_90')) else 0.0
        dc_threshold = 10.0 if row.get('position') in ['DEF', 'GKP'] else 12.0
        dc_ratio = dc_rate / dc_threshold if dc_threshold else 0.0
        if dc_ratio >= 1.0:
            defcon_bonus = 2.0      # Regularly clears the threshold most matches
        elif dc_ratio >= 0.75:
            defcon_bonus = 1.0      # Frequently close to / over the threshold
        elif dc_ratio >= 0.5:
            defcon_bonus = 0.4      # Occasional DEFCON returns
        if defcon_bonus > 0:
            subtotal += defcon_bonus

        # E. Dynamic Squad Depth Rotation Risk (Sprint 3)
        # Apply proportional rotation haircut only to non-guaranteed starters on high-competition squads
        starter_prob = float(row.get('starter_prob', 100)) if pd.notna(row.get('starter_prob')) else 100.0
        is_guaranteed_starter = (starter_prob >= 90) or (val_m >= 9.0)
        
        rotation_tax = 0.0
        if row.get('team_name') in deep_squad_teams and row.get('position') in ['MID', 'FWD'] and not is_guaranteed_starter:
            post_tax = subtotal * 0.75
            rotation_tax = subtotal - post_tax
            subtotal = post_tax
            
        # F. Selection & Partial Injury Doubt Multipliers
        rotation_factor = starter_prob / 100.0
        has_partial_injury = (0 < chance < 100)
        injury_factor = (chance / 100.0) if has_partial_injury else 1.0
        
        final_xp = subtotal * rotation_factor * injury_factor
        final_xp = float(np.clip(final_xp, 0.0, 18.0))
        
        # Build Reasoning String (Starting with base xP)
        parts = []
        parts.append(f"{ai_points:.1f} (base xP)")
        if fixture_bonus > 0:
            parts.append(f"+ {fixture_bonus:.2f} (easy fixture)")
        elif fixture_bonus < 0:
            parts.append(f"- {abs(fixture_bonus):.2f} (tough fixture)")
        if premium_mult > 1.0:
            parts.append(f"x {premium_mult:.2f} (premium attacker form)")
        if star_bonus > 0:
            parts.append(f"+ {star_bonus:.1f} (star captain)")
        if cs_bonus > 0:
            parts.append(f"+ {cs_bonus:.1f} (clean sheet)")
        if defcon_bonus > 0:
            parts.append(f"+ {defcon_bonus:.1f} (DEFCON)")
        if rotation_tax > 0:
            parts.append(f"- {rotation_tax:.1f} (Squad depth rotation risk)")
        if rotation_factor < 1.0 and rotation_factor > 0:
            parts.append(f"x {rotation_factor:.2f} (starter prob)")
        if has_partial_injury:
            parts.append(f"x {injury_factor:.2f} ({int(chance)}% injury doubt)")

        first = parts[0]
        rest = " ".join([p if p.startswith(("+", "-", "x")) else f"+ {p}" for p in parts[1:]])
        expr = f"{first} {rest}".strip()
        reasoning = f"{prefix}{final_xp:.1f} pts = {expr}"

        return pd.Series([final_xp, reasoning], index=['final_xp', 'reasoning'])

    results = df.apply(apply_expert_logic, axis=1)
    df['final_xp'] = results['final_xp'].fillna(0.0)
    df['reasoning'] = results['reasoning'].fillna(f"{'(Prior season baseline) ' if is_preseason else ''}0.0 pts")
    
    return df