from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src import config
from src.components import predict_all_components

def load_model():
    """
    Loads the legacy single all-positions LinearRegression
    (Models/linear_reg_v1.pkl). The live app no longer serves predictions
    from this -- see load_position_models() -- it's kept on disk and
    loadable purely so evaluation/backtest.py can backtest it side by side
    against the current production model (see that script's --compare).
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "Models" / "linear_reg_v1.pkl"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None


def load_position_models():
    """
    Loads the current production model: Models/position_models_v1.pkl (see
    training/train_position_models.py) -- a dict of
    {'positions': {'GKP'/'DEF'/'MID'/'FWD': MedianEnsembleRegressor},
    'features': [...], 'meta': {...}}. Replaces the single all-positions
    LinearRegression because a goalkeeper's points come from saves and
    clean sheets while a forward's come from xG -- one linear model can't
    represent both, which is why minutes-played alone used to recover
    ~91% of that model's R^2. Trained on the same 7 features
    process_data() already computes live, so no live feature-engineering
    change was needed to serve it.
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "Models" / "position_models_v1.pkl"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

def load_component_models():
    """
    Loads Models/component_models_v1.pkl (see
    training/train_component_models.py) -- the fitted piece of the
    component prediction pipeline (src/components.py): logistic
    regressions for minutes/clean-sheet, linear calibrations for attack/
    goals-conceded/saves, and the bonus regression. Used when
    src/config.py::USE_COMPONENT_MODEL is True; the per-position ensemble
    (load_position_models) is the fallback otherwise.
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "Models" / "component_models_v1.pkl"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None


def load_active_model():
    """
    Loads whichever model src/config.py::USE_COMPONENT_MODEL selects --
    the single entry point app.py should call instead of choosing between
    load_component_models()/load_position_models() itself.
    """
    return load_component_models() if config.USE_COMPONENT_MODEL else load_position_models()


def ceiling_stars(ceiling_scores: pd.Series) -> pd.Series:
    """
    Star rating for the Top Captain Picks card, scaled against this
    gameweek's own top candidate rather than an absolute Ceiling Score
    threshold. An absolute cutoff meant every player who survived the
    captaincy eligibility filter (already a fairly high bar -- see
    `captain_eligible` above) cleared it too, so the whole column showed
    the same rating and carried no information. Relative-to-the-max always
    gives the top pick 5 stars and spreads the rest below it.
    """
    top = ceiling_scores.max()
    if not top or pd.isna(top):
        return pd.Series(3, index=ceiling_scores.index)
    return pd.cut(
        ceiling_scores / top,
        bins=[-0.01, 0.60, 0.72, 0.84, 0.93, 1.01],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

def make_predictions(df, model, is_preseason=False):
    """
    Dispatches to whichever prediction pipeline src/config.py's
    USE_COMPONENT_MODEL flag selects. Both paths return the same shape
    (final_xp, reasoning, ceiling_score, etc.) so app.py never needs to
    know which one ran.

    True (default): the component pipeline (_make_predictions_component)
    -- src/components.py predicts each scoring event separately (minutes,
    attacking, clean sheet, DEFCON, bonus, cards) and sums them through the
    real 2026/27 scoring matrix; the rule layer is cut down to real news
    overrides only.

    False: _make_predictions_position_ensemble, the previous architecture
    (one XGBoost+RandomForest ensemble per position, predicting
    total_points directly, plus a hand-tuned rule layer of fixture bonus/
    premium multiplier/star bonus/rotation tax on top). Kept as a config
    flag, not deleted, specifically so this can be rolled back in one line
    if the component pipeline ever backtests worse -- see
    evaluation/backtest.py's component-vs-position-ensemble comparison for
    the numbers that should decide which one is live.
    """
    if config.USE_COMPONENT_MODEL:
        return _make_predictions_component(df, model, is_preseason)
    return _make_predictions_position_ensemble(df, model, is_preseason)


def _make_predictions_component(df, model, is_preseason=False):
    """
    The current production pipeline. `model` is the bundle from
    load_component_models(): fitted minutes/clean-sheet classifiers and
    attack/goals-conceded/saves/bonus calibrations (see
    training/train_component_models.py). src/components.py::
    predict_all_components() does the actual per-scoring-event prediction
    and sum; this function's own job is just the rule layer, cut down to
    real news overrides -- injured/suspended/unavailable wipe to zero, plus
    a manual override flag (`manual_unavailable_override`, settable by the
    UI for a player the model has no way to know is out, e.g. a news story
    minutes old). The old fixture-difficulty bonus, premium multiplier,
    star-captain bonus, and rotation tax are gone -- the component models
    already learn fixture context (team/opponent rolling goals feed the
    clean-sheet and attack calibrations), price-correlated output (the
    calibrated attack rate itself), and rotation risk (squad_depth_count is
    a minutes-model input) from data, so re-applying a hand-tuned version
    of the same adjustment on top would double-count it.
    """
    components = predict_all_components(df, model)
    df = df.copy()
    for col in components.columns:
        df[col] = components[col]

    def apply_news_override(row):
        prefix = "(Prior season baseline) " if is_preseason else ""
        status = str(row.get('status', 'a')).lower() if pd.notna(row.get('status')) else 'a'
        chance_val = row.get('chance_of_playing_next_round', row.get('chance_of_playing', 100))
        try:
            chance = float(chance_val) if pd.notna(chance_val) else 100.0
        except (ValueError, TypeError):
            chance = 100.0
        manual_override = bool(row.get('manual_unavailable_override', False))

        if manual_override:
            reasoning = f"{prefix}0.0 pts (Manual override)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])
        if chance == 0 or status in ['i', 's']:
            reasoning = f"{prefix}0.0 pts (Injured/Suspended)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])
        if status == 'u':
            reasoning = f"{prefix}0.0 pts (Unavailable)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])

        raw_val = float(row.get('raw_xp', 0.0)) if pd.notna(row.get('raw_xp')) else 0.0
        final_xp = float(np.clip(raw_val, 0.0, 18.0))
        reasoning_text = str(row.get('reasoning', '') or '')
        if abs(final_xp - raw_val) > 0.05 and ' = ' in reasoning_text:
            # Clipping changed the total (a genuinely rare edge case --
            # heavy card/goals-conceded penalties outweighing everything
            # else, or the [0, 18] ceiling) -- keep the component
            # breakdown but make the leading number match what's actually
            # shown as final_xp everywhere else in the UI.
            reasoning_text = f"{final_xp:.1f} = " + reasoning_text.split(' = ', 1)[1]
        reasoning = f"{prefix}{reasoning_text}"
        return pd.Series([final_xp, reasoning], index=['final_xp', 'reasoning'])

    results = df.apply(apply_news_override, axis=1)
    df['final_xp'] = results['final_xp'].fillna(0.0)
    df['reasoning'] = results['reasoning'].fillna(f"{'(Prior season baseline) ' if is_preseason else ''}0.0 pts")

    df['val_m'] = df['value'].apply(lambda v: (v / 10.0 if v > 20 else v) if pd.notna(v) else 0.0)
    xgi90 = pd.to_numeric(df.get('expected_goal_involvements_per_90', 0.0), errors='coerce').fillna(0.0)
    df['xgi_per_90'] = xgi90
    df['captain_eligible'] = df['position'].isin(['MID', 'FWD']) | ((df['position'] == 'DEF') & (xgi90 > 0.30))

    pen_order = pd.to_numeric(df.get('penalties_order', np.nan), errors='coerce')
    df['is_penalty_taker'] = pen_order == 1
    price_mult = np.where(df['val_m'] >= 9.5, 1.25, 1.0)
    pen_mult = np.where(df['is_penalty_taker'], 1.15, 1.0)
    df['ceiling_score'] = df['final_xp'] * price_mult * pen_mult

    return df


def _make_predictions_position_ensemble(df, model, is_preseason=False):
    """
    Hybrid Prediction System (previous architecture, kept behind
    src/config.py::USE_COMPONENT_MODEL for rollback):
    1. Per-position ensemble (Base Performance without appearance double-counting)
    2. Expert Rules (Context Awareness & Dynamic Squad Depth Rotation)
    3. Edge Case Guards & Mathematical Reasoning Tracking

    `model` is the bundle from load_position_models(): one
    MedianEnsembleRegressor (XGBoost + RandomForest) per GKP/DEF/MID/FWD,
    replacing the single all-positions LinearRegression that used to be
    passed here -- a goalkeeper's points come from saves and clean sheets,
    a forward's from xG, and one linear model can't represent both (see
    src/predictor.py::load_position_models docstring).
    """
    features = model.get('features', [
        'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
        'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
    ])

    # 1. AI Prediction (Raw Potential - inherently models baseline appearance)
    # Routed per position -- each position has its own ensemble, so this
    # isn't a single model.predict() call over the whole frame.
    df['raw_xp'] = 0.0
    for position, ensemble in model['positions'].items():
        mask = df['position'] == position
        if not mask.any():
            continue
        X = df.loc[mask, features].fillna(0)
        df.loc[mask, 'raw_xp'] = ensemble.predict(X)
    
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
            
        # Hard Unavailable Guard: 0% chance, or status flags injury/suspension
        # ('u' -- left the league/on loan elsewhere -- is also a hard wipe;
        # it just doesn't get its own label since "Injured/Suspended" would
        # be inaccurate for it).
        if chance == 0 or status in ['i', 's']:
            reasoning = f"{prefix}0.0 pts (Injured/Suspended)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])
        if status == 'u':
            reasoning = f"{prefix}0.0 pts (Unavailable)"
            return pd.Series([0.0, reasoning], index=['final_xp', 'reasoning'])

        # A. AI Performance Points (Raw prediction directly from LR without appearance double-counting)
        raw_val = float(row.get('raw_xp', 0)) if pd.notna(row.get('raw_xp')) else 0.0
        ai_points = max(0.0, raw_val)

        # B. Fixture Difficulty Adjustments (Attackers only -- DEF/GKP now
        # get their fixture context through the dynamic clean sheet model
        # below instead of this flat, position-blind bonus)
        diff = float(row.get('next_match_difficulty', 3)) if pd.notna(row.get('next_match_difficulty')) else 3.0
        position = row.get('position')
        fixture_bonus = 0.0
        if position in ['MID', 'FWD']:
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
        if val_m >= 10.0 and position in ['MID', 'FWD']:
            premium_mult = 1.15

        star_bonus = 0.0
        if val_m >= 10.0 and diff <= 3:
            star_bonus = 1.0

        subtotal = (ai_points + fixture_bonus) * premium_mult + star_bonus

        # D. Dynamic Clean Sheet Odds (DEF/GKP only).
        # Replaces the old flat "+0.5 if FDR<=2" step function, which gave
        # every defender against a top-half team the same clean sheet value
        # regardless of how strong their own defence actually is. cs_probability
        # (see src/features.py) is a continuous 0.05-0.55 estimate from real
        # team-strength data, not a fixture-difficulty-band lookup -- a
        # struggling defence away at a mid-table side and a title-chasing
        # defence away at the same side now score visibly differently.
        cs_xp = 0.0
        cs_probability = 0.0
        if position in ['DEF', 'GKP']:
            cs_probability = float(row.get('cs_probability', 0.30)) if pd.notna(row.get('cs_probability')) else 0.30
            cs_xp = cs_probability * 4.0
            subtotal += cs_xp

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
        if cs_xp > 0:
            parts.append(f"+ {cs_xp:.1f} ({cs_probability * 100:.0f}% CS Odds)")
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

    # 3. Captaincy Ceiling Score & Eligibility
    # The plain final_xp ranking let budget defenders and low-ceiling
    # midfielders crowd the captaincy board just by being nailed starters
    # with an easy fixture -- fine for overall points, wrong for captaincy,
    # where variance/upside matters as much as the median outcome.
    #
    # Eligibility: attackers only, unless a defender's own real expected
    # goal involvement rate clears a genuine attacking threshold -- not a
    # proxy, `expected_goal_involvements_per_90` is FPL's own official xGI
    # rate for that player.
    xgi90 = pd.to_numeric(df.get('expected_goal_involvements_per_90', 0.0), errors='coerce').fillna(0.0)
    df['xgi_per_90'] = xgi90
    df['captain_eligible'] = df['position'].isin(['MID', 'FWD']) | ((df['position'] == 'DEF') & (xgi90 > 0.30))

    # Ceiling Score: rewards the two things that actually raise a captain's
    # ceiling above their median expected points -- being a genuine premium
    # asset (price is a reasonable proxy for a club's attacking hierarchy)
    # and being the designated penalty taker (a nailed-on extra scoring
    # route). `penalties_order == 1` is FPL's own field for first-choice
    # taker; deliberately not a hardcoded list of player names, which would
    # silently go stale the moment a transfer or injury changes who takes
    # penalties for a given club.
    pen_order = pd.to_numeric(df.get('penalties_order', np.nan), errors='coerce')
    df['is_penalty_taker'] = pen_order == 1
    price_mult = np.where(df['val_m'] >= 9.5, 1.25, 1.0)
    pen_mult = np.where(df['is_penalty_taker'], 1.15, 1.0)
    df['ceiling_score'] = df['final_xp'] * price_mult * pen_mult

    return df