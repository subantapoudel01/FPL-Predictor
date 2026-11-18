"""
src/components.py -- the component prediction pipeline that replaced the
single points-regression + hand-tuned rule layer.

Each scoring event FPL actually pays out for is predicted separately, then
summed through the real 2026/27 scoring matrix, instead of one model
predicting `total_points` directly and a rule layer bolting fixture/
premium/rotation adjustments on top by feel. See
training/train_component_models.py for how each fitted piece
(Models/component_models_v1.pkl) is trained; this module is the shared
inference path both the live app (src/predictor.py) and the backtest
(evaluation/backtest.py) call, so "what the app predicts" and "what gets
backtested" can never drift apart.

Every function here takes a DataFrame carrying a fixed set of input
columns and returns Series aligned to that DataFrame's index -- it does not
care whether those columns came from process_data() (live) or
fetch_full_history() (historical walk-forward); see CLAUDE.md and
training/train_component_models.py's docstring for exactly which columns
are expected to mean the same thing on both sides, and where a live column
is a season-aggregate proxy for a historical per-match rolling one.

Points rates below are the real 2026/27 FPL scoring matrix (confirmed
unchanged from 2025-26 for DEFCON specifically; the rest have been stable
across recent seasons):
  - Appearance: 1 pt for any minutes played, +1 more (2 total) for 60+.
  - Goals: GKP/DEF 6, MID 5, FWD 4. Assists: 3, every position.
  - Clean sheet (60+ minutes, team concedes 0): GKP/DEF 4, MID 1, FWD 0.
  - Goals conceded: -1 per 2 conceded while on the pitch, GKP/DEF only.
  - Saves: GKP only, 1 pt per 3 saves.
  - DEFCON: +2, binary, DEF/GKP >=10 combined clearances+blocks+
    interceptions+tackles, MID/FWD >=12 including recoveries.
  - Cards: yellow -1, red -3.
"""
import numpy as np
import pandas as pd

GOAL_POINTS = {'GKP': 6, 'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
ASSIST_POINTS = 3.0
CS_POINTS = {'GKP': 4, 'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
SAVE_DIVISOR = 3.0
GC_DIVISOR = 2.0
DEFCON_POINTS = 2.0


def _numeric(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


# Horizon used to build each "_input" per-90 rate from the historical
# rolling_{w}_* columns -- rolling_5 (roughly a month of matches) as a
# compromise between recency (react to a genuine change in role/form) and
# stability (not swung by one big or blank match). A single documented
# choice, not tuned per component.
_RATE_HORIZON = 5


def derive_component_inputs(df):
    """
    Translates whichever raw columns are actually present -- historical
    rolling_{w}_* from fetch_full_history(), or live season-aggregate
    per-90 fields from process_data() -- into the fixed set of "_input"
    per-90 rate columns every predict_*() function in this module expects.
    Centralizing this here means the historical/live branching only has to
    be gotten right once, not separately inside every component.

    Historical: rolling_5_X is a per-*gameweek* sum (see
    fetch_full_history), so dividing by rolling_5_minutes and scaling by
    90 turns it into a genuine per-90 rate, matching what the live
    season-aggregate *_per_90 fields already are directly.

    Returns a NEW DataFrame (input df is not mutated) with the "_input"
    columns added alongside everything already on `df`.
    """
    out = df.copy()

    # Position one-hot, used as numeric features by the minutes/bonus
    # models -- 'GK' folded into 'GKP' since both spellings appear
    # depending on source (historical CSV vs live API).
    pos = df['position'].replace({'GK': 'GKP'})
    for p in ['GKP', 'DEF', 'MID', 'FWD']:
        out[f'is_{p.lower()}'] = (pos == p).astype(float)

    def _hist_rate(stat_col, minutes_col=f'rolling_{_RATE_HORIZON}_minutes', min_minutes=45.0, cap=3.0):
        """
        (num / mins * 90) blows up for a player whose last-5-game minutes
        history is tiny (a few late-substitute cameos) -- dividing a real
        stat total by a near-zero minutes sample extrapolates it to an
        absurd per-90 rate (observed directly: a max of 54 "expected goals
        per 90" before this guard existed, on a correlation with actual
        goals scored of 0.06 -- the extrapolation noise was drowning the
        real signal). Below `min_minutes` of rolling history, there simply
        isn't enough of a sample to extrapolate a rate from -- 0.0 (the
        same cold-start default every other rolling feature uses) until
        there is. `cap` is a generous, physically-implausible-to-exceed
        ceiling (no real player sustains 3 goals or 3 xG-worth of chances
        per 90) as a second guard against the rare case that clears
        min_minutes but is still a small, noisy sample.
        """
        if f'rolling_{_RATE_HORIZON}_{stat_col}' not in df.columns:
            return None
        num = df[f'rolling_{_RATE_HORIZON}_{stat_col}']
        mins = df.get(minutes_col, pd.Series(0.0, index=df.index))
        rate = (num / mins.replace(0, np.nan) * 90.0).fillna(0.0)
        return rate.where(mins >= min_minutes, 0.0).clip(upper=cap)

    # Attacking
    hist = _hist_rate('expected_goals')
    out['attack_xg90_input'] = hist if hist is not None else _numeric(df, 'expected_goals_per_90')
    hist = _hist_rate('expected_assists')
    out['attack_xa90_input'] = hist if hist is not None else _numeric(df, 'expected_assists_per_90')

    # Goals conceded -- a team can concede several in a bad match, so a
    # higher cap than the 3.0 default (a real 5-goal-per-90-averaged
    # concession rate would mean a team conceding 5+ every match for 5
    # straight matches, itself implausible, but closer to reality than a
    # single player's goal/assist rate).
    hist = _hist_rate('expected_goals_conceded', cap=6.0)
    out['gc_xgc90_input'] = hist if hist is not None else _numeric(df, 'expected_goals_conceded_per_90')

    # Saves -- goalkeepers can rack up far more than 3 saves in a busy
    # match, so the default cap would clip a genuinely busy keeper's real
    # rate; 15/90 is generous enough to never bind on a real match.
    hist = _hist_rate('saves', cap=15.0)
    out['saves90_input'] = hist if hist is not None else _numeric(df, 'saves_per_90')

    # Starts rate: historical rolling fraction of last-5 games started
    # (0-1); live season starts_per_90 as the closest available proxy --
    # same "how nailed-on is this player" concept, on a slightly different
    # normalization (a start is ~90 minutes by definition, so the two
    # scales are close in practice).
    if f'rolling_{_RATE_HORIZON}_starts' in df.columns:
        out['starts_rate_input'] = df[f'rolling_{_RATE_HORIZON}_starts']
    else:
        out['starts_rate_input'] = _numeric(df, 'starts_per_90')

    # Bonus: historical rolling BPS per gameweek; live has only a
    # season-cumulative `bps` total (no *_per_90 equivalent FPL exposes),
    # so divide by minutes/90 as a "matches played" proxy -- the same
    # normalization idea as minutes_per_game elsewhere, just from total
    # minutes instead of gameweek count (avoids needing completed_gws
    # threaded through here).
    if f'rolling_{_RATE_HORIZON}_bps' in df.columns:
        out['bps_rate_input'] = df[f'rolling_{_RATE_HORIZON}_bps']
    else:
        bps_total = _numeric(df, 'bps')
        match_equiv = (_numeric(df, 'minutes') / 90.0).clip(lower=1.0)
        out['bps_rate_input'] = bps_total / match_equiv

    # DEFCON live-only rate (historical rows use defcon_hit_prob directly,
    # computed in fetch_full_history() -- see predict_defcon)
    out['defcon_rate90_input'] = _numeric(df, 'defensive_contribution_per_90')

    # Cards: historical rolling card-points rate, already combined
    # (yellow -1, red -3) per COMPONENT_PLAYER_STATS' rolling of
    # yellow_cards/red_cards; live equivalent built from the season-total
    # counts / games played.
    if f'rolling_{_RATE_HORIZON}_yellow_cards' in df.columns:
        yc = df[f'rolling_{_RATE_HORIZON}_yellow_cards']
        rc = df[f'rolling_{_RATE_HORIZON}_red_cards']
        out['card_points_rate_input'] = -(yc * 1.0 + rc * 3.0)
    else:
        yc = _numeric(df, 'yellow_cards')
        rc = _numeric(df, 'red_cards')
        games = _numeric(df, 'starts', default=1.0).replace(0, 1.0)
        out['card_points_rate_input'] = -((yc * 1.0 + rc * 3.0) / games)

    return out


def predict_minutes(df, bundle):
    """
    P(plays this match) and P(60+ minutes) from the fitted logistic
    regressions in `bundle['minutes']` (see
    training/train_component_models.py::MINUTES_FEATURES for the exact
    input columns). Returns (p_plays, p_60plus, exp_minutes) -- exp_minutes
    is E[minutes] = p_plays * 90 * (a blend toward 60-90 for those who do
    play) is deliberately NOT modeled as a third regression: everything
    downstream only needs P(plays)/P(60+) and a *minutes fraction* to scale
    attacking/defensive rates by, and P(60+) already IS that fraction in
    all but the rare case of a player subbed in the 60-89th minute -- using
    p_60plus directly as the minutes-scaling fraction avoids a third,
    weakly-identified regression for a case that barely changes the
    expectation.
    """
    from training.train_component_models import MINUTES_FEATURES
    X = df[MINUTES_FEATURES].fillna(0)
    p_plays = bundle['minutes']['plays'].predict_proba(X)[:, 1]
    p_60plus = bundle['minutes']['plays60'].predict_proba(X)[:, 1]
    # P(60+) can't exceed P(plays) -- the classifiers are fit independently
    # and can disagree slightly near the boundary; clip rather than let a
    # downstream ratio exceed 1.
    p_60plus = np.minimum(p_60plus, p_plays)
    return pd.Series(p_plays, index=df.index), pd.Series(p_60plus, index=df.index)


def predict_attack(df, bundle, minutes_fraction):
    """
    Expected goals/assists this match = calibrated rolling xG90/xA90 rate
    (see training/train_component_models.py::fit_attack_calibration) x
    minutes_fraction, converted to points via GOAL_POINTS[position] and
    ASSIST_POINTS. Returns (exp_goals, exp_assists, points).
    """
    xg90 = _numeric(df, 'attack_xg90_input')
    xa90 = _numeric(df, 'attack_xa90_input')
    cal = bundle['attack_calibration']
    exp_goals90 = np.clip(cal['goals_slope'] * xg90 + cal['goals_intercept'], 0, None)
    exp_assists90 = np.clip(cal['assists_slope'] * xa90 + cal['assists_intercept'], 0, None)

    exp_goals = exp_goals90 * minutes_fraction
    exp_assists = exp_assists90 * minutes_fraction

    goal_rate = df['position'].map(GOAL_POINTS).fillna(4)
    points = exp_goals * goal_rate + exp_assists * ASSIST_POINTS
    return exp_goals, exp_assists, points


def predict_clean_sheet_and_gc(df, bundle, minutes_fraction):
    """
    P(clean sheet) from the fitted logistic regression in
    bundle['clean_sheet'] over team/opponent rolling goals-for/against (see
    training/train_component_models.py::CS_FEATURES). Expected goals
    conceded = calibrated rolling expected_goals_conceded90 x
    minutes_fraction (GC_FEATURES). Returns (p_cs, exp_goals_conceded,
    cs_points, gc_penalty_points) -- cs_points uses CS_POINTS[position]
    (0 for FWD, so this is a genuine no-op there, not a special case);
    gc_penalty_points is 0 outside GKP/DEF, per the real scoring rule.
    """
    from training.train_component_models import CS_FEATURES
    X = df[CS_FEATURES].fillna(0)
    p_cs = bundle['clean_sheet'].predict_proba(X)[:, 1]
    p_cs = pd.Series(p_cs, index=df.index)

    gc90_input = _numeric(df, 'gc_xgc90_input')
    cal = bundle['gc_calibration']
    exp_gc90 = np.clip(cal['slope'] * gc90_input + cal['intercept'], 0, None)
    exp_goals_conceded = exp_gc90 * minutes_fraction

    # Real FPL rule: clean sheet points require the player to have played
    # 60+ minutes -- p_cs on its own is a TEAM-level probability, blind to
    # whether *this* player (a nailed starter or a permanent benchwarmer)
    # is even going to be on the pitch. Without the minutes_fraction
    # scaling, every player at a strong defensive team got close to full
    # clean-sheet credit regardless of their own game time -- verified
    # directly as the dominant cause of a real regression (GKP MAE nearly
    # doubled backtesting this before the fix: backup goalkeepers who
    # rarely play were still credited close to a starter's clean-sheet
    # value purely from their team's clean-sheet probability).
    cs_rate = df['position'].map(CS_POINTS).fillna(0)
    cs_points = p_cs * cs_rate * minutes_fraction

    is_def_gk = df['position'].isin(['GKP', 'GK', 'DEF'])
    gc_penalty_points = np.where(is_def_gk, -(exp_goals_conceded / GC_DIVISOR), 0.0)

    return p_cs, exp_goals_conceded, cs_points, pd.Series(gc_penalty_points, index=df.index)


def predict_saves(df, bundle, minutes_fraction):
    """
    GKP only. Expected saves = calibrated rolling saves90 rate x
    minutes_fraction, /3 for points. 0 for every outfield position -- FPL
    doesn't award save points to them, so this is a genuine 0, not a
    missing-data default.
    """
    is_gk = df['position'].isin(['GKP', 'GK'])
    saves90_input = _numeric(df, 'saves90_input')
    cal = bundle['saves_calibration']
    exp_saves90 = np.clip(cal['slope'] * saves90_input + cal['intercept'], 0, None)
    exp_saves = np.where(is_gk, exp_saves90 * minutes_fraction, 0.0)
    points = exp_saves / SAVE_DIVISOR
    return pd.Series(exp_saves, index=df.index), pd.Series(points, index=df.index)


def predict_defcon(df, bundle, minutes_fraction):
    """
    P(hits the DEFCON threshold this match). Historical rows already carry
    `defcon_hit_prob` from fetch_full_history()'s own walk-forward
    Beta-Binomial estimator (that IS the model here -- no separate fit) --
    its cumulative hit/miss record already counts a non-playing gameweek
    as a miss, so it's already conditioned on this player's own appearance
    pattern; no further minutes scaling needed there. Live rows don't have
    match-by-match CBIT/CBIRT history to walk forward over, so they carry
    `defcon_rate90_input` (this season's defensive_contribution_per_90 --
    a rate *per 90 minutes actually played*, with no appearance
    information left in it) instead; predict_defcon converts that season
    rate into a per-match hit probability via a Poisson threshold-crossing
    estimate (P(count >= threshold) assuming match-to-match action counts
    are approximately Poisson(rate) -- standard for count data like
    tackles/clearances), then -- unlike the historical path -- multiplies
    by minutes_fraction, since a Poisson rate conditioned on "if they play
    90" says nothing about whether they'll be on the pitch at all this
    match (see predict_clean_sheet_and_gc's cs_points fix for the same
    class of bug: crediting a bench player as if their team's rate applies
    to them regardless of their own game time). Returns points too
    (P(hit) x DEFCON_POINTS).

    KNOWN GAP (see src/config.py::USE_COMPONENT_MODEL's comment): the
    value returned here feeds predict_bonus as `p_defcon`, but
    fit_bonus_model trains that feature exclusively against the
    historical branch's Beta-Binomial estimate -- the live branch's
    Poisson estimate is a different distribution shape for the same
    feature name, uncaught by the backtest since it never exercises this
    live branch. Not calibrated here; needs fixing before
    USE_COMPONENT_MODEL is ever flipped back on.
    """
    from scipy.stats import poisson
    threshold = np.where(df['position'].isin(['DEF', 'GKP', 'GK']), 10, 12)

    if 'defcon_hit_prob' in df.columns:
        p_hit = _numeric(df, 'defcon_hit_prob', default=np.nan)
    else:
        p_hit = pd.Series(np.nan, index=df.index)

    needs_poisson = p_hit.isna()
    if needs_poisson.any():
        rate = _numeric(df, 'defcon_rate90_input')
        poisson_p = poisson.sf(threshold - 1, np.clip(rate, 1e-6, None)) * minutes_fraction
        p_hit = p_hit.where(~needs_poisson, pd.Series(poisson_p, index=df.index))

    p_hit = p_hit.fillna(0.0).clip(0, 1)
    points = p_hit * DEFCON_POINTS
    return p_hit, points


def predict_bonus(df, bundle, p_defcon, minutes_fraction):
    """
    Expected bonus points from the fitted regression in bundle['bonus']
    over rolling BPS rate AND p_defcon together (see
    training/train_component_models.py::fit_bonus_model) -- 2026/27
    recalibrated BPS so heavy-DEFCON players collect less bonus in the
    same match, so p_defcon is a genuine input feature the regression can
    learn a negative coefficient on, not two independent draws summed.
    The regression's own intercept is a small constant applied to every
    row regardless of whether the player plays -- minutes_fraction scales
    it out for a bench player, the same fix applied to cs_points/DEFCON's
    live path for the same reason, just a smaller effect here (an
    intercept of ~0.02 points, not a multi-point clean-sheet rate).
    """
    from training.train_component_models import BONUS_FEATURES
    X = df[BONUS_FEATURES].copy()
    X['p_defcon'] = p_defcon.values if hasattr(p_defcon, 'values') else p_defcon
    X = X.fillna(0)
    points = np.clip(bundle['bonus'].predict(X), 0, 3) * minutes_fraction
    return pd.Series(points, index=df.index)


def predict_cards(df, minutes_fraction):
    """
    Small negative expected value from each player's own rolling
    card-points rate (yellow -1, red -3, already combined into one
    per-match points value historically -- see COMPONENT_PLAYER_STATS'
    rolling of `yellow_cards`/`red_cards` and how the live equivalent is
    built), scaled by minutes_fraction. No separate fit -- an empirical
    rolling rate is the whole model, same spirit as the DEFCON hit-rate
    but without the shrinkage (cards are common enough, and low-stakes
    enough, that a plain rolling rate is an adequate estimate here).
    """
    rate = _numeric(df, 'card_points_rate_input')
    return rate * minutes_fraction


def predict_all_components(df, bundle):
    """
    Runs every component above and returns a DataFrame of per-component
    expected points plus `raw_xp` (their sum) and `reasoning` (the
    plain-English breakdown string), aligned to `df`'s index. This is the
    one function both src/predictor.py and evaluation/backtest.py call --
    "what the app predicts" and "what gets backtested" run through the
    exact same code path.
    """
    df = derive_component_inputs(df)

    p_plays, p_60plus = predict_minutes(df, bundle)
    minutes_fraction = p_60plus  # see predict_minutes docstring

    appearance_points = p_plays + p_60plus

    exp_goals, exp_assists, attack_points = predict_attack(df, bundle, minutes_fraction)
    p_cs, exp_gc, cs_points, gc_penalty_points = predict_clean_sheet_and_gc(df, bundle, minutes_fraction)
    exp_saves, save_points = predict_saves(df, bundle, minutes_fraction)
    p_defcon, defcon_points = predict_defcon(df, bundle, minutes_fraction)
    bonus_points = predict_bonus(df, bundle, p_defcon, minutes_fraction)
    card_points = predict_cards(df, minutes_fraction)

    raw_xp = (
        appearance_points + attack_points + cs_points + gc_penalty_points
        + save_points + defcon_points + bonus_points + card_points
    )

    out = pd.DataFrame({
        'p_plays': p_plays, 'p_60plus': p_60plus, 'appearance_points': appearance_points,
        'exp_goals': exp_goals, 'exp_assists': exp_assists, 'attack_points': attack_points,
        'p_cs': p_cs, 'exp_goals_conceded': exp_gc, 'cs_points': cs_points, 'gc_penalty_points': gc_penalty_points,
        'exp_saves': exp_saves, 'save_points': save_points,
        'p_defcon': p_defcon, 'defcon_points': defcon_points,
        'bonus_points': bonus_points, 'card_points': card_points,
        'raw_xp': raw_xp,
    }, index=df.index)

    out['reasoning'] = _build_reasoning(df, out)
    return out


def _build_reasoning(df, out):
    """
    Plain-English breakdown matching the format specified: every number in
    it is one of the component values just computed, not a hand-written
    rule-layer string. Terms under 0.05 points are omitted so the string
    stays readable (a genuinely-zero save/bonus/card contribution for an
    outfield player shouldn't clutter every row).
    """
    def _fmt_row(i):
        parts = [f"{out.at[i, 'appearance_points']:.1f} appearance"]

        atk = out.at[i, 'attack_points']
        if abs(atk) >= 0.05:
            parts.append(
                f"{atk:.1f} attack ({out.at[i, 'exp_goals']:.2f} xG, {out.at[i, 'exp_assists']:.2f} xA)"
            )

        cs = out.at[i, 'cs_points']
        if abs(cs) >= 0.05:
            parts.append(f"{cs:.1f} CS ({out.at[i, 'p_cs'] * 100:.0f}%)")

        gc = out.at[i, 'gc_penalty_points']
        if abs(gc) >= 0.05:
            parts.append(f"{gc:.1f} GC ({out.at[i, 'exp_goals_conceded']:.2f} xGC)")

        sv = out.at[i, 'save_points']
        if abs(sv) >= 0.05:
            parts.append(f"{sv:.1f} saves ({out.at[i, 'exp_saves']:.1f})")

        dc = out.at[i, 'defcon_points']
        if abs(dc) >= 0.05:
            parts.append(f"{dc:.1f} DEFCON ({out.at[i, 'p_defcon'] * 100:.0f}% x 2)")

        bp = out.at[i, 'bonus_points']
        if abs(bp) >= 0.05:
            parts.append(f"{bp:.1f} bonus")

        cp = out.at[i, 'card_points']
        if abs(cp) >= 0.05:
            parts.append(f"{cp:.1f} cards")

        total = out.at[i, 'raw_xp']
        return f"{total:.1f} = " + " + ".join(parts).replace("+ -", "- ")

    return pd.Series([_fmt_row(i) for i in out.index], index=out.index)
