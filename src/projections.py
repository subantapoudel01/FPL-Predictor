"""
src/projections.py -- multi-gameweek xP projection export, formatted for
the open-source FPL MILP solvers (solioanalytics/open-fpl-solver -- the
repo that used to live at sertalpbilal/FPL-Optimization-Tools; GitHub
confirms it's the same repository, renamed/transferred, not a fork). These
handle multi-gameweek planning, free-transfer valuation, chip strategy and
point-hit decisions -- exactly what this app's own "Rate My Team" tab
doesn't do, and none of which this app should build itself.

CSV format verified directly against that repo's own parser, not guessed
from its README (which documents solver *settings*, not the projection
file's column layout):
  - dev/data_parser.py::read_solio / read_fplreview -- both just
    pd.read_csv() the file, so the real contract lives in what
    dev/solver.py::prep_data() reads out of the result.
  - dev/solver.py::prep_data() merges the CSV to live FPL data on the
    `ID` column, normalizes `Pos` from either GKP/GK/DEF/MID/FWD or
    G/D/M/F, and requires one `{week}_Pts` column per gameweek in the
    horizon (week = the real, absolute FPL gameweek number -- prep_data()
    does `f"{week}_Pts"` for week in range(current_gw, current_gw +
    horizon), not a 1-indexed offset). It also sums every `{week}_xMins`
    column into a `total_min` it filters the ENTIRE player pool by
    (default cutoff 100 minutes) -- omitting `_xMins` isn't a hard error
    the way missing `_Pts` is, but it silently reads every non-locked
    player as 0 expected minutes and drops them from the pool.
  - `Name`/`Team`/`Value` aren't directly read back out of the CSV by
    prep_data() (it re-derives player identity from the live FPL merge),
    but they're part of the standard "FPL Review"-style format these
    readers are named after and are what a human opening the file expects
    -- included for that reason, not because the solver's own merge logic
    strictly requires them.

Why per-GW form features are frozen: attacking/defensive form
(rolling_3_minutes, rolling_3_ict_index, etc.) reflects a player's
season-to-date performance and doesn't change gameweek to gameweek within
a single export -- only fixture context (opponent, cs_probability, team/
opponent rolling goals) is genuinely GW-specific, and process_data()
already recomputes exactly that per `target_gw` (see its "3. FIXTURE
DIFFICULTY & OPPONENT MAPPING" section) without needing any change here.
The further out a projected gameweek is, the more this projection is
really just "current form against that fixture" rather than a genuine
forecast of how form will have moved by then -- stated in the UI next to
the download button (see app.py), not just in this docstring.
"""
import numpy as np
import pandas as pd

from src.features import process_data
from src.predictor import make_predictions

# Sources disagree on position codes (FPL Review uses GKP/DEF/MID/FWD,
# Solio uses G/D/M/F) -- prep_data() normalizes either on read, but
# exporting the single-letter form directly matches what dev/solver.py's
# own code comment says it expects ("Everything downstream expects the
# single-letter form").
POSITION_CODE = {'GKP': 'G', 'GK': 'G', 'DEF': 'D', 'MID': 'M', 'FWD': 'F'}


def _detect_start_gw(events):
    """
    Matches dev/solver.py::prep_data()'s own gameweek-detection exactly
    (`is_next`, full stop -- no is_current preference), NOT
    process_data()'s convention elsewhere in this app (is_current first,
    so the UI reads "GW3 Active" the moment GW3 kicks off rather than
    jumping straight to GW4).

    This divergence from the rest of the app is deliberate and was found
    by actually running the real prep_data() against a real export while
    a gameweek was live: process_data()'s is_current-first convention
    produced a CSV starting one gameweek earlier than prep_data() expects
    once a gameweek is under way (is_current=True, is_next=False), so
    prep_data() looked for a `{gw+horizon}_Pts` column one past the last
    one this export wrote and raised exactly the ValueError its own code
    raises for a genuinely-too-short horizon. Interop correctness with
    the tool this file exists to feed matters more here than matching
    process_data()'s own "what's live right now" convention.
    """
    active_event = next((e for e in events if e.get('is_next')), None)
    if active_event is None:
        active_event = next((e for e in events if not e.get('finished')), None)
    return active_event['id'] if active_event else 1


def build_projection_csv(raw_json, model, horizon=5):
    """
    Returns (DataFrame, gw_list). The DataFrame is ready to write straight
    to CSV for the solver's data/ folder: one row per player --
    ID/Name/Team/Pos/Value plus a `{gw}_Pts`/`{gw}_xMins` pair for each of
    the next `horizon` real gameweek numbers in `gw_list`.

    Blank gameweeks (a team with no fixture in a given target_gw) are
    reported as 0 Pts / 0 xMins for that column, not a fixture-blind
    guess -- without this, process_data() would fall back to a
    league-average fixture difficulty/opponent for a team that isn't
    actually playing that gameweek, which is worse than just saying 0.

    Known limitation, not handled here: a double gameweek (a team playing
    twice in one FPL gameweek) is scored as a single fixture, same as
    every other fixture-aware feature in this app -- process_data()
    resolves one opponent per team per gameweek. This under-counts a
    double-gameweek player's true expected total for that column; flagged
    in the UI next to the download button alongside the "frozen form"
    caveat, not silently absorbed into the number.
    """
    static = raw_json['static']
    fixtures = raw_json['fixtures']
    events = static.get('events', [])

    start_gw = _detect_start_gw(events)
    gw_list = list(range(start_gw, start_gw + horizon))

    teams_with_fixture = {
        gw: (
            {f['team_h'] for f in fixtures if f.get('event') == gw}
            | {f['team_a'] for f in fixtures if f.get('event') == gw}
        )
        for gw in gw_list
    }

    base_df = None
    per_gw_xp = {}
    per_gw_mins = {}

    for gw in gw_list:
        df_gw, _, is_preseason_gw = process_data(raw_json, target_gw=gw)
        df_gw = make_predictions(df_gw, model, is_preseason=is_preseason_gw)

        if base_df is None:
            base_df = df_gw[['id', 'web_name', 'team_name', 'position', 'value', 'team']].copy()

        blanking = ~df_gw['team'].isin(teams_with_fixture.get(gw, set()))
        xp = df_gw['final_xp'].where(~blanking, 0.0)
        # Expected minutes for the solver's xMins column, from the same
        # starter_prob signal process_data() already computes for the UI
        # (not a new estimate invented for this export).
        starter_prob = pd.to_numeric(df_gw.get('starter_prob', 0), errors='coerce').fillna(0) / 100.0
        exp_mins = (starter_prob * 90.0).where(~blanking, 0.0)

        per_gw_xp[gw] = xp.values
        per_gw_mins[gw] = exp_mins.values

    out = pd.DataFrame({
        'ID': base_df['id'].astype(int),
        'Name': base_df['web_name'],
        'Team': base_df['team_name'],
        'Pos': base_df['position'].map(POSITION_CODE).fillna(base_df['position']),
        'Value': base_df['value'],
    })
    for gw in gw_list:
        out[f'{gw}_Pts'] = np.round(per_gw_xp[gw], 2)
        out[f'{gw}_xMins'] = np.round(per_gw_mins[gw], 1)

    return out, gw_list
