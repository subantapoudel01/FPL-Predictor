"""
src/understat.py -- Understat as a fourth data source.

Two genuinely different fetch paths, because Understat itself splits the
data this way:

1. TEAM-level (xG, xGA, PPDA, PPDA-allowed, Deep, Deep allowed): scraped
   from `understat.com/getLeagueData/EPL/{year}` -- a clean internal JSON
   endpoint (found by inspecting the page's own network requests; the old
   "embedded JSON.parse('...') in the HTML" scraping recipe documented
   elsewhere no longer works, Understat now loads this client-side) that
   returns every team's full match-by-match history for that season in one
   request. Requires a Referer header or it 404s/redirects; no API key.

2. PLAYER-level (xG, xA, xGChain, xGBuildup, key passes, shots) per match:
   the same endpoint's `players` key is season-aggregate only, not
   match-by-match. Historical Understat data is already bundled, pre-
   scraped, in the vaastav FPL Historical Dataset this app already depends
   on for everything else (one CSV per player under
   data/{season}/understat/*.csv) -- reused here instead of scraping
   understat.com per-player (same request volume, strictly worse
   reliability, and we'd be scraping data understat.com itself doesn't
   expose match-by-match anyway).

Matching: Understat has no FPL id and no Opta code. Every player/team name
match here reuses match_player_name() from src/features.py -- the exact
same 5-tier cascade already validated for pre-season baseline matching --
rather than inventing a second, subtly different one. Every unmatched name
is logged (via `warnings`) instead of silently dropped, and the caller gets
a match rate back to decide whether the result is trustworthy.

Fails soft, everywhere: this module is scraping a public website with no
SLA, not calling an official API. Any network/parse failure at any stage
returns an empty result and logs a warning -- callers (src/features.py)
must keep working on FPL-only data if Understat is unreachable. Nothing
here ever raises out to the caller.

Caching: understat.com fetches are cached 24h (team data barely changes
mid-season and this is a scraped site -- hammering it is both rude and a
good way to get blocked). The vaastav player-file fetch is cached 7 days,
matching how the rest of this app treats historical data that's finished
changing once a season CSV is published.
"""
import io
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.features import _normalize_name, match_player_name, ROLLING_HORIZONS, _rolling_walk_forward

UNDERSTAT_BASE = "https://understat.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
GITHUB_API_BASE = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data"

# FPL season label -> Understat's season query param (the season's start year)
FPL_TO_UNDERSTAT_SEASON = {
    "2025-26": "2025", "2024-25": "2024", "2023-24": "2023", "2022-23": "2022", "2021-22": "2021",
}

PLAYER_STATS = ['xG', 'xA', 'xGChain', 'xGBuildup', 'key_passes', 'shots']
TEAM_STATS = ['xG', 'xGA', 'deep', 'deep_allowed', 'ppda', 'ppda_allowed']

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _warn(msg):
    warnings.warn(f"[understat] {msg}")


@st.cache_data(ttl=604800)
def _date_to_gw_map(fpl_season):
    """
    Understat's own match records carry a date but no FPL gameweek number.
    Built from vaastav's fixtures.csv for the same season (event = GW,
    kickoff_time -> calendar date), so both the team and player fetchers
    below can convert an Understat match date into the GW this whole
    pipeline is otherwise indexed by. Keyed by calendar date (not full
    timestamp) since Understat and FPL don't necessarily agree on kickoff
    time to the minute for the same match.

    Fails soft: returns {} on any error, which just means every Understat
    row for that season goes unmapped (logged by the caller) rather than
    the whole fetch crashing.
    """
    try:
        url = f"{GITHUB_RAW_BASE}/{fpl_season}/fixtures.csv"
        r = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        fx = pd.read_csv(io.StringIO(r.text), usecols=['event', 'kickoff_time'])
        fx = fx.dropna(subset=['event', 'kickoff_time'])
        fx['date'] = pd.to_datetime(fx['kickoff_time'], utc=True, errors='coerce').dt.strftime('%Y-%m-%d')
        return dict(zip(fx['date'], fx['event'].astype(int)))
    except Exception as e:
        _warn(f"could not build date->GW map for {fpl_season}: {e}")
        return {}


def fetch_understat_team_data(fpl_season):
    """
    Every team's full match-by-match xG/xGA/PPDA/PPDA-allowed/Deep/Deep-
    allowed history for one season, one request. Returns a DataFrame with
    columns: team (Understat's own team title), season, GW, xG, xGA, deep,
    deep_allowed, ppda, ppda_allowed -- already aggregated (summed for
    xG/xGA/deep counts, recombined att/def totals for ppda ratios so a
    double gameweek is one correct rate, not an average of two ratios) to
    one row per (team, GW). Empty DataFrame on any failure.
    """
    understat_season = FPL_TO_UNDERSTAT_SEASON.get(fpl_season)
    if understat_season is None:
        _warn(f"no Understat season mapping for {fpl_season!r} -- skipping team data")
        return pd.DataFrame()

    try:
        headers = {**_REQUEST_HEADERS, "Referer": f"{UNDERSTAT_BASE}/league/EPL/{understat_season}", "X-Requested-With": "XMLHttpRequest"}
        r = requests.get(f"{UNDERSTAT_BASE}/getLeagueData/EPL/{understat_season}", headers=headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        _warn(f"could not fetch team data for {fpl_season} ({understat_season}): {e}")
        return pd.DataFrame()

    date_to_gw = _date_to_gw_map(fpl_season)
    rows = []
    unmapped_dates = 0
    for team_obj in payload.get('teams', {}).values():
        team_name = team_obj.get('title', '')
        for m in team_obj.get('history', []):
            date_str = str(m.get('date', ''))[:10]
            gw = date_to_gw.get(date_str)
            if gw is None:
                unmapped_dates += 1
                continue
            ppda = m.get('ppda') or {}
            ppda_allowed = m.get('ppda_allowed') or {}
            rows.append({
                'team': team_name, 'season': fpl_season, 'GW': gw,
                'xG': float(m.get('xG') or 0), 'xGA': float(m.get('xGA') or 0),
                'deep': float(m.get('deep') or 0), 'deep_allowed': float(m.get('deep_allowed') or 0),
                'ppda_att': float(ppda.get('att') or 0), 'ppda_def': float(ppda.get('def') or 0),
                'ppda_allowed_att': float(ppda_allowed.get('att') or 0), 'ppda_allowed_def': float(ppda_allowed.get('def') or 0),
            })

    if unmapped_dates:
        _warn(f"{fpl_season}: {unmapped_dates} Understat team-match date(s) did not map to a GW (dropped)")
    if not rows:
        _warn(f"{fpl_season}: Understat team data fetched but zero rows mapped to a GW")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sum both legs of a double gameweek; PPDA/PPDA-allowed stay as summed
    # attacking/defensive-action totals here and get turned into a ratio
    # after aggregation, not before -- summing two ratios would double-
    # count the wrong thing.
    agg = df.groupby(['team', 'season', 'GW'], as_index=False).agg(
        xG=('xG', 'sum'), xGA=('xGA', 'sum'),
        deep=('deep', 'sum'), deep_allowed=('deep_allowed', 'sum'),
        ppda_att=('ppda_att', 'sum'), ppda_def=('ppda_def', 'sum'),
        ppda_allowed_att=('ppda_allowed_att', 'sum'), ppda_allowed_def=('ppda_allowed_def', 'sum'),
    )
    agg['ppda'] = agg['ppda_att'] / agg['ppda_def'].replace(0, np.nan)
    agg['ppda_allowed'] = agg['ppda_allowed_att'] / agg['ppda_allowed_def'].replace(0, np.nan)
    agg[['ppda', 'ppda_allowed']] = agg[['ppda', 'ppda_allowed']].fillna(0)
    return agg[['team', 'season', 'GW'] + TEAM_STATS]


# merged_gw.csv (the fetch_full_history() join key) uses FPL's short
# names, not full club names -- verified directly: for 2024-25 that's a
# normalized-substring match for 15/20 clubs (e.g. "Newcastle" is a
# substring of Understat's "Newcastle United"), but 5 are genuine
# contractions or nicknames sharing no substring with the full name at all
# ("Spurs" contains none of "Tottenham"; "Man City" isn't a substring of
# "Manchester City" because of where the break falls; "Wolves" isn't a
# substring of "Wolverhampton Wanderers" either). A small explicit alias
# table is the right tool here, unlike player names: the Premier League's
# club vocabulary is closed and tiny (~50 clubs have ever existed in it),
# where player identity is open-ended and constantly changing -- that's
# exactly why player matching earns a general fuzzy cascade and this
# doesn't. Keyed by normalized Understat name -> normalized FPL short name.
TEAM_NAME_ALIASES = {
    'manchestercity': 'mancity',
    'manchesterunited': 'manutd',
    'tottenham': 'spurs',
    'wolverhamptonwanderers': 'wolves',
    'nottinghamforest': 'nottmforest',
    'sheffieldunited': 'sheffieldutd',
    'westbromwichalbion': 'westbrom',
    'queensparkrangers': 'qpr',
    'newcastleunited': 'newcastle',
    'leicestercity': 'leicester',
}


def match_understat_teams(understat_team_df, fpl_team_names):
    """
    Understat uses full club names; FPL's own short names (what
    fetch_full_history() actually joins on) are often genuine contractions
    rather than substrings of the full name -- see TEAM_NAME_ALIASES.
    Normalized exact match, then the alias table, then a substring
    fallback for anything neither covers. Every miss logged. Returns
    understat_team_df with an added 'fpl_team' column (NaN where
    unmatched) and the match rate as a float.

    An empty `understat_team_df` reports rate 0.0, not 1.0 -- it almost
    always means the upstream Understat fetch failed or was rate-limited,
    not that there was genuinely nothing to match. Reporting 1.0 here once
    let a total fetch failure masquerade as a perfect match in the
    match_report a caller checks against the 90% gate.
    """
    if understat_team_df.empty:
        return understat_team_df, 0.0

    fpl_index = {_normalize_name(n): n for n in fpl_team_names}
    understat_names = understat_team_df['team'].unique()

    resolved = {}
    unmatched = []
    for name in understat_names:
        norm = _normalize_name(name)
        if norm in fpl_index:
            resolved[name] = fpl_index[norm]
            continue
        alias = TEAM_NAME_ALIASES.get(norm)
        if alias and alias in fpl_index:
            resolved[name] = fpl_index[alias]
            continue
        hit = next((fpl_name for fpl_norm, fpl_name in fpl_index.items() if norm in fpl_norm or fpl_norm in norm), None)
        if hit:
            resolved[name] = hit
        else:
            unmatched.append(name)

    if unmatched:
        _warn(f"unmatched Understat team name(s): {unmatched}")

    match_rate = 1.0 - (len(unmatched) / len(understat_names)) if len(understat_names) else 1.0
    understat_team_df = understat_team_df.copy()
    understat_team_df['fpl_team'] = understat_team_df['team'].map(resolved)
    return understat_team_df, match_rate


@st.cache_data(ttl=604800)
def _list_understat_player_files(fpl_season):
    """One GitHub API call listing every per-player Understat CSV for a season. Empty list on failure."""
    try:
        r = requests.get(f"{GITHUB_API_BASE}/{fpl_season}/understat", headers=_REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
        return [item['name'] for item in r.json() if item['name'].endswith('.csv')]
    except Exception as e:
        _warn(f"could not list Understat player files for {fpl_season}: {e}")
        return []


def _understat_filename_to_name(filename):
    """'Erling_Haaland_8260.csv' -> 'Erling Haaland' (drops the trailing Understat id)."""
    stem = filename[:-4] if filename.endswith('.csv') else filename
    parts = stem.split('_')
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return unquote(' '.join(parts))


def _fetch_one_player_file(fpl_season, filename):
    """
    `pd.read_csv(url, ...)` has no timeout of its own -- it hands the URL to
    urllib under the hood, which will hang indefinitely on a stalled or
    throttled connection. With up to 20 of these in flight at once against
    the same host, one rate-limited/slow connection can silently turn a
    ~2-minute fetch into a run that never finishes. Fetch the bytes via
    `requests` (explicit timeout) first, then hand the in-memory content to
    `pd.read_csv` -- so a stalled connection fails fast instead of hanging.
    """
    url = f"{GITHUB_RAW_BASE}/{fpl_season}/understat/{filename}"
    try:
        r = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), usecols=['date'] + PLAYER_STATS)
        df['understat_name'] = _understat_filename_to_name(filename)
        return df
    except Exception:
        return None


@st.cache_data(ttl=604800)
def fetch_understat_player_data(fpl_season, max_workers=20):
    """
    Every listed player's full match-by-match xG/xA/xGChain/xGBuildup/
    key_passes/shots for one season, fetched concurrently (one HTTP request
    per player -- vaastav has no combined file -- so this is genuinely the
    most expensive part of this module; cached 7 days precisely because of
    that cost). Returns one row per (understat_name, GW), summed across
    double-gameweek legs same as everywhere else in this pipeline.
    """
    filenames = _list_understat_player_files(fpl_season)
    if not filenames:
        return pd.DataFrame()

    date_to_gw = _date_to_gw_map(fpl_season)
    frames = []
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_player_file, fpl_season, fn): fn for fn in filenames}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                failed += 1
            else:
                frames.append(result)

    if failed:
        _warn(f"{fpl_season}: {failed}/{len(filenames)} Understat player file(s) failed to fetch")
    if not frames:
        _warn(f"{fpl_season}: zero Understat player files fetched successfully")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in PLAYER_STATS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['GW'] = df['date'].astype(str).str[:10].map(date_to_gw)
    unmapped = int(df['GW'].isna().sum())
    if unmapped:
        _warn(f"{fpl_season}: {unmapped} Understat player-match row(s) did not map to a GW (dropped)")
    df = df.dropna(subset=['GW']).copy()
    df['GW'] = df['GW'].astype(int)
    df['season'] = fpl_season

    agg = df.groupby(['understat_name', 'season', 'GW'], as_index=False)[PLAYER_STATS].sum()
    return agg


def match_understat_players(understat_player_df, fpl_names):
    """
    Matches each distinct Understat player name against the FPL roster for
    this season using the exact same 5-tier cascade as pre-season baseline
    matching (match_player_name in src/features.py) -- Understat has no
    Opta code, so tier 1 never fires here, which is expected, not a bug.

    `fpl_names` is a DataFrame with first_name/second_name/web_name columns
    (one row per distinct FPL player for the season being matched).

    Returns (understat_player_df with an 'fpl_name' column added, NaN
    where unmatched; match_rate as a float over distinct Understat names).
    Every unmatched name is logged, never silently dropped.

    An empty `understat_player_df` reports rate 0.0, not 1.0 -- see
    match_understat_teams()'s docstring for why: an empty input almost
    always means the Understat fetch itself failed (network error, GitHub
    API rate limit -- both observed in practice), and reporting a perfect
    match in that case would let a total outage silently pass the 90% gate
    a caller checks the match rate against.
    """
    if understat_player_df.empty:
        return understat_player_df, 0.0

    # name_index keyed by normalized full name -> canonical FPL `name`
    # (the exact join key fetch_full_history() uses), built the same way
    # _load_historical_baselines() builds its own index.
    #
    # NOTE: this docstring says fpl_names carries a web_name column, and
    # code review flagged that this function never actually reads it --
    # true, and *tried and reverted*: adding web_name-alone keys to
    # name_index (gated on uniqueness across the roster) measurably
    # REGRESSED the real match rate on 2024-25 data (35 unmatched vs. 33
    # before, zero net gains) -- extra index entries increased the odds of
    # a substring collision in match_player_name's separate mononym-
    # fallback tier (which counts *all* substring hits for a single-name
    # Understat player like "Thiago"), breaking two previously-unique
    # mononym matches for no offsetting improvement anywhere. Left
    # unused deliberately now, not silently -- see the code review this
    # comment traces back to for the verified before/after numbers.
    name_index = {}
    for _, r in fpl_names.iterrows():
        full = f"{r['first_name']} {r['second_name']}".strip()
        name_index[_normalize_name(full)] = full

    understat_names = understat_player_df['understat_name'].unique()
    resolved = {}
    unmatched = []
    for u_name in understat_names:
        parts = u_name.split(' ', 1)
        first = parts[0]
        second = parts[1] if len(parts) > 1 else ''
        matched_key = match_player_name(first, second, u_name, None, None, name_index)
        if matched_key is not None:
            resolved[u_name] = name_index[matched_key]
        else:
            unmatched.append(u_name)

    if unmatched:
        _warn(f"{len(unmatched)}/{len(understat_names)} Understat player(s) unmatched: {unmatched}")

    match_rate = 1.0 - (len(unmatched) / len(understat_names)) if len(understat_names) else 1.0
    understat_player_df = understat_player_df.copy()
    understat_player_df['fpl_name'] = understat_player_df['understat_name'].map(resolved)
    return understat_player_df, match_rate


def build_understat_rolling_features(player_df, team_df, main_df):
    """
    Attaches Understat-derived columns to `main_df` (the already-built,
    already-double-GW-aggregated player-gameweek table from
    fetch_full_history()) and rolls every stat at ROLLING_HORIZONS -- same
    mechanism as the FPL-native features (see _rolling_walk_forward),
    applied to player stats directly and to team stats for both the
    player's own team and their opponent (reusing the opponent_name
    resolution fetch_full_history() already computes).

    Returns main_df with the new columns added (0-filled wherever a player
    or team didn't match/wasn't found for a given row -- never NaN, so
    nothing downstream needs special-casing for "Understat data missing").
    """
    out = main_df.copy()

    # --- Player-level ---
    if not player_df.empty:
        player_keyed = player_df.dropna(subset=['fpl_name']).rename(columns={'fpl_name': 'name'})
        # Two different Understat names can in principle fuzzy-match to the
        # same FPL name (a false-positive collision) -- summing here
        # guarantees (name, season, GW) is unique before the merge below,
        # so a collision degrades to slightly-wrong stats for that one
        # player rather than silently multiplying rows for everyone
        # (a left merge on a duplicated key repeats every matching row).
        player_keyed = player_keyed.groupby(['name', 'season', 'GW'], as_index=False)[PLAYER_STATS].sum()
        out = out.merge(player_keyed, on=['name', 'season', 'GW'], how='left')
    for stat in PLAYER_STATS:
        if stat not in out.columns:
            out[stat] = 0.0
        out[stat] = out[stat].fillna(0.0)
        for window in ROLLING_HORIZONS:
            out[f'us_rolling_{window}_{stat}'] = _rolling_walk_forward(out, ['name', 'season'], stat, window)
        out.drop(columns=[stat], inplace=True)

    # --- Team-level (own team) ---
    team_feature_cols = []
    if not team_df.empty:
        # team_df carries BOTH Understat's own 'team' name and the resolved
        # 'fpl_team' -- drop the former before renaming the latter to
        # 'team', or the rename produces two columns both called 'team'
        # and every groupby/selection on 'team' below silently returns a
        # 2D slice instead of the column, breaking at the first groupby.
        team_gw = (
            team_df.dropna(subset=['fpl_team'])
            .drop(columns=['team'])
            .rename(columns={'fpl_team': 'team'})
            [['team', 'season', 'GW'] + TEAM_STATS]
            .copy()
        )
        # Same duplicate-key guard as the player merge above, in case two
        # Understat team names ever resolved to the same FPL team.
        team_gw = team_gw.groupby(['team', 'season', 'GW'], as_index=False)[TEAM_STATS].sum()
        for stat in TEAM_STATS:
            for window in ROLLING_HORIZONS:
                col = f'us_team_rolling_{window}_{stat}'
                team_gw[col] = _rolling_walk_forward(team_gw, ['team', 'season'], stat, window)
                team_feature_cols.append(col)

        out = out.merge(team_gw[['team', 'season', 'GW'] + team_feature_cols], on=['team', 'season', 'GW'], how='left')

        # --- Opponent (same resolved name fetch_full_history() built) ---
        if 'opponent_name' in out.columns:
            opp_lookup = team_gw[['team', 'season', 'GW'] + team_feature_cols].rename(
                columns={'team': 'opponent_name', **{c: f'opp_{c}' for c in team_feature_cols}}
            )
            out = out.merge(opp_lookup, on=['opponent_name', 'season', 'GW'], how='left')

    fill_cols = team_feature_cols + [f'opp_{c}' for c in team_feature_cols]
    if fill_cols:
        out[fill_cols] = out[fill_cols].fillna(0)

    return out


def attach_understat_features(main_df, fpl_names_by_season):
    """
    High-level entry point for src/features.py::fetch_full_history(). Runs
    per season present in `main_df` so one season's Understat outage
    doesn't take out the others, and fails soft to returning `main_df`
    completely unmodified (no partial/half-populated columns) if Understat
    is unreachable for every season -- the app must still run on FPL data
    alone regardless of what Understat does.

    `fpl_names_by_season`: dict of season -> DataFrame with
    first_name/second_name/web_name/name columns for that season's roster
    (distinct players), used for player matching.

    Returns (enriched_df, match_report) where match_report is a dict of
    season -> {'player_match_rate': float, 'team_match_rate': float}.
    """
    seasons = sorted(main_df['season'].unique())
    match_report = {}
    enriched_frames = []

    for season in seasons:
        season_main = main_df[main_df['season'] == season].copy()
        try:
            # match_understat_teams/match_understat_players each handle an
            # empty input themselves now (reporting rate 0.0, not a
            # misleadingly-perfect 1.0 -- see their docstrings), so no
            # separate empty-df branch is needed here.
            team_df = fetch_understat_team_data(season)
            fpl_team_names = season_main['team'].dropna().unique().tolist()
            team_df, team_rate = match_understat_teams(team_df, fpl_team_names)

            player_df = fetch_understat_player_data(season)
            names_df = fpl_names_by_season.get(season, pd.DataFrame(columns=['first_name', 'second_name', 'web_name']))
            player_df, player_rate = match_understat_players(player_df, names_df)

            enriched = build_understat_rolling_features(player_df, team_df, season_main)
            match_report[season] = {'player_match_rate': player_rate, 'team_match_rate': team_rate}
            enriched_frames.append(enriched)
        except Exception as e:
            _warn(f"Understat integration failed for {season}, continuing on FPL data alone for this season: {e}")
            match_report[season] = {'player_match_rate': 0.0, 'team_match_rate': 0.0, 'error': str(e)}
            enriched_frames.append(season_main)

    return pd.concat(enriched_frames, ignore_index=True), match_report
