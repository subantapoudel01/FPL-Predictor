import time
import requests
import pandas as pd
import streamlit as st

# API Endpoints
API_BASE_URL = "https://fantasy.premierleague.com/api"

# --- FIELD WHITELISTS ---
# Every name here must exist on the live API record it trims, or the field
# is either a typo or FPL renamed/removed it -- in either case the app would
# otherwise silently read it as missing/None (a 0-looking value, no error)
# rather than fail loudly. This has actually happened four times: DEFCON,
# bonus/price/ownership, penalty/xGI, and fixture status all shipped
# correctly-written code that produced zeros because nobody added the field
# here first. _assert_whitelist_subset() below is what turns the fifth
# occurrence of this mistake into an immediate KeyError instead of a silent
# zero. Module-level (not local to the fetch functions) so tests/test_whitelist.py
# can import and check these directly against the live API.
ELEMENT_KEYS = [
    'id', 'code', 'first_name', 'second_name', 'web_name', 'element_type',
    'team', 'now_cost', 'minutes', 'form', 'ict_index', 'creativity',
    'influence', 'threat', 'chance_of_playing_next_round', 'status',
    'defensive_contribution_per_90',
    'bonus', 'bps', 'cost_change_event', 'selected_by_percent',
    'penalties_order', 'expected_goal_involvements_per_90',
    'total_points', 'goals_scored', 'assists',
    # Added for the component-model prediction pipeline (src/components.py)
    # -- each of these is the live, season-aggregate "per 90" or season-total
    # equivalent of a per-match stat the historical CSV pipeline (see
    # fetch_full_history) rolls forward walk-forward. Kept in the same
    # naming spirit as the existing `rolling_3_minutes = minutes_per_game`
    # live proxy: no per-gameweek live history is fetched per player, these
    # season-aggregate rates stand in for it.
    'starts', 'starts_per_90',
    'expected_goals_per_90', 'expected_assists_per_90',
    'expected_goals_conceded_per_90', 'saves_per_90', 'clean_sheets_per_90',
    'clearances_blocks_interceptions', 'tackles', 'recoveries',
    'defensive_contribution', 'yellow_cards', 'red_cards',
]
# strength_attack_home/away and strength_defence_home/away exist on the API
# but read 0 for every team early in a season (FPL's own algorithm hasn't
# populated them yet) -- pulled anyway so the cs_probability calc in
# features.py can prefer them the moment they go live, but
# strength_overall_home/away (a coarser 2-5 scale, always populated) is
# what it actually uses today.
TEAM_KEYS = [
    'id', 'short_name', 'name',
    'strength_overall_home', 'strength_overall_away',
    'strength_attack_home', 'strength_attack_away',
    'strength_defence_home', 'strength_defence_away'
]
EVENT_KEYS = ['id', 'name', 'is_current', 'is_next', 'finished', 'deadline_time']
FIXTURE_KEYS = [
    'id', 'event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty',
    'kickoff_time', 'started', 'finished', 'finished_provisional', 'minutes',
    'team_h_score', 'team_a_score'
]


def _assert_whitelist_subset(record, whitelist, label):
    """
    Raises KeyError if any whitelist entry isn't a real key on `record` (the
    first raw record of that type from the API). Catches a typo or an FPL
    field rename at fetch time, loudly, instead of the field silently
    becoming None/0 downstream with nothing to indicate why.
    """
    missing = [k for k in whitelist if k not in record]
    if missing:
        raise KeyError(
            f"{label} whitelist in src/data_loader.py references field(s) not present "
            f"on the live API record: {missing}. Either it's a typo, or FPL renamed/"
            f"removed the field -- check the raw record before fixing the whitelist."
        )


@st.cache_data(ttl=3600)
def fetch_static_data():
    """
    Fetches bootstrap-static/ (players, teams, gameweek events) and trims it
    to an explicit field whitelist -- reduces memory and payload exposure,
    but also means any field used anywhere else in the app that isn't listed
    in ELEMENT_KEYS/TEAM_KEYS/EVENT_KEYS above silently reads as missing/None.
    Verify a new field's raw value before wiring logic on top of it; add it
    to the relevant whitelist above first.

    Cached 1 hour: player prices/status and gameweek boundaries don't change
    fast enough to need anything shorter.

    Returns: dict with 'elements'/'teams'/'events' keys, or None on failure.
    """
    try:
        r = requests.get(f"{API_BASE_URL}/bootstrap-static/", timeout=10)
        r.raise_for_status()
        static_data = r.json()

        elements = static_data.get('elements', [])
        if elements:
            _assert_whitelist_subset(elements[0], ELEMENT_KEYS, "ELEMENT_KEYS")
        trimmed_elements = [{k: elem.get(k) for k in ELEMENT_KEYS if k in elem} for elem in elements]

        teams = static_data.get('teams', [])
        if teams:
            _assert_whitelist_subset(teams[0], TEAM_KEYS, "TEAM_KEYS")
        trimmed_teams = [{k: team.get(k) for k in TEAM_KEYS if k in team} for team in teams]

        events = static_data.get('events', [])
        if events:
            _assert_whitelist_subset(events[0], EVENT_KEYS, "EVENT_KEYS")
        trimmed_events = [{k: event.get(k) for k in EVENT_KEYS if k in event} for event in events]

        return {
            "elements": trimmed_elements,
            "teams": trimmed_teams,
            "events": trimmed_events
        }
    except requests.RequestException as e:
        print(f"Error fetching static data: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_fixtures_data():
    """
    Fetches fixtures/ and trims it to FIXTURE_KEYS above (same caveat as
    fetch_static_data -- add a field there before using it elsewhere).
    Cached only 60 seconds, not the 3600s used for bootstrap-static:
    `started`/`finished`/`finished_provisional`/`minutes` change during live
    matches, and a stale `finished=False` read for up to an hour is exactly
    what caused the fixture ticker to show "LIVE" on matches that had
    already ended.

    Returns: list of trimmed fixture dicts, or None on failure.
    """
    try:
        r = requests.get(f"{API_BASE_URL}/fixtures/", timeout=10)
        r.raise_for_status()
        fixtures_data = r.json()

        dict_fixtures = [f for f in fixtures_data if isinstance(f, dict)]
        if dict_fixtures:
            _assert_whitelist_subset(dict_fixtures[0], FIXTURE_KEYS, "FIXTURE_KEYS")
        trimmed_fixtures = [{k: fix.get(k) for k in FIXTURE_KEYS if k in fix} for fix in dict_fixtures]

        return trimmed_fixtures
    except requests.RequestException as e:
        print(f"Error fetching fixtures data: {e}")
        return None


def fetch_api_data():
    """
    Combines fetch_static_data() and fetch_fixtures_data() into the single
    {'static': ..., 'fixtures': ...} shape the rest of the app expects.
    Each half is cached independently at its own TTL (see above) -- this
    wrapper itself must NOT be cached, or the fixtures half's 60s TTL is
    defeated by whatever longer TTL wraps this function.
    """
    static_data = fetch_static_data()
    if static_data is None:
        return None
    return {
        "static": static_data,
        "fixtures": fetch_fixtures_data() or []
    }

def fetch_gameweek_history(gw_id):
    """
    Fetches live performance stats for a completed Gameweek.
    Used for the 'Reality Check' (Backtesting) feature.
    
    Args:
        gw_id (int): The gameweek number to query (e.g., 5).
        
    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'actual_points'.
    """
    url = f"{API_BASE_URL}/event/{gw_id}/live/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: 
            return None
            
        data = r.json()
        stats = []
        
        # Extract points for every player in that GW
        for player in data['elements']:
            stats.append({
                'id': player['id'],
                'actual_points': player['stats']['total_points'],
                'goals': player['stats']['goals_scored'],
                'assists': player['stats']['assists']
            })
            
        return pd.DataFrame(stats)
    except requests.RequestException:
        return None

@st.cache_data(ttl=600)
def fetch_player_live_summary(player_id):
    """
    Fetches live performance stats for an individual player from FPL element-summary API.
    Cached 10 minutes -- one call per player looked at, not per rerun.

    Args:
        player_id (int): FPL element ID
    Returns:
        dict: Contains 'history' (played GWs) and 'fixtures' (upcoming schedule).
    """
    url = f"{API_BASE_URL}/element-summary/{player_id}/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None

def fetch_user_team(team_id, current_gw):
    """
    Fetches the live 15-player squad picks for a given FPL Team ID.
    Queries event/{current_gw}/picks/ and falls back to prior gameweeks if unreleased.
    
    Args:
        team_id (int): FPL entry/manager ID
        current_gw (int): Current or upcoming target gameweek
        
    Returns:
        dict: Contains 'gw', 'picks' (15 player dicts), and 'active_chip', or None if unavailable.
    """
    if not team_id or team_id < 1:
        return None
        
    gws_to_try = [current_gw] + [gw for gw in range(current_gw - 1, 0, -1)]
    
    for gw in gws_to_try:
        url = f"{API_BASE_URL}/entry/{team_id}/event/{gw}/picks/"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if 'picks' in data and len(data['picks']) > 0:
                    return {
                        'gw': gw,
                        'active_chip': data.get('active_chip'),
                        'picks': data['picks']
                    }
        except requests.RequestException:
            continue

    return None


@st.cache_data(ttl=900)
def fetch_league_standings(league_id, max_pages=10):
    """
    Fetches leagues-classic/{id}/standings/, paginating via
    ?page_standings=N until the API's own `has_next` flag goes False or
    `max_pages` is hit (50 entries/page, so 10 pages = 500 managers --
    plenty for a friend-group mini-league; a genuinely huge league is
    reported as truncated rather than fetched to exhaustion, which is
    both slow and outside what this tab is for).

    Cached 15 minutes: standings and squads don't move fast enough
    within a gameweek to need fresher, and this endpoint is
    rate-limited -- a 0.3s pause between pages (only on an actual cache
    miss; a cache hit never re-enters this function body at all) keeps
    repeated page requests from firing back-to-back.

    Returns None for a private league (403) or a league ID that doesn't
    exist (404) or any other failure -- callers show a clear message
    instead of a stack trace, they don't get a dict to misread. On
    success: {'league': {...name/admin info...}, 'standings': [...one
    dict per entry...], 'truncated': bool}.
    """
    if not league_id or league_id < 1:
        return None

    all_results = []
    league_info = None
    truncated = False

    for page in range(1, max_pages + 1):
        url = f"{API_BASE_URL}/leagues-classic/{league_id}/standings/"
        try:
            r = requests.get(url, params={"page_standings": page}, timeout=10)
        except requests.RequestException:
            return None

        if r.status_code in (403, 404):
            return None
        if r.status_code != 200:
            return None

        data = r.json()
        if league_info is None:
            league_info = data.get('league', {})
        standings_block = data.get('standings', {})
        all_results.extend(standings_block.get('results', []))

        if not standings_block.get('has_next', False):
            break
        if page < max_pages:
            time.sleep(0.3)
    else:
        # Loop exhausted max_pages without has_next ever going False --
        # there was more standings data than we fetched.
        truncated = True

    if league_info is None:
        return None

    return {'league': league_info, 'standings': all_results, 'truncated': truncated}


@st.cache_data(ttl=900)
def fetch_entry_picks(entry_id, gw):
    """
    Fetches entry/{entry_id}/event/{gw}/picks/ for one manager -- the
    per-entry counterpart to fetch_league_standings, used to build a
    mini-league's projected points/captaincy/effective-ownership view.
    Same 15-minute cache and the same politeness rationale; the pause
    lives here (not in the caller's loop) so it only costs anything on a
    genuine cache miss, never on a page rerun hitting already-cached
    entries.

    Returns None if picks aren't available for this entry/gameweek
    (private profile some FPL privacy settings can still restrict,
    invalid ID, or a network failure) -- never raises out to the caller.
    """
    if not entry_id or entry_id < 1:
        return None

    url = f"{API_BASE_URL}/entry/{entry_id}/event/{gw}/picks/"
    try:
        time.sleep(0.15)
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if 'picks' in data and len(data['picks']) > 0:
                return data
        return None
    except requests.RequestException:
        return None


def fetch_league_squads(entry_ids, gw):
    """
    Fetches every entry's picks for `gw`, one at a time (each individual
    call is itself cached and paced -- see fetch_entry_picks). Returns
    {entry_id: picks_dict_or_None}; a None value means that one manager's
    squad couldn't be retrieved and callers should skip them, not fail
    the whole league view over one manager's unavailable picks.
    """
    return {entry_id: fetch_entry_picks(entry_id, gw) for entry_id in entry_ids}


# --- TIMEZONE CONVERSION UTILITIES ---
from datetime import datetime
from zoneinfo import ZoneInfo

TZ_MAP = {
    "Local / Regional": datetime.now().astimezone().tzinfo,
    "UTC": ZoneInfo("UTC"),
    "BST (UK / UTC+1)": ZoneInfo("Europe/London"),
    "CET (Europe / UTC+2)": ZoneInfo("Europe/Paris"),
    "NPT (Nepal / UTC+5:45)": ZoneInfo("Asia/Kathmandu"),
    "IST (India / UTC+5:30)": ZoneInfo("Asia/Kolkata"),
    "EST (US East / UTC-5)": ZoneInfo("America/New_York"),
    "PST (US West / UTC-8)": ZoneInfo("America/Los_Angeles")
}

def format_timestamp_tz(ts_str, tz_name="Local / Regional", fmt="%a %H:%M Local"):
    """
    Converts a UTC ISO timestamp string to target timezone and formats it.
    """
    if not ts_str:
        return ""
    try:
        dt = pd.to_datetime(ts_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        target_tz = TZ_MAP.get(tz_name, datetime.now().astimezone().tzinfo)
        converted = dt.tz_convert(target_tz)
        return converted.strftime(fmt)
    except Exception:
        return str(ts_str)


def fixture_status(fixture, tz_name="Local / Regional"):
    """
    Single source of truth for a fixture's display state. Returns
    (label, css_class).

    `finished` lags the real match end by up to an hour (FPL holds it False
    until bonus points are confirmed) -- `finished_provisional` and
    `minutes >= 90` both flip the moment the match itself is actually over,
    so either one is treated as full-time here even when `finished` is
    still False. Every place in app.py that shows fixture state must call
    this rather than checking `started`/`finished` itself, so there is one
    place that knows the FPL API's finished-flag lag instead of several.
    """
    finished = bool(fixture.get('finished', False))
    finished_provisional = bool(fixture.get('finished_provisional', False))
    started = bool(fixture.get('started', False))
    minutes = fixture.get('minutes') or 0

    if finished or finished_provisional:
        return "FT", "status-ft"
    if started and minutes >= 90:
        return "FT", "status-ft"
    if started:
        if minutes > 0:
            return f"{minutes}'", "status-live"
        return "LIVE", "status-live"

    ko_str = format_timestamp_tz(fixture.get('kickoff_time'), tz_name, fmt="%a %H:%M Local")
    return ko_str, "status-upcoming"