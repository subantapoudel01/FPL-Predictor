import requests
import pandas as pd

# API Endpoints
API_BASE_URL = "https://fantasy.premierleague.com/api"

def fetch_api_data():
    """
    Fetches core FPL data required for the model and trims response payloads
    to retain strictly essential keys, reducing memory overhead and security exposure.
    
    Returns:
        dict: Contains trimmed 'static' (elements/teams/events) and 'fixtures' data.
        None: If the API request fails.
    """
    try:
        # 1. Fetch Bootstrap Static (Players, Teams, Events)
        r_static = requests.get(f"{API_BASE_URL}/bootstrap-static/", timeout=10)
        r_static.raise_for_status()
        static_data = r_static.json()
        
        # 2. Fetch Fixtures (Required for Difficulty Calculation)
        r_fixtures = requests.get(f"{API_BASE_URL}/fixtures/", timeout=10)
        r_fixtures.raise_for_status()
        fixtures_data = r_fixtures.json()
        
        # Trim static elements to retain strictly necessary feature engineering fields
        trimmed_elements = []
        element_keys = [
            'id', 'code', 'first_name', 'second_name', 'web_name', 'element_type',
            'team', 'now_cost', 'minutes', 'form', 'ict_index', 'creativity',
            'influence', 'threat', 'chance_of_playing_next_round', 'chance_of_playing', 'status',
            'defensive_contribution_per_90',
            'bonus', 'bps', 'cost_change_event', 'selected_by_percent'
        ]
        for elem in static_data.get('elements', []):
            trimmed_elements.append({k: elem.get(k) for k in element_keys if k in elem})
            
        trimmed_teams = []
        team_keys = ['id', 'short_name', 'name']
        for team in static_data.get('teams', []):
            trimmed_teams.append({k: team.get(k) for k in team_keys if k in team})
            
        trimmed_events = []
        event_keys = ['id', 'name', 'is_current', 'is_next', 'finished', 'deadline_time']
        for event in static_data.get('events', []):
            trimmed_events.append({k: event.get(k) for k in event_keys if k in event})
            
        trimmed_fixtures = []
        fixture_keys = [
            'id', 'event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty',
            'kickoff_time', 'started', 'finished', 'team_h_score', 'team_a_score'
        ]
        for fix in fixtures_data:
            if isinstance(fix, dict):
                trimmed_fixtures.append({k: fix.get(k) for k in fixture_keys if k in fix})
            
        return {
            "static": {
                "elements": trimmed_elements,
                "teams": trimmed_teams,
                "events": trimmed_events
            },
            "fixtures": trimmed_fixtures
        }
    except requests.RequestException as e:
        print(f"Error fetching API data: {e}")
        return None

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

def fetch_player_live_summary(player_id):
    """
    Fetches live performance stats for an individual player from FPL element-summary API.
    
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