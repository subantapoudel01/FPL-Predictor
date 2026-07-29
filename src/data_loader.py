import requests
import pandas as pd

# API Endpoints
API_BASE_URL = "https://fantasy.premierleague.com/api"

def fetch_api_data():
    """
    Fetches core FPL data required for the model.
    
    Returns:
        dict: Contains 'static' (players/teams) and 'fixtures' (schedule) data.
        None: If the API request fails.
    """
    try:
        # 1. Fetch Bootstrap Static (Players, Teams, Events)
        r_static = requests.get(f"{API_BASE_URL}/bootstrap-static/", timeout=10)
        r_static.raise_for_status()
        
        # 2. Fetch Fixtures (Required for Difficulty Calculation)
        r_fixtures = requests.get(f"{API_BASE_URL}/fixtures/", timeout=10)
        r_fixtures.raise_for_status()
        
        return {
            "static": r_static.json(),
            "fixtures": r_fixtures.json()
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