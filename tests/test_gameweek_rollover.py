import pandas as pd

from src.features import detect_target_gameweek, process_data


def _payload(current_finished, gw=None):
    return {
        "static": {
            "elements": [
                {
                    "id": 1,
                    "code": 1,
                    "first_name": "Test",
                    "second_name": "Forward",
                    "web_name": "Forward",
                    "element_type": 4,
                    "team": 1,
                    "now_cost": 80,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "minutes": 450,
                    "form": "5.0",
                    "ict_index": "10.0",
                    "creativity": "10.0",
                    "influence": "10.0",
                    "threat": "10.0",
                    "total_points": 30,
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "short_name": "AAA",
                    "name": "Alpha",
                    "strength_overall_home": 4,
                    "strength_overall_away": 4,
                    "strength_attack_home": 4,
                    "strength_attack_away": 4,
                    "strength_defence_home": 4,
                    "strength_defence_away": 4,
                },
                {
                    "id": 2,
                    "short_name": "BBB",
                    "name": "Beta",
                    "strength_overall_home": 3,
                    "strength_overall_away": 3,
                    "strength_attack_home": 3,
                    "strength_attack_away": 3,
                    "strength_defence_home": 3,
                    "strength_defence_away": 3,
                },
            ],
            "events": [
                {
                    "id": 10,
                    "name": "Gameweek 10",
                    "is_current": True,
                    "is_next": False,
                    "finished": current_finished,
                    "deadline_time": "2026-11-15T18:30:00Z",
                },
                {
                    "id": 11,
                    "name": "Gameweek 11",
                    "is_current": False,
                    "is_next": True,
                    "finished": False,
                    "deadline_time": "2099-11-20T18:30:00Z",
                },
            ],
        },
        "fixtures": [
            {
                "id": 101,
                "event": gw or 11,
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
                "started": False,
                "finished": False,
                "finished_provisional": False,
            }
        ],
    }


def test_finished_current_rolls_to_upcoming_gameweek():
    now = pd.Timestamp("2026-11-18T12:00:00Z")
    payload = _payload(current_finished=True, gw=11)

    assert detect_target_gameweek(payload["static"]["events"], now=now) == 11

    frame, target_gw, is_preseason = process_data(payload)
    frame = frame.iloc[:1]
    assert target_gw == 11
    assert not is_preseason
    assert frame.iloc[0]["next_opponent"] == "BBB (H)"
    assert not bool(frame.iloc[0]["season_finished"])


def test_unfinished_current_gameweek_stays_live():
    now = pd.Timestamp("2026-11-18T12:00:00Z")
    payload = _payload(current_finished=False, gw=10)

    assert detect_target_gameweek(payload["static"]["events"], now=now) == 10

    frame, target_gw, is_preseason = process_data(payload, target_gw=None)
    assert target_gw == 10
    assert not is_preseason
    assert frame.iloc[0]["next_opponent"] == "BBB (H)"


def test_explicit_target_gw_override_is_preserved():
    payload = _payload(current_finished=True, gw=12)
    payload["static"]["events"].append(
        {
            "id": 12,
            "name": "Gameweek 12",
            "is_current": False,
            "is_next": False,
            "finished": False,
            "deadline_time": "2026-11-27T18:30:00Z",
        }
    )

    frame, target_gw, is_preseason = process_data(payload, target_gw=12)
    assert target_gw == 12
    assert not is_preseason
    assert frame.iloc[0]["next_opponent"] == "BBB (H)"


def test_season_end_has_no_future_gameweek():
    now = pd.Timestamp("2026-11-30T12:00:00Z")
    payload = _payload(current_finished=True, gw=11)
    for event in payload["static"]["events"]:
        event["is_current"] = False
        event["is_next"] = False
        event["finished"] = True

    assert detect_target_gameweek(payload["static"]["events"], now=now) is None

    frame, target_gw, is_preseason = process_data(payload)
    assert target_gw == 11
    assert not is_preseason
    assert bool(frame.iloc[0]["season_finished"])
