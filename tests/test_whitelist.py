"""
Guards against the exact bug that has now shipped four times: a field used
somewhere in src/features.py or src/predictor.py that isn't in the field
whitelist in src/data_loader.py, so it silently reads as missing/None
(usually ending up as 0) instead of raising an error anywhere.

Hits the live FPL API once for bootstrap-static/ (shared by the elements/
teams/events checks) and once for fixtures/, then asserts each whitelist in
src/data_loader.py is a subset of the real keys on that record. Skips
gracefully -- doesn't fail the suite -- if the API is unreachable, since
that's an environment problem, not a code problem.
"""
import requests
import pytest

from src.data_loader import (
    API_BASE_URL,
    ELEMENT_KEYS,
    TEAM_KEYS,
    EVENT_KEYS,
    FIXTURE_KEYS,
)


@pytest.fixture(scope="module")
def bootstrap_static():
    try:
        r = requests.get(f"{API_BASE_URL}/bootstrap-static/", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        pytest.skip(f"Live FPL API unreachable (bootstrap-static/): {e}")


@pytest.fixture(scope="module")
def fixtures_payload():
    try:
        r = requests.get(f"{API_BASE_URL}/fixtures/", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        pytest.skip(f"Live FPL API unreachable (fixtures/): {e}")


def _assert_subset(whitelist, real_keys, label):
    missing = sorted(set(whitelist) - set(real_keys))
    assert not missing, (
        f"{label} has field(s) the live API doesn't return: {missing}. "
        f"Either it's a typo, or FPL renamed/removed the field."
    )


def test_element_whitelist_is_subset_of_live_fields(bootstrap_static):
    elements = bootstrap_static.get('elements', [])
    if not elements:
        pytest.skip("Live API returned no elements to check against")
    _assert_subset(ELEMENT_KEYS, elements[0].keys(), "ELEMENT_KEYS")


def test_team_whitelist_is_subset_of_live_fields(bootstrap_static):
    teams = bootstrap_static.get('teams', [])
    if not teams:
        pytest.skip("Live API returned no teams to check against")
    _assert_subset(TEAM_KEYS, teams[0].keys(), "TEAM_KEYS")


def test_event_whitelist_is_subset_of_live_fields(bootstrap_static):
    events = bootstrap_static.get('events', [])
    if not events:
        pytest.skip("Live API returned no events to check against")
    _assert_subset(EVENT_KEYS, events[0].keys(), "EVENT_KEYS")


def test_fixture_whitelist_is_subset_of_live_fields(fixtures_payload):
    dict_fixtures = [f for f in fixtures_payload if isinstance(f, dict)]
    if not dict_fixtures:
        pytest.skip("Live API returned no fixtures to check against")
    _assert_subset(FIXTURE_KEYS, dict_fixtures[0].keys(), "FIXTURE_KEYS")
