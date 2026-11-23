"""Presentation for the Deadline Sheet Gameweek page."""

from html import escape
from math import isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import TZ_MAP, fixture_status


def render_theme():
    """Load styles for the owned markup and keyed Streamlit containers."""
    stylesheet = Path(__file__).with_name("deadline_sheet.css").read_text(encoding="utf-8")
    st.html(f"<style>{stylesheet}</style>")


def render_wordmark():
    st.html("""
        <div class="deadline-brand">
            <span class="deadline-mark" aria-hidden="true"><i></i></span>
            <span>Deadline Sheet</span>
        </div>
    """)


def _timestamp(value):
    if not isinstance(value, str):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _local_time(timestamp, selected_tz):
    timezone = TZ_MAP.get(selected_tz, TZ_MAP["Local / Regional"])
    local = timestamp.tz_convert(timezone)
    abbreviation = "NPT" if getattr(timezone, "key", None) == "Asia/Kathmandu" else local.strftime("%Z")
    return f"{local:%a %d %b} · {local:%H:%M} {abbreviation}".strip()


def _deadline_state(deadline, event, is_preseason, season_finished):
    if season_finished:
        return "Season complete", "Review my team"
    if event.get("finished"):
        return "Gameweek complete", "Review my team"
    if deadline is None:
        if is_preseason:
            return "Preseason", "Rate my team"
        return "", "Open My Team"

    remaining = deadline - pd.Timestamp.now(tz="UTC")
    if remaining.total_seconds() <= 0:
        return "Deadline passed · Lineups locked", "Review my team"

    hours, minutes = divmod(remaining.seconds // 60, 60)
    if remaining.days:
        countdown = f"{remaining.days}d {hours}h"
    elif hours:
        countdown = f"{hours}h {minutes}m"
    else:
        countdown = f"{minutes}m" if minutes else "under 1m"
    snapshot = f"Countdown snapshot · {countdown}"
    state = f"Preseason · {snapshot}" if is_preseason else snapshot
    return state, "Rate my team"


def render_gameweek_header(
    current_gw,
    event,
    selected_tz,
    *,
    on_rate_team,
    is_preseason=False,
    season_finished=False,
):
    event = event or {}
    deadline = _timestamp(event.get("deadline_time"))
    state, team_action = _deadline_state(deadline, event, is_preseason, season_finished)
    state_html = f'<span class="deadline-state">{state}</span>' if state else ""
    if deadline is None:
        time_html = '<strong class="deadline-time">Deadline unavailable</strong>'
    else:
        datetime_value = escape(deadline.isoformat(), quote=True)
        local_time = escape(_local_time(deadline, selected_tz))
        time_html = f'<time class="deadline-time" datetime="{datetime_value}">{local_time}</time>'

    with st.container(key="deadline_week_header"):
        title, date, team = st.columns([1.3, 1, 0.65], vertical_alignment="bottom")
        with title:
            st.html(f"""
                <div class="deadline-week">
                    <h1 id="gameweek-heading">Gameweek {escape(str(current_gw))}</h1>
                    <p>Fantasy Premier League forecasts and squad decisions</p>
                </div>
            """)
        with date:
            st.html(f"""
                <div class="deadline-date">
                    <span class="deadline-label">{'Deadline' if deadline is not None else 'Matchday'}</span>
                    {time_html}
                    {state_html}
                </div>
            """)
        with team:
            st.button(
                team_action,
                type="primary",
                key="deadline_rate_team",
                on_click=on_rate_team,
            )
        if is_preseason and not season_finished:
            st.caption("Gameweek 1 projections use prior-season data and position medians for new players.")
        st.html("""
            <nav class="deadline-gameweek-links" aria-label="Gameweek sections">
                <a href="#gameweek-projections">Projections</a>
                <a href="#deadline-fixtures">Fixtures</a>
            </nav>
        """)


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _display_number(value):
    number = _finite_number(value)
    return f"{number:.1f}" if number is not None else "—"


def _display_text(value, empty="—"):
    return escape(str(value)) if pd.notna(value) and value != "" else empty


def _captain_rationale(player):
    difficulty = _finite_number(player.get("next_match_difficulty"))
    starter = _finite_number(player.get("starter_prob"))
    penalty_taker = player.get("is_penalty_taker")
    parts = []
    if difficulty is not None and 1 <= difficulty <= 2:
        parts.append(f"Favorable fixture (FDR {difficulty:g})")
    elif difficulty is not None and difficulty >= 4:
        parts.append(f"Tough fixture (FDR {difficulty:g})")
    if pd.notna(penalty_taker) and penalty_taker:
        parts.append("First-choice penalty taker")
    if player.get("position") == "DEF":
        xgi = _finite_number(player.get("xgi_per_90"))
        if xgi is not None:
            parts.append(f"{xgi:.2f} expected goal involvements per 90")
    if starter is not None:
        parts.append(f"{starter:g}% estimated to start")
    return escape(" · ".join(parts))


def _captain_fixture(player):
    opponent = player.get("next_opponent")
    if pd.isna(opponent) or opponent in ("", "-"):
        return "No fixture"
    club, separator, venue = str(opponent).rpartition(" (")
    venue_names = {"H)": "Home", "A)": "Away"}
    if separator and venue in venue_names:
        return f'vs {escape(club)} (<abbr title="{venue_names[venue]}">{venue[0]}</abbr>)'
    return f"vs {escape(str(opponent))}"


def _captain_card(player, pick):
    return f"""
        <article class="deadline-captain">
            <div class="deadline-captain-order">
                <span class="deadline-label">{pick}</span>
                <span class="deadline-captain-score">Captain score <strong>{_display_number(player.get("ceiling_score"))}</strong></span>
            </div>
            <div class="deadline-captain-main">
                <div>
                    <h3>{_display_text(player.get("web_name"))}</h3>
                    <p>{_display_text(player.get("team_name"))} · {_captain_fixture(player)}</p>
                </div>
                <div class="deadline-captain-points">
                    <strong>{_display_number(player.get("final_xp"))}</strong>
                    <span>Projected points</span>
                </div>
            </div>
            <p class="deadline-captain-reason">{_captain_rationale(player)}</p>
        </article>
    """


def render_captaincy_card(df_candidates):
    """Show the existing captain order and eligibility policy."""
    heading = """
        <div class="deadline-section-heading">
            <h2 id="deadline-captains">Captain picks</h2>
            <p>Ranked by captain score</p>
        </div>
    """
    if df_candidates is None or df_candidates.empty or "ceiling_score" not in df_candidates:
        st.html(f"""
            <section aria-labelledby="deadline-captains">
                {heading}<p class="deadline-muted">Captain picks are unavailable.</p>
            </section>
        """)
        return

    # Match the existing sort and fall back to the full pool if no player qualifies.
    flags = df_candidates.get("captain_eligible", pd.Series(False, index=df_candidates.index))
    eligible = df_candidates[flags.eq(True).fillna(False)]
    candidates = eligible if not eligible.empty else df_candidates
    picks = candidates.sort_values("ceiling_score", ascending=False).head(5)

    cards = "".join(
        _captain_card(player, label)
        for (_, player), label in zip(picks.head(3).iterrows(), ("Top pick", "Second pick", "Third pick"))
    )
    st.html(f"""
        <section aria-labelledby="deadline-captains">
            {heading}
            <div class="deadline-captain-grid">{cards}</div>
        </section>
    """)

    if len(picks) > 3:
        columns = {
            "web_name": "Player",
            "team_name": "Club",
            "next_opponent": "Opponent",
            "final_xp": "Projected points",
            "ceiling_score": "Captain score",
        }
        extra = picks.iloc[3:].reindex(columns=list(columns)).rename(columns=columns)
        with st.expander("More captain picks"):
            st.dataframe(
                extra,
                width="stretch",
                hide_index=True,
                column_config={
                    "Projected points": st.column_config.NumberColumn(format="%.1f"),
                    "Captain score": st.column_config.NumberColumn(format="%.1f"),
                },
            )


def _club_badge(team):
    short_name = _display_text(team.get("short_name"), "TBC")
    full_name = _display_text(team.get("name"), short_name)
    return f'<abbr class="deadline-club" title="{full_name}">{short_name}</abbr>'


def _fixture_row(fixture, teams, status_label, status, selected_tz):
    home = _club_badge(teams.get(fixture.get("team_h"), {}))
    away = _club_badge(teams.get(fixture.get("team_a"), {}))
    details = ""
    if status == "status-upcoming":
        kickoff = _timestamp(fixture.get("kickoff_time"))
        if kickoff is None:
            details = '<span class="deadline-kickoff">Kickoff to be confirmed</span>'
        else:
            datetime_value = escape(kickoff.isoformat(), quote=True)
            local_kickoff = escape(_local_time(kickoff, selected_tz))
            details = f'<time class="deadline-kickoff" datetime="{datetime_value}">{local_kickoff}</time>'
        score = "vs"
    else:
        home_score, away_score = fixture.get("team_h_score"), fixture.get("team_a_score")
        score = f"{_display_text(home_score)}–{_display_text(away_score)}"
        if status == "status-live" and status_label != "LIVE":
            details = f'<span class="deadline-kickoff">{escape(status_label)}</span>'
    return f"""
        <li class="deadline-match">
            <div class="deadline-match-line">{home}<span>{score}</span>{away}</div>
            {details}
        </li>
    """


def render_gameweek_fixtures(raw_json, current_gw, selected_tz):
    teams = {team["id"]: team for team in raw_json.get("static", {}).get("teams", [])}
    fixtures = [fixture for fixture in raw_json.get("fixtures", []) if fixture.get("event") == current_gw]
    statuses = {"status-live": "Live", "status-upcoming": "Upcoming", "status-ft": "Full time"}
    groups = {status: [] for status in statuses}
    for fixture in sorted(fixtures, key=lambda match: (not match.get("kickoff_time"), match.get("kickoff_time") or "")):
        label, status = fixture_status(fixture, selected_tz)
        groups[status].append(_fixture_row(fixture, teams, label, status, selected_tz))

    groups_html = "".join(
        f"""
        <div class="deadline-fixture-group {status}">
            <h3>{statuses[status]}</h3>
            <ul>{''.join(rows)}</ul>
        </div>
        """
        for status, rows in groups.items()
        if rows
    )
    empty = '<p class="deadline-fixtures-empty">Fixture information is unavailable for this gameweek.</p>'
    st.html(f"""
        <section class="deadline-fixtures" aria-labelledby="deadline-fixtures">
            <h2 id="deadline-fixtures">Fixtures</h2>
            {groups_html or empty}
        </section>
    """)
