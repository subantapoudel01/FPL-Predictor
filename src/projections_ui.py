"""Shared projection search for the Gameweek and Players views."""

from unicodedata import normalize

import streamlit as st


def render_player_projections(df_live, *, key_prefix, sanitize_query):
    """Display rule-adjusted forecasts while keeping the scores numeric."""
    with st.container(key=f"deadline_projections_{key_prefix}"):
        st.header("Player projections", anchor=f"{key_prefix}-projections")
        st.caption("Projected points are the forecast after the app’s scoring adjustments.")
        if df_live.empty:
            st.info("Player projections are unavailable for this gameweek.")
            return

        search_key = f"{key_prefix}_projection_search"
        position_key = f"{key_prefix}_projection_position"
        club_key = f"{key_prefix}_projection_club"

        def reset_filters():
            st.session_state[search_key] = ""
            st.session_state[position_key] = "All positions"
            st.session_state[club_key] = "All clubs"

        clubs = sorted(df_live["team_name"].dropna().unique())
        with st.container(key=f"deadline_projection_filters_{key_prefix}"):
            search, positions, teams = st.columns([2, 1, 1])
            with search:
                query = st.text_input("Search players", placeholder="Player name", key=search_key)
            with positions:
                position = st.selectbox(
                    "Position", ["All positions", "GKP", "DEF", "MID", "FWD"], key=position_key
                )
            with teams:
                club = st.selectbox("Club", ["All clubs", *clubs], key=club_key)

        players = df_live
        if position != "All positions":
            players = players[players["position"] == position]
        if club != "All clubs":
            players = players[players["team_name"] == club]

        def search_text(value):
            return sanitize_query(normalize("NFKD", value).casefold())

        clean_query = search_text(query)
        if clean_query:
            names = players["web_name"].fillna("").map(search_text)
            players = players[names.str.contains(clean_query, regex=False, na=False)]

        with st.container(
            key=f"deadline_projection_actions_{key_prefix}",
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            count = len(players)
            st.caption(f"{count:,} {'player' if count == 1 else 'players'}")
            st.button(
                "Reset filters",
                on_click=reset_filters,
                disabled=not (query or position != "All positions" or club != "All clubs"),
                key=f"{key_prefix}_projection_reset",
            )

        if players.empty:
            st.info("No players match these filters")
            return

        players = players.sort_values("final_xp", ascending=False)
        columns = {
            "web_name": st.column_config.TextColumn("Player", width=132, pinned=True),
            "final_xp": st.column_config.NumberColumn("Projected points", width=120, format="%.1f"),
            "team_name": st.column_config.TextColumn("Club", width=110),
            "position": st.column_config.TextColumn("Pos", width=56),
            "next_opponent": st.column_config.TextColumn("Opponent", width=95),
            "value": st.column_config.NumberColumn("Price", width=72, format="£%.1fm"),
        }
        visible_rows = min(len(players), 9 if key_prefix == "gameweek" else 15)
        st.dataframe(
            players[list(columns)],
            column_config=columns,
            width="stretch",
            height=36 + 35 * visible_rows,
            hide_index=True,
            alt="Player projections sorted by projected points by default",
        )
        _render_projection_details(players, key_prefix)


def _render_projection_details(players, key_prefix):
    with st.expander("Projection details"):
        by_id = players.set_index("id")
        names = (by_id["web_name"] + " · " + by_id["team_name"]).to_dict()
        player_id = st.selectbox(
            "Choose a player",
            options=list(names),
            index=None,
            format_func=names.__getitem__,
            key=f"{key_prefix}_projection_detail_player",
        )

        if player_id is not None:
            player = by_id.loc[player_id]
            st.write(
                f"**Base xP** {player['raw_xp']:.1f} · "
                f"**Projected points** {player['final_xp']:.1f}"
            )
            st.caption("Base xP is the model forecast before scoring adjustments.")
            st.write(player["reasoning"])

    with st.expander("Advanced player stats"):
        columns = {
            "web_name": st.column_config.TextColumn("Player", pinned=True),
            "defensive_contribution_per_90": st.column_config.NumberColumn("DEFCON / 90", format="%.1f"),
            "bonus": st.column_config.NumberColumn("Bonus", format="%d"),
            "bps": st.column_config.NumberColumn("BPS", format="%d"),
            "value": st.column_config.NumberColumn("Price", format="£%.1fm"),
            "price_change_today": st.column_config.NumberColumn("Price change (£m)", format="%+.1f"),
            "ownership_pct": st.column_config.NumberColumn("Owned", format="%.1f%%"),
            "next_match_difficulty": st.column_config.NumberColumn("Fixture difficulty", format="%d"),
        }
        st.dataframe(
            players[list(columns)],
            column_config=columns,
            width="stretch",
            hide_index=True,
            alt="Advanced metrics for the filtered players",
        )
