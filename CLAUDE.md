# CLAUDE.md — CatalanPlays (FPL Points Predictor)

Streamlit app that predicts Fantasy Premier League player points and explains
each prediction in plain arithmetic. Single maintainer, personal project
(formerly "ApexFPL"), not affiliated with the Premier League or FPL.

## Files

- `app.py` — entire UI, all four tabs, all orchestration. No other frontend.
- `src/data_loader.py` — fetches FPL API endpoints, trims payloads to a field
  whitelist, timezone helpers.
- `src/features.py` — `process_data()` (live gameweek feature engineering)
  and `fetch_full_history()` (historical CSV pipeline).
- `src/predictor.py` — `load_active_model()` + `make_predictions()`: dispatches
  to whichever pipeline `src/config.py::USE_COMPONENT_MODEL` selects, plus
  captaincy scoring shared by both. `load_model()` still loads the legacy
  single LinearRegression, but only for `evaluation/backtest.py`'s `--compare`.
- `src/config.py` — `USE_COMPONENT_MODEL` (currently **False**): the component
  pipeline (`src/components.py`, `Models/component_models_v1.pkl`) backtested
  worse than the shipped per-position ensemble on the two metrics that matter
  most (Haulers/Tickers RMSE — see that file's comment and
  `evaluation/results/2025-26_component_models_v1.json`), so it stays off
  until that's fixed. Flipping this one flag is the entire rollback/rollout
  mechanism between the two architectures.
- `Models/position_models_v1.pkl` — the production model: one
  `MedianEnsembleRegressor` (XGBoost + RandomForest, median-blended) per
  GKP/DEF/MID/FWD, trained by `training/train_position_models.py` on the same
  7 features as the model it replaced. `Models/linear_reg_v1.pkl` — the
  original single-model `sklearn.LinearRegression` — is kept on disk
  specifically so the backtest can compare old against new.
- `Models/component_models_v1.pkl` — the not-currently-served alternative:
  fitted minutes/clean-sheet classifiers and attack/goals-conceded/saves/bonus
  calibrations (`training/train_component_models.py`), summed through the
  real scoring matrix by `src/components.py`. See `src/config.py` above.
- `metrics.json` — `{rmse, r2, notes}` shown in the sidebar; not recomputed
  by the app itself (comes from offline training in `Notebooks/`).
- `Data/Cleaned/`, `Data/Raw/` — committed CSVs used **only** as the
  pre-season baseline fallback (§Data sources, source C).

## The prediction pipeline

`raw_xp` (model) → rule layer → `final_xp` (shown to users), computed once
per page load in `src/predictor.py::make_predictions`:

1. Routed by position to that position's own ensemble (one `MedianEnsembleRegressor`
   per GKP/DEF/MID/FWD — see `training/train_position_models.py`), each
   `.predict()`ing on the same 7 features the old single model used
   (`rolling_3_minutes`, `rolling_3_ict_index`, `rolling_3_creativity`,
   `rolling_3_influence`, `rolling_3_threat`, `rolling_3_total_points`,
   `value`) → `raw_xp`. Replaced the single all-positions `LinearRegression`
   because a goalkeeper's points come from saves/clean sheets and a
   forward's from xG — one linear model can't represent both, which is why
   minutes-played alone used to recover ~91% of that model's R².
2. Hard availability wipe: `status in ('i','s')` or `chance_of_playing_next_round
   == 0` → `final_xp = 0.0` immediately, before any other rule runs.
3. Fixture difficulty bonus/penalty — MID/FWD only.
4. Premium (`×1.15` if price ≥ £10m) and star-captain (`+1.0`) bonuses for
   MID/FWD.
5. Dynamic clean sheet odds — DEF/GKP only: `cs_probability × 4.0`, where
   `cs_probability` (0.05–0.55) is computed in `features.py` from team
   strength deltas, not a fixture-difficulty-band lookup.
6. DEFCON bonus (stepped +2.0/+1.0/+0.4) from `defensive_contribution_per_90`
   vs. the real 2025-26 scoring threshold (10 for DEF/GKP, 12 for MID/FWD).
7. Squad-depth rotation tax (−25%) for non-nailed MID/FWD at clubs with 5+
   other priced-up attackers.
8. Starter-probability and partial-injury-doubt multipliers, then clip to
   `[0, 18]`.

Every step is echoed into a `reasoning` string shown in the UI. Separately,
`ceiling_score` (`final_xp × premium × penalty-taker` multipliers) drives
**only** the Top Captain Picks card — it is not the same ranking as the main
predictions table or the pitch-visualizer's captain badge, which both use
plain `final_xp`.

## Three data sources — do not conflate them

- **A. Live FPL REST API** (`data_loader.py`): `bootstrap-static/`,
  `fixtures/`, `event/{gw}/live/`, `element-summary/{id}/`,
  `entry/{id}/event/{gw}/picks/`. Cached 1 hour. This is the only source for
  anything "right now" — current gameweek, prices, ownership, injury status.
- **B. Historical per-gameweek CSVs** (`features.py::fetch_full_history`,
  from `github.com/vaastav/Fantasy-Premier-League`). The **only** source for
  the Last Gameweek Recap and the Player History tab. Rolling features use
  `.shift(1).rolling(3)` grouped by player+season — genuinely walk-forward,
  no lookahead. A gameweek's recap is recomputed fresh from this every load;
  nothing is cached to local disk.
- **C. Local committed CSVs** (`Data/Cleaned/model_ready_data.csv`) — used
  **only** before Gameweek 1 has started, via a 5-tier match cascade (Opta
  `code` → normalized full name → first+web name → web name → substring →
  positional-median imputation). The live API has nothing to compute rolling
  features from yet at that point in the season.

## The field-whitelist trap

`src/data_loader.py::fetch_api_data()` deliberately trims every API payload
(elements, teams, events, fixtures) down to an explicit list of named keys
before anything else touches it. **Any field used anywhere in `features.py`
or `predictor.py` that isn't already in that whitelist will silently read as
`None`/missing and get filled with a default (usually 0) — no error, no
warning.** This exact mistake has happened four separate times already: for
`defensive_contribution_per_90`, for `bonus`/`bps`/`cost_change_event`/
`selected_by_percent`, for `penalties_order`/`expected_goal_involvements_per_90`,
and for the team `strength_*` fields. **Before wiring up any new FPL API
field, add it to the relevant `*_keys` list in `data_loader.py` first**, then
verify with a live fetch that it isn't all-zero/all-null before building
logic on top of it (team `strength_attack_*`/`strength_defence_*` currently
read 0 for every team this early in the season — verify, don't assume).

## Deployment constraint: no local persistence

This app deploys to **Streamlit Community Cloud's free tier**, where the
container's local filesystem is wiped on every redeploy and the app sleeps
after inactivity. **No feature may depend on writing to local disk and
reading it back later** — an earlier recap design that snapshotted
predictions to a local JSON file was deliberately scrapped for exactly this
reason and rebuilt on source B above, which needs no persistence because
it's fully recomputable from the historical CSVs on every load.

## Design system — Blaugrana

Navy ground `#0F1420`, surface `#1A2235`, garnet accent `#D64B5E`, gold
`#FFC72C`. **Gold is reserved specifically for captain badges, star ratings,
and section-heading emphasis — not used as a general highlight color.**
Semantic colors are kept deliberately separate from the brand palette and
must not be changed to match it: the FDR ticker's difficulty bands
(`#006437` green → `#FF005A` pink) and the Sell (`#7f1d1d`/`#f87171`) / Buy
(`#064e3b`/`#34d399`) transfer tags carry meaning independent of branding.
Fonts: Big Shoulders Display (headings), Source Sans 3 (body), IBM Plex Mono
(all numeric values), loaded via a Google Fonts `<link>` injected once near
the top of `app.py`. These CSS rules cannot reach text inside `st.dataframe`
— Streamlit renders grid cells to an HTML canvas (glide-data-grid), outside
the DOM entirely.

## `render_html()` — required for any custom HTML/CSS

All raw HTML/CSS in `app.py` (pitch visualizer, transfer card, footer, font
`<style>` block) must go through the `render_html()` helper defined near the
top of the file, never a bare `st.markdown(..., unsafe_allow_html=True)`.
Streamlit's markdown renderer runs CommonMark first, and **any line indented
4+ spaces is treated as a preformatted code block and printed as literal
text instead of parsed as HTML** — Python f-strings written inside indented
functions inherit that source indentation by default. `render_html()`
strips leading whitespace per line before rendering, which fixes this at a
single call site. Do not add a new raw HTML block via plain `st.markdown`.

One related, separate gotcha: do not add a blanket `font-family` CSS rule
targeting bare `span` — Streamlit renders its own icon glyphs (expander
chevrons, toolbar icons) as ligature text inside `span` elements keyed to a
dedicated icon font, and overriding that font-family makes the icon show as
its literal name (e.g. `arrow_right`) instead of rendering.
