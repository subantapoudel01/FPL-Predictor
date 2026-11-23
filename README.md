# Deadline Sheet

Deadline Sheet helps Fantasy Premier League managers check the gameweek deadline,
compare captain picks and plan squad decisions. It builds on **FPL Forecast Lab**,
the original applied machine learning project by Subanta Poudel.

**Designed by Animythstudios.**

| View | What you can do |
| --- | --- |
| Gameweek | Check the deadline and fixtures, compare captain picks, and scan player projections. |
| My Team | Load your public Team ID, check the starting XI and bench, review transfer suggestions, and export a solver CSV. |
| Players | Search the full projection list, inspect scoring adjustments, and explore season points and player history. |
| Mini-League | Load a classic league by ID and compare standings, squad projections, captains, and ownership. |
| Model Report | Compare saved backtest results and explore evaluation methods and historical records. |

Team and league links support `?team_id=123456` and `?league_id=123456`. The
**Timezone** control sets the display timezone for deadlines and kickoffs. The
deadline countdown is recalculated when the screen reruns.

## Reading the forecasts

- **Base xP** is the model forecast before scoring adjustments.
- **Projected points** or **Adjusted xP** includes the app’s rules for factors such as fixtures, rotation, and availability.
- **Captain score** is the ranking heuristic used to order captain options.

The serving model uses position-specific XGBoost and Random Forest ensembles in
`Models/position_models_v1.pkl`. The saved 2025–26 base-forecast backtest covers
38 gameweeks and 29,338 player-gameweeks. The ensemble recorded MAE of 1.167
points and mean gameweek rank correlation of 0.693. Linear regression recorded
MAE of 1.055 points and mean rank correlation of 0.684.

The solver CSV uses one representative fixture per player in double gameweeks.
Fixture context changes across the horizon and form inputs use the current snapshot.

## Run locally

The macOS setup uses Python 3.11 and Homebrew’s `libomp` for XGBoost.

```bash
brew install python@3.11 libomp
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
.venv/bin/python -m pip check
.venv/bin/python -m streamlit run app.py
```

The macOS profile pins scikit-learn 1.9.0 to match the saved ensemble and
Streamlit 1.67.0 for the UI. The existing hosted release is listed below.
[FPL Forecast Lab on Streamlit](https://catalan-fpl.streamlit.app)

## Verify locally

```bash
.venv/bin/python scripts/verify_local.py
.venv/bin/python scripts/verify_app.py
.venv/bin/python scripts/verify_app.py --api-unavailable
.venv/bin/python -m pytest -q
```

The app smoke check uses the committed synthetic API fixture and historical data.
Pass `--live` to `scripts/verify_app.py` to check the current FPL feed. The live
API whitelist tests skip if the feed is unreachable.

## Repository layout and data policy

The app and its supporting pipelines are intentionally separated:

| Area | Purpose | Lifecycle |
|---|---|---|
| `app.py`, `src/` | Live Streamlit UI, API ingestion, feature preparation, prediction dispatch, and projections | Production code |
| `Models/` | Versioned serving and comparison model artifacts | Keep only named, evaluated model versions |
| `Data/Raw/` | Reusable source snapshots | Keep one canonical copy of each source |
| `Data/Cleaned/` | Notebook outputs and the committed pre-season fallback | Rebuild only as a reviewed experiment |
| `training/`, `evaluation/` | Reproducible fitting and walk-forward evaluation | Source code and small result metadata belong in Git |
| `Notebooks/` | Exploratory/reproducible data workflow | Do not use as the live serving path |
| `Docs/`, `Images/` | Coursework and portfolio evidence | Supporting material, not runtime dependencies |
| `scripts/` | Local verification checks | Development-only |
| `tests/fixtures/` | Synthetic API fixtures for verification | Test data |

`Data/Raw/cleaned_merged_seasons.csv` is the canonical historical seasons
snapshot. Do not add a second copy under another filename. Generated caches,
virtual environments, notebook checkpoints, local locks, and scratch CSVs
remain outside version control. Rebuilt datasets, models, or metrics should
be written to a new experiment directory until their evaluation justifies
replacing a production artifact.
