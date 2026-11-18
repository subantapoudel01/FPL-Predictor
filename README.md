# FPL Forecast Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://catalan-fpl.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Personal applied machine learning project by Subanta Poudel. The app
forecasts Fantasy Premier League points using position-specific XGBoost and
Random Forest ensembles. It shows adjusted points and a separate captain
ranking.

In the saved 2025-26 backtest (38 gameweeks, 29,338 player-gameweek rows),
the ensemble had MAE 1.167 points versus 1.055 for linear regression. Mean
rank correlation improved from 0.684 to 0.693. These results measure base
forecasts before adjustments.

---

## ❓ What is FPL Predictor?

FPL Predictor is an explainable decision-support application for Fantasy
Premier League managers. It forecasts upcoming player performance and
explains the adjusted points shown to users.

Unlike traditional "black-box" artificial intelligence that simply outputs a number without context, FPL Forecast Lab uses an **Explainable AI framework**—every projection clearly explains why a player is expected to score points (factoring in recent form, matchup difficulty, clean sheet odds, and squad rotation risks). Users can enter their unique FPL Team ID to immediately receive tailored starting XI optimizations and budget-compliant transfer recommendations.

---

## ✨ Key Features

- 🔮 **Explainable xP Engine:** Real-time point predictions broken down into base performance, fixture ease, clean sheet odds, and squad-depth rotation risk.
- 📊 **Rate My Team & Squad Import:** Enter your FPL Team ID for instant Starting XI projections, formation-valid bench optimization, and 3-player-club-rule compliant transfer targets.
- ⚽ **Matchday Center:** Local timezone-aware fixture kickoffs, live deadline countdown, and dynamic 5-gameweek FDR scheduling matrix.
- 📈 **Historical Performance Analysis:** Zero-leakage multi-season tracking plotting predicted vs. actual points across 110,000+ match records.
- 🏆 **FPL Hall of Fame:** Curated all-time career and single-season legend records.

---

## 🛠️ Tech Stack & Architecture

- **Backend / Modeling:** Python, Scikit-learn, XGBoost, Random Forest, Pandas, PyArrow
- **Dashboard / Frontend:** Streamlit, Plotly Express, Custom Responsive CSS
- **Data Ingestion:** Official Fantasy Premier League REST API + Vaastav Multi-Season Historical Archive + Understat historical enrichment
- **Security:** CSRF protection, sanitized regex inputs, trimmed API payloads, zero hardcoded credentials

---

## 🚀 Quickstart

```bash
brew install python@3.11 libomp
/opt/homebrew/bin/python3.11 -m venv ~/.venvs/fpl-macos
~/.venvs/fpl-macos/bin/python -m pip install -r requirements-macos.txt
~/.venvs/fpl-macos/bin/python -m pip install scikit-learn==1.9.0
~/.venvs/fpl-macos/bin/python -m pip check
~/.venvs/fpl-macos/bin/python -m streamlit run app.py
```

The tested macOS runtime is Python 3.11.18. `requirements-macos.txt`
pins NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.4.2, SciPy 1.13.1,
joblib 1.4.2, Streamlit 1.67.0, requests 2.34.2, Plotly 7.1.0,
PyArrow 19.0.1, tzdata 2026.6, pytest 9.2.0, ipykernel 7.4.0, and
matplotlib 3.9.4. The active position ensemble was trained with
scikit-learn 1.9.0, so that pin is applied after the base profile.

The active model is the position-specific XGBoost and Random Forest ensemble
in `Models/position_models_v1.pkl`. Linear regression is the legacy baseline;
the component model remains disabled.

## Release verification

All checks below used `~/.venvs/fpl-macos/bin/python`:

| Check | Result |
|---|---|
| `scripts/verify_local.py` | Passed |
| `scripts/verify_app.py` | Passed with synthetic API fixture and committed history |
| `scripts/verify_app.py --api-unavailable` | Passed |
| `-m pytest -q` | Passed: 8 tests |
| Live forecast | Passed: GW11; FUL (H) opponent matched against fresh FPL API |
| Current live API | Passed during release check |

The export currently represents one fixture per player for a gameweek. In a
double gameweek, historical export logic keeps the first representative
fixture/opponent rather than counting both fixtures. This remains a documented
release limitation; live fixture data still contains the season's fixture
rows.

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
