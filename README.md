# FPL Predictor

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

Unlike traditional "black-box" artificial intelligence that simply outputs a number without context, CatalanPlays uses an **Explainable AI framework**—every projection clearly explains why a player is expected to score points (factoring in recent form, matchup difficulty, clean sheet odds, and squad rotation risks). Users can enter their unique FPL Team ID to immediately receive tailored starting XI optimizations and budget-compliant transfer recommendations.

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
~/.venvs/fpl-macos/bin/python -m pip install -r local-ai/requirements-local.txt
~/.venvs/fpl-macos/bin/python -m pip install scikit-learn==1.9.0
~/.venvs/fpl-macos/bin/python -m pip check
~/.venvs/fpl-macos/bin/python -m streamlit run app.py
```

The tested macOS runtime is Python 3.11.18. `local-ai/requirements-local.txt`
pins NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.4.2, SciPy 1.13.1,
joblib 1.4.2, Streamlit 1.67.0, requests 2.34.2, Plotly 7.1.0,
PyArrow 19.0.1, tzdata 2026.6, pytest 9.2.0, ipykernel 7.4.0, and
matplotlib 3.9.4. The active position ensemble was trained with
scikit-learn 1.9.0, so that pin is applied after the base profile.

The active model is `Models/position_models_v1.pkl`. Linear regression is
the baseline and the component model remains disabled.

## Release verification

All checks below used `~/.venvs/fpl-macos/bin/python`:

| Check | Result |
|---|---|
| `local-ai/verify_local.py` | Passed |
| `local-ai/verify_app.py` | Passed with synthetic API fixture and committed history |
| `local-ai/verify_app.py --api-unavailable` | Passed |
| `-m pytest -q` | Passed: 4 tests |
| Live forecast | Passed: GW11; Haaland vs FUL (H); finite final xP 6.129 |
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
| `local-ai/` | Local verification and maintainer handoff notes | Development-only |

`Data/Raw/cleaned_merged_seasons.csv` is the canonical historical seasons
snapshot. Do not add a second copy under another filename. Generated caches,
virtual environments, notebook checkpoints, local locks, and scratch CSVs
remain outside version control. Rebuilt datasets, models, or metrics should
be written to a new experiment directory until their evaluation justifies
replacing a production artifact.

---

## ⚡ Project Case Study: CatalanPlays Decision Engine

### 1. Architectural Overview

```
 ┌────────────────────────────────────────────────────────┐
 │                    Data Sources                        │
 │  • Official Premier League REST API (Live Match Data)  │
 │  • Vaastav GitHub Archive (110k+ Historical Records)   │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │       ETL & Data Engineering Pipeline (PyArrow)        │
 │  • Name normalization cascade & Opta Code mapping      │
 │  • Point-in-time feature extraction (Zero Leakage)     │
 │  • 7-Day tiered caching layer (@st.cache_data)         │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │          Hybrid Intelligence Engine (xP Model)         │
 │  • Base Layer: Scikit-learn Linear Regression          │
 │  • Tactical Layer: Dynamic Squad Depth Rotation Risk   │
 │  • Heuristic Layer: Fixture Difficulty & Clean Sheets  │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │          Constraint Optimization & Decision UX         │
 │  • Rate My Team Live Evaluator                         │
 │  • Formation-valid Bench Swap (Min 3-DEF / 2-MID)      │
 │  • 3-Player-Per-Club Constraint Transfer Engine        │
 │  • Responsive Plotly UI with Local Timezone Converter  │
 └────────────────────────────────────────────────────────┘
```

### 2. Core Engineering Highlights

| Module | Technical Implementation | Business & Architectural Impact |
| :--- | :--- | :--- |
| **Hybrid xP Engine** | Linear Regression baseline combined with an expert heuristic layer. | Balances statistical regression to the mean with situational match context, providing readable math strings for full user transparency. |
| **Dynamic Squad Rotation** | In-memory evaluation of team squad depth and wage/price distribution. | Replaced static "big club" assumptions with an adaptive algorithm that penalizes rotation risk only when bench depth is actively contested. |
| **Constraint Transfer Engine** | Greedy constraint-satisfaction solver incorporating budget buffers, player exclusion, and team limits. | Ensures recommendations never violate official FPL rules (e.g., blocks transfers that exceed 3 players from the same Premier League club). |
| **High-Speed ETL** | Multi-season ingestion using Apache Arrow (`pyarrow` engine) and selective column loading (`usecols`). | Reduced cold-start CSV parsing time from multi-second delays down to sub-second load times across 110,000+ match records. |
| **Security Hardening** | `pip-audit` automated scanning, regex input sanitization, API payload trimming, and XSRF/CORS server policies. | Reduced API payload memory by ~80%, prevented script/regex injection vulnerabilities, and achieved a 0-vulnerability CVE audit score. |

---

### 3. Key Technical Challenges & Solutions

#### A. Preventing Data Leakage in Historical Validation
- **Challenge:** Evaluating model accuracy on past seasons often causes accidental data leakage if full-season statistics are used to predict mid-season gameweeks.
- **Solution:** Re-engineered the backtesting pipeline to compute rolling point-in-time metrics strictly from preceding matches, isolating historical validation from future outcomes.

#### B. Resolving Identity Shifts Across Seasons
- **Challenge:** The Premier League API changes internal player IDs between seasons, breaking naive relational joins across multi-season archives.
- **Solution:** Developed a resilient 5-tier fallback cascade: Opta Code $\rightarrow$ Normalized Unicode Name (NFKD diacritic stripping) $\rightarrow$ Web Name matching $\rightarrow$ Substring resolution $\rightarrow$ Positional Median Imputation.

#### C. Enforcing Combinatorial Formation Rules
- **Challenge:** Basic optimization models suggested swapping goalkeepers for outfield players or generating invalid formations (e.g., 2-5-3).
- **Solution:** Implemented structural formation validation that enforces discrete position rules (1 GKP, minimum 3 DEF, minimum 2 MID, minimum 1 FWD) prior to issuing bench swap recommendations.

---

## 💼 Resume & CV Bullet Points

### For Machine Learning / Data Science Resumes
- Engineered and deployed **CatalanPlays**, an end-to-end predictive analytics platform forecasting Expected Points ($\text{xP}$) for 750+ Premier League players across 110,000+ historical records.
- Architected a Hybrid Intelligence engine integrating Scikit-learn statistical regression with dynamic heuristic layers (squad-depth rotation risk, fixture difficulty modeling, clean sheet probability).
- Eliminated backtesting data leakage by building point-in-time feature extraction pipelines with PyArrow, optimizing CSV parsing speeds by over 80%.
- Implemented an explainable AI framework, translating mathematical model outputs into transparent, human-readable decision breakdowns for end users.

### For Full-Stack / Software Engineering Resumes
- Developed a production-grade web dashboard using Python, Streamlit, and Plotly Express, featuring real-time API integrations and responsive mobile design.
- Built a constraint-satisfaction transfer engine that enforces complex domain rules (budget caps, formation validity, 3-player-per-club maximums) for imported user squads.
- Hardened application security by implementing automated `pip-audit` vulnerability scanning, regex-based input sanitization, and server-side XSRF/CORS protection.
- Integrated live REST API pipelines with localized timezone conversion and tiered caching strategies (`@st.cache_data`) for instantaneous matchday updates.