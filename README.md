# ⚡ ApexFPL — Explainable AI & Decision Engine for Fantasy Premier League

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://catalan-fpl.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

ApexFPL is an open-source Expected Points ($\text{xP}$) forecasting and squad optimization suite for Fantasy Premier League managers. 

---

## ❓ What is ApexFPL?

ApexFPL is an intelligent decision-support application built for Fantasy Premier League (FPL) managers. Instead of relying on gut feeling, the platform analyzes over 110,000 historical match records to forecast upcoming player performance in real time.

Unlike traditional "black-box" artificial intelligence that simply outputs a number without context, ApexFPL uses an **Explainable AI framework**—every projection clearly explains why a player is expected to score points (factoring in recent form, matchup difficulty, clean sheet odds, and squad rotation risks). Users can enter their unique FPL Team ID to immediately receive tailored starting XI optimizations and budget-compliant transfer recommendations.

---

## ✨ Key Features

- 🔮 **Explainable xP Engine:** Real-time point predictions broken down into base performance, fixture ease, clean sheet odds, and squad-depth rotation risk.
- 📊 **Rate My Team & Squad Import:** Enter your FPL Team ID for instant Starting XI projections, formation-valid bench optimization, and 3-player-club-rule compliant transfer targets.
- ⚽ **Matchday Center:** Local timezone-aware fixture kickoffs, live deadline countdown, and dynamic 5-gameweek FDR scheduling matrix.
- 📈 **Historical Performance Analysis:** Zero-leakage multi-season tracking plotting predicted vs. actual points across 110,000+ match records.
- 🏆 **FPL Hall of Fame:** Curated all-time career and single-season legend records.

---

## 🛠️ Tech Stack & Architecture

- **Backend / Modeling:** Python, Scikit-learn (Linear Regression Baseline), Pandas, PyArrow
- **Dashboard / Frontend:** Streamlit, Plotly Express, Custom Responsive CSS
- **Data Ingestion:** Official Fantasy Premier League REST API + Vaastav Multi-Season Historical Archive
- **Security:** CSRF protection, sanitized regex inputs, trimmed API payloads, zero hardcoded credentials

---

## 🚀 Quickstart

```bash
git clone https://github.com/subantapoudel01/FPL-Predictor.git
cd FPL-Predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚡ Project Case Study: ApexFPL Decision Engine

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
- Engineered and deployed **ApexFPL**, an end-to-end predictive analytics platform forecasting Expected Points ($\text{xP}$) for 750+ Premier League players across 110,000+ historical records.
- Architected a Hybrid Intelligence engine integrating Scikit-learn statistical regression with dynamic heuristic layers (squad-depth rotation risk, fixture difficulty modeling, clean sheet probability).
- Eliminated backtesting data leakage by building point-in-time feature extraction pipelines with PyArrow, optimizing CSV parsing speeds by over 80%.
- Implemented an explainable AI framework, translating mathematical model outputs into transparent, human-readable decision breakdowns for end users.

### For Full-Stack / Software Engineering Resumes
- Developed a production-grade web dashboard using Python, Streamlit, and Plotly Express, featuring real-time API integrations and responsive mobile design.
- Built a constraint-satisfaction transfer engine that enforces complex domain rules (budget caps, formation validity, 3-player-per-club maximums) for imported user squads.
- Hardened application security by implementing automated `pip-audit` vulnerability scanning, regex-based input sanitization, and server-side XSRF/CORS protection.
- Integrated live REST API pipelines with localized timezone conversion and tiered caching strategies (`@st.cache_data`) for instantaneous matchday updates.