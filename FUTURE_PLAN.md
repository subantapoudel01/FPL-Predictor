# FPL Points Predictor — Technical Audit & Future Plan

**Audit date:** 2026-08-18
**Auditor:** Code + empirical model review of the repository as committed at `0847b83`
**Subject:** Catalan FPL Predictor (CU6051NP AI Coursework), Subanta Poudel

---

## 0. Verdict at a glance

| Dimension | Score | Comment |
|---|---|---|
| As coursework | **8 / 10** | Clean structure, real live API, honest limitations tab, working backtest UI. Deserves a good grade. |
| Modelling rigour | **5 / 10** | Correct temporal split and no target leakage in training — but collinear features, one wasted gameweek of signal, and a headline metric measured on a 6-GW test set. |
| Production readiness | **2 / 10** | Currently emits garbage predictions in the live app (see C-1). Will not boot on Linux (C-2). |
| Product potential | **4 / 10** | Real market exists, but this needs data sources it does not yet have to compete. |

**Bottom line:** this is a solid academic project with a genuinely working end-to-end pipeline, and the underlying model has real (if modest) predictive skill — measured properly it explains ~26% of next-gameweek points variance and its top-20 picks average ~4.8 pts vs a ~1.2 league average. But the shipped app is broken right now, and the gap between "works" and "a product people pay for" is a data problem, not a model problem.

---

## 1. Current features

### Working today
- **Live data ingestion** — pulls `bootstrap-static` and `fixtures` from the official FPL API with timeouts and error handling (`src/data_loader.py`).
- **Hybrid architecture** — a trained Linear Regression base model plus a hand-written expert-rules layer (`src/predictor.py`). This is a defensible design and the most interesting part of the project.
- **Expert rule layer** implementing:
  - Appearance points floor tied to starter probability
  - Fixture Difficulty Rating adjustments (−1.0 to +2.5)
  - Clean-sheet boost for DEF/GKP against weak opposition
  - "Premium captain" boost for £10m+ assets with good fixtures
  - "Pep roulette tax" — 25% haircut on sub-£9m attackers at MCI/ARS/LIV/CHE
  - Injury multiplier from `chance_of_playing_next_round`
  - Rotation multiplier from a stepped minutes curve
- **Streamlit dashboard** — position filter, player search, top-5 highlighting, sortable prediction table.
- **Backtest tab** — replays a chosen gameweek against `event/{gw}/live/` and reports RMSE.
- **Opponent + difficulty display** with blank-gameweek handling (`"-"` fallback).
- **Model diagnostics sidebar** reading `metrics.json`.
- **Reproducible pipeline** — four notebooks covering cleaning → feature engineering → training → prediction.
- **One-click launcher** (`run__app.bat`) and pinned pre-trained model for examiners.

### Honestly documented limitations (already in the app)
No price-change forecasting, no cup/European fatigue modelling, backtest uses current injury status for past weeks.

---

## 2. How accurate is it, really?

### 2.1 The shipped number is misleading in both directions

`metrics.json` reports **RMSE 2.16, R² 0.196**. That was measured on the 2025-26 season, which in this dataset contains **only gameweeks 1–6 (4,329 rows)**. A six-gameweek test set is far too small and too early-season to be representative.

Re-running the identical model against a **full held-out season (2024-25, 9,927 rows, train = 2021-24)**:

| Model | RMSE ↓ | R² ↑ | Spearman ↑ | Avg actual pts of its top-20 picks ↑ |
|---|---|---|---|---|
| **Current shipped LinearRegression** | **2.070** | **+0.255** | 0.643 | **4.77** |
| GBM, identical 7 features | 2.063 | +0.260 | 0.656 | 4.44 |
| GBM + lag/bps/position features | 2.048 | +0.271 | 0.670 | 4.61 |
| Two-stage: P(start) × pts&#124;start | 2.064 | +0.259 | 0.663 | 4.57 |
| *League-average player* | — | — | — | *1.23* |
| *Perfect foresight (ceiling)* | — | — | — | *10.81* |

**The model has real skill.** Its top-20 recommendations return ~3.9× the points of an average player. Spearman 0.64 means it ranks players sensibly.

### 2.2 The uncomfortable finding: the features are the ceiling, not the model

On the 2025-26 slice, benchmarking the model against trivial baselines:

| Predictor | RMSE | R² |
|---|---|---|
| Full 7-feature LinearRegression | 2.161 | +0.196 |
| **Calibrated `rolling_3_minutes` alone** | **2.184** | **+0.178** |
| Calibrated recent form alone | 2.216 | +0.154 |
| Predict the training mean | 2.409 | 0.000 |

**A single feature — average minutes played — recovers 91% of the model's R².** The other six features add ~1%. In effect the model is a sophisticated availability detector.

This is confirmed by the architecture experiments above: swapping Linear Regression for gradient boosting moves R² from 0.255 → 0.260. Adding eight richer lag features moves it to 0.271. **Model sophistication buys almost nothing.** Any real accuracy gain must come from new information sources (§5), not new algorithms.

### 2.3 The coefficients are not interpretable

```
rolling_3_minutes        +0.020
rolling_3_ict_index      -1.013   ← negative
rolling_3_creativity     +0.111
rolling_3_influence      +0.102
rolling_3_threat         +0.104
rolling_3_total_points   +0.129
value                    +0.033
intercept                -1.222
```

ICT Index is by definition a composite of Influence, Creativity and Threat. Feeding all four into an unregularised linear model creates severe multicollinearity, which is why ICT lands on a large **negative** coefficient. Predictions still work (collinearity hurts interpretation, not fit), but any claim in the report about "ICT being an important driver" is unsupportable. **Do not present these coefficients as feature importances.**

### 2.4 Structural consequence: 36.5% of raw predictions are negative

The unclamped model outputs negative expected points for over a third of players, rescued only by `max(0, raw_xp)`. This is the expected behaviour of an unconstrained linear model on a target that is 61% zeros — but it means the model is fundamentally mis-specified for the shape of the data.

---

## 3. Flaws

### 🔴 Critical — the app is currently broken

**C-1. Live predictions are numerically meaningless right now.**

`src/features.py:24` sets `normalization_gw = max(1, target_gw)`, then divides season-cumulative API totals by it (`features.py:59`, `features.py:65`). During pre-season the FPL API still serves **last season's cumulative totals** and reports `is_current: []`, `is_next: 1`, so `normalization_gw = 1`.

Verified live output, today:

```
web_name   team  pos  starter_prob   raw_xp   final_xp
Raya       ARS   GKP        100      66.95      18.0
Gabriel    ARS   DEF        100      57.18      18.0
Saka       ARS   MID        100      54.29      18.0
Rice       ARS   MID        100      72.41      18.0
...all 12 top rows = 18.0, all Arsenal
```

Raya's 3,330 season minutes are being read as "3,330 minutes per game"; Saka's 847.6 season creativity total is fed to a model trained on 3-game averages of ~10. Everything saturates the `clip(0, 18)` ceiling at `predictor.py:92`, every top prediction reads exactly **18.0**, and the "top picks" are simply the first twelve Arsenal players by ID because `nlargest` breaks ties by index. **Anyone opening the app today sees confident, precise-looking, entirely fabricated output.**

**C-2. The app cannot start on Linux.** `src/predictor.py:9` builds the path as `os.path.join(base_dir, "models", ...)` — lowercase — but the directory on disk is `Models/`. Windows is case-insensitive so this passes locally; Streamlit Community Cloud, Docker, and every Linux host will fail to find the model and hit the "Critical Error: API or Model missing" branch. **This is a hard blocker for any deployment.**

**C-3. The backtest tab is scientifically invalid.** `app.py:114` calls `process_data(raw_json, target_gw=target_gw)` — passing *today's* API snapshot with a past gameweek number. Backtesting GW3 in April therefore builds features from 30 gameweeks of accumulated future data divided by 3. This is not "slightly leaky", it is total look-ahead contamination plus a 10× scale error. Any RMSE the tab reports is meaningless. The app's own limitations text ("uses current injury status for past weeks") massively understates the problem.

### 🟠 Major

**M-1. Train/serve skew.** Training features are true 3-game rolling means. Serving features are season-to-date totals ÷ gameweek number — a season average. These are different statistics with different variance, and the gap widens all season as season averages smooth out while 3-game means stay volatile. Worse, the six features are inconsistent with each other: `rolling_3_total_points` is served from the API's `form` field (a ~30-day average, roughly correct) while the other five use the season-average approximation.

**M-2. One gameweek of signal is thrown away.** In `2_Feature_Engineering.ipynb`, `rolling(3).mean().shift(1)` produces the mean of GW t−3…t−1, while the target is GW t+1. Gameweek **t is never used** — the single most predictive recent observation is discarded. Removing the extra `.shift(1)` (the rolling window already excludes the target) recovers it.

**M-3. Group-boundary leakage.** `df.groupby([...])[col].rolling(3).mean().shift(1).values` applies `.shift(1)` to the *flattened* result, so the first row of every player-season inherits the last rolling value of the preceding player. One contaminated row per group across ~94k rows — small, but it is real leakage and would be flagged in review. Use `.transform()` instead.

**M-4. `app.py:28` returns a 3-tuple into a 2-variable unpack.** `return None, None, None` against `raw_json, model = load_app_data()` raises `ValueError: too many values to unpack` — so an FPL API outage produces a raw Python traceback instead of the intended friendly error screen.

**M-5. The expert-rule layer has never been evaluated.** `metrics.json` measures `raw_xp` (the bare regression). What the app actually displays is `final_xp` after ~8 hand-tuned adjustments. **No number anywhere in this project tells you whether those rules help or hurt.** The sidebar labelling `R²` as "Accuracy" for a figure that doesn't describe the displayed output is, unintentionally, a misleading claim.

**M-6. Rule constants are unjustified magic numbers.** Where does 0.75 for the Pep tax come from? +2.5 for difficulty 1? +1.5 for premium captains? The `super_teams` list is hardcoded to `["MCI","ARS","LIV","CHE"]` and will rot within a season. Every one of these is a fitted parameter chosen by intuition rather than data.

**M-7. No dependency pinning.** `requirements.txt` lists six unversioned packages. The model was pickled with scikit-learn 1.4.2 and already emits `InconsistentVersionWarning` under 1.9.0 locally. A fresh deploy installing current scikit-learn risks silent behaviour change or an outright load failure.

### 🟡 Minor

- **F-1.** README instructs double-clicking `run_app.bat`; the actual file is `run__app.bat` (double underscore). The documented instructions fail.
- **F-2.** `app.py:53` reads `metrics.get('note')` but the JSON key is `notes` — the sidebar always shows the fallback `"v1.0"`.
- **F-3.** `st.cache_data` is used to cache a scikit-learn model object; `st.cache_resource` is the correct primitive for unhashable non-data objects.
- **F-4.** No `.gitignore`. A Word lock file (`Docs/~$...pptx`) is committed, and 75 MB of CSV plus coursework `.docx`/`.pptm`/PDF sit in version control — the `.git` directory is already 22 MB.
- **F-5.** `ict_rolling` is a leftover column from an earlier iteration.
- **F-6.** Zero tests. No CI. No linting.
- **F-7.** The Prediction notebook and `src/` implement *different* rule layers (notebook uses ×1.1 DEF / ×1.05 GKP multipliers and a `/75` starter curve; `src/` uses a stepped curve and additive bonuses). Two divergent sources of truth.
- **F-8.** Dataset coverage is uneven — 2024-25 stops at GW19 and 2025-26 at GW6, so the most recent and most relevant data is the thinnest.

---

## 4. What to fix, in order

### Phase 0 — Stop shipping wrong numbers (1–2 days)
1. **Fix C-1.** Compute per-gameweek rates from *actual gameweeks played*, not from the calendar gameweek index. Detect pre-season (`finished == 0` for all events) and either show last season's data explicitly labelled as such, or display an honest "season has not started" state. **Never display a number the pipeline cannot support.**
2. **Fix C-2.** Change `"models"` → `"Models"` in `predictor.py:9`, or better, `pathlib.Path(__file__).resolve().parents[1] / "Models" / "linear_reg_v1.pkl"`.
3. **Fix C-3.** Either rebuild the backtest on point-in-time historical CSVs (the honest fix), or disable the tab with a clear explanation. A silently invalid validation feature is worse than no validation feature.
4. Fix M-4, F-1, F-2. Pin all dependencies with `pip freeze`. Add a `.gitignore`.
5. Add a sanity assertion in the prediction path: if more than 5% of predictions hit the clip ceiling, refuse to render and raise an alert.

### Phase 1 — Make the numbers trustworthy (1 week)
6. **Fix M-2 and M-3** in feature engineering; retrain and re-measure.
7. **Evaluate the rule layer (M-5).** Backtest `raw_xp` vs `final_xp` on held-out seasons. Report both. Delete or retune any rule that does not earn its place. This is the single most valuable experiment available to you and it costs an afternoon.
8. **Replace the metric.** Report on a full held-out season, with rolling-origin cross-validation across seasons rather than a single split. Add the metrics that actually matter to an FPL manager: Spearman rank correlation, and mean actual points of the top-N recommended picks. RMSE on a 61%-zero target rewards timidity.
9. Relabel the sidebar honestly — R² is not "accuracy", and the displayed figure must describe the displayed prediction.
10. Drop `ict_index` or drop I/C/T; regularise with Ridge/ElasticNet. Then the coefficients become quotable.

### Phase 2 — Actually improve accuracy (2–4 weeks)
The experiments in §2.2 show algorithms are not the bottleneck. Attack the data:

11. **A dedicated minutes model.** Since minutes dominate the signal, model them properly: a classifier for P(start), P(60+ mins), P(cameo), trained on lineup history — not a four-step hardcoded curve.
12. **Underlying stats.** Ingest xG, xA, xGI, shots-in-box and key passes from Understat or FBref. Actual goals are extremely noisy over one gameweek; expected goals are the standard fix and the biggest single available upgrade.
13. **Team strength ratings.** Model each team's attack and defence separately (Dixon-Coles or an Elo variant) and derive expected clean-sheet probability from it, instead of trusting the FPL FDR — which is a crude editorial number, not a model output.
14. **Decompose the target.** Predict goals, assists, clean sheets, bonus (via BPS) and appearance points separately, then sum. This handles the 61% zero-inflation naturally, gives per-position models a chance, and makes every prediction explainable — "3.2 pts = 2 appearance + 0.7 goal + 0.5 CS".
15. **Set-piece and penalty duty** as explicit features — currently invisible to the model and worth substantial points.
16. Only after 11–15: revisit gradient boosting. It will help more once the features carry more information.

### Phase 3 — Product surface (4–8 weeks)
17. **Squad optimiser.** Given budget, current team and free transfers, recommend transfers under FPL's constraints (£100m, 3-per-club, valid formation) via linear programming. **This is where user value actually lives** — a ranked player table is a commodity; "here is your optimal move this week" is not.
18. Multi-gameweek horizon (predict GW+1 through GW+6) to support planning and chip timing.
19. Captaincy recommendations with uncertainty, not just point estimates.
20. Price-change prediction from net transfer momentum.
21. FPL team ID import so users see predictions for *their* squad.

---

## 5. Can this be a real product?

**Honest answer: not in its current form, and not by competing on prediction accuracy alone.**

### The market is real but crowded
FPL has ~11 million players. Established paid tools — FPLReview, Fantasy Football Hub, FPL Analytics, LiveFPL — already sell exactly this, at roughly £2–5/month, with years of accumulated data, richer feeds (Opta-grade underlying stats) and established communities. Free open-source alternatives exist too.

### Where this project stands against them
- Accuracy is in a reasonable amateur band but well behind tools using xG-based underlying data.
- The rule layer is a genuine differentiator in *concept* — an explainable hybrid — but is currently unvalidated and hand-tuned.
- There is no optimiser, no multi-gameweek horizon, no user account, no team import.

### Realistic paths to a product
- **Best odds: explainability as the wedge.** Competitors output a number. This architecture can output *"4.2 pts: 2 appearance + 1.1 goal threat + 0.6 clean sheet + 0.5 fixture bonus, −25% rotation risk."* Nobody in this market leads with legible reasoning. That is a defensible position, and Phase 2 item 14 is exactly the work that unlocks it.
- **Second: the optimiser.** Predictions are a commodity input; decisions are the product.
- **Third: keep it free and build reputation.** A well-run open-source FPL model with a public accuracy leaderboard builds a portfolio and an audience. This is the highest-value outcome relative to effort, and a genuinely strong career asset.

### Blockers you must clear first
1. **Legal.** The FPL API is undocumented and unofficial, with no public licence for commercial use. Premier League data and club/player names are protected IP. **Get this checked before charging anyone money.** Rebrand away from any Premier League trade dress.
2. **Liability.** If people pay for advice, expectations change. Terms of service, no-guarantee disclaimers, and — in some jurisdictions — care about anything adjacent to gambling.
3. **Personal branding.** "Catalan FPL Predictor", the London Met ID, and the coursework framing must all come out of a public product.
4. **Cost of goods.** Underlying-stats feeds are the main expense. FBref/Understat scraping is free but fragile and has its own terms; Opta-grade licensing is far outside hobby budget.

**Recommendation:** publish it free and open-source, compete on explainability and honesty about accuracy, build an audience, and only then consider monetising an optimiser tier. Do not try to out-predict FPLReview.

---

## 6. Publishing techniques

### Tier 1 — Free, ~1 hour (do this after Phase 0)
- **Streamlit Community Cloud** — connect the GitHub repo, deploy free. **Requires C-2 fixed** or it will not boot. Best effort-to-reward ratio available.
- **Hugging Face Spaces** — free Streamlit hosting, better ML-community discovery, no cold starts on the free tier.
- **Polish the GitHub repo** — this is the actual portfolio artifact. Add screenshots to the README (you already have them in `Images/`), an honest accuracy section with the benchmark table from §2, an architecture diagram, `.gitignore`, MIT licence, and topic tags (`fpl`, `machine-learning`, `streamlit`, `sports-analytics`).

### Tier 2 — Credibility (~1 week)
- **Move the coursework out.** Strip `Docs/` and the student ID from the public repo; keep the academic version on a private branch.
- **Public accuracy log.** Commit each gameweek's predictions *before* the deadline, then a scored post-mortem. Verifiable, honest, and instantly more trustworthy than any competitor's marketing claim. This is your best differentiator and it is nearly free.
- **Write it up.** A blog post — "I benchmarked my FPL model against predicting-the-mean and here's what I found" — with the §2.2 table is more compelling than a claim of success and demonstrates real rigour.
- **Add CI** — GitHub Actions running tests and lint on every push.

### Tier 3 — Real deployment (2–4 weeks)
- **Split the architecture.** FastAPI backend serving predictions + Next.js or React frontend. Streamlit does not scale past a handful of concurrent users and cannot be styled into a real product.
- **Containerise** with Docker; deploy to Railway, Render, or Fly.io (~$5–20/month).
- **Scheduled jobs** — a GitHub Action or cron that refreshes predictions after each deadline and writes to a database, instead of hitting the FPL API on every page load.
- **Persistence** — Postgres (Supabase free tier) for point-in-time snapshots. This also permanently fixes C-3: real backtesting requires stored historical snapshots, which is exactly what you lack today.
- **Caching** — Redis or a CDN in front of prediction endpoints.
- Custom domain, Plausible/Umami analytics, Sentry for error tracking.

### Tier 4 — Audience
- **r/FantasyPL** — very active, receptive to free open-source tools, hostile to unvalidated hype. Lead with the accuracy log, not the claim.
- **FPL Twitter/X** — post weekly predictions and score them publicly.
- **Kaggle** — publish the cleaned dataset and a notebook.
- **YouTube/TikTok** — weekly "what my model says" clips; strong organic reach in this niche.
- **Product Hunt** — only once there is an optimiser and a real frontend.

---

## 7. Recommended next three actions

1. **Fix C-1 and C-2 today.** The live app is currently showing fabricated 18.0-point predictions to anyone who opens it, and it cannot deploy anywhere but your laptop.
2. **Run the experiment in Phase 1 item 7** — does the expert-rule layer actually improve on the raw model? You have built an entire hybrid architecture and do not yet know the answer. One afternoon.
3. **Replace `metrics.json` with full-season, cross-validated numbers measured on what the app displays**, and publish the baseline comparison table honestly. A project that says "my model beats predict-the-mean by 26% and beats a minutes-only baseline by 1%" is far more impressive to a reviewer than one that quotes an unexplained R² of 0.196.

---

## Appendix — Reproducing the audit numbers

All figures in §2 come from `Data/Cleaned/model_ready_data.csv` (93,995 rows; 60.9% zero targets), with the shipped `Models/linear_reg_v1.pkl` and scikit-learn's `HistGradientBoosting*`. §2.1 trains on 2021-22 → 2023-24 and tests on 2024-25 (9,927 rows). §2.2 uses the project's own split (train 2021-25, test 2025-26, 2,842 rows) for comparability with `metrics.json`. Live output in C-1 was captured from `fantasy.premierleague.com/api` on 2026-08-18.
