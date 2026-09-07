# CatalanPlays — Comprehensive Project Analysis

**As of:** 2026-09-06, commit `d6463b8` + uncommitted working-tree changes (see §12)
**Live app:** `apexfpl.streamlit.app` (URL not yet renamed to match the CatalanPlays brand)
**Repo:** `github.com/subantapoudel01/FPL-Predictor`

This document describes the project exactly as it stands today — mechanism by mechanism, not aspirationally. Where something is a heuristic rather than a measured fact, that's said plainly.

---

## 1. What this project is

A **Fantasy Premier League (FPL) decision-support web app**: it predicts each Premier League player's expected points ("xP") for the upcoming gameweek, explains *why* in plain arithmetic, and gives an FPL manager a handful of concrete decisions — who to captain, what to change, whether their bench is optimal.

Concretely, it is:
- A **Streamlit single-page app** (`app.py`, 1,342 lines) with four tabs
- Backed by a **hybrid prediction engine**: one small trained statistical model (`src/predictor.py`) plus a hand-written layer of football-domain rules on top of it
- Reading **live data** from the official (but unofficial-to-use) FPL REST API on every page load, cached for an hour
- Supplemented by a **historical data pipeline** pulled from a community-maintained GitHub archive, used for both a "how did last gameweek's predictions do" recap and a multi-season player-history browser
- A **solo, personal project** by Subanta Poudel — started as CU6051NP university coursework, since stripped of academic framing and rebranded as an independent product

## 2. What this project is *not*

Stated explicitly because each of these has come up as a real point of confusion during development:

- **Not a machine-learning showcase.** The core model is a 7-feature `LinearRegression` — deliberately simple, not gradient boosting or a neural net. An earlier audit in this project's own history found that a single feature (minutes played) recovers ~91% of the trained model's R². The accuracy comes mostly from the rule layer's football-domain knowledge, not model sophistication.
- **Not a price/transfer-market predictor.** Uses each player's *current* FPL price; there is no model of future price rises/falls (the live "Price Δ (Today)" column shows what already happened, not a forecast).
- **Not a multi-gameweek squad optimizer.** "Rate My Team" suggests a *single* sell/buy for the current gameweek by greedy budget/eligibility search — it is not a linear-programming optimizer planning transfers across several future gameweeks or around chip strategy (Wildcard, Bench Boost, etc.).
- **Not using third-party expected-goals data.** All "expected" stats (`expected_goal_involvements_per_90`, etc.) come from FPL's own official API fields — there is no Understat/FBref scraping or external xG provider.
- **Not accounting for non-Premier-League competitions.** No Champions League/cup fatigue modelling (stated in-app under Documentation → Limitations).
- **Not authenticated or multi-user.** No accounts, no login, no server-side saved squads. The one piece of "personalization" is a bookmarkable URL (`?team_id=...`) for the Rate My Team section — anyone with that link sees that squad; there's no access control.
- **Not commercially licensed.** Built entirely against the FPL API without a documented public license for that usage, and not affiliated with, endorsed by, or licensed by the Premier League or Fantasy Premier League in any way.
- **Not running on infrastructure that guarantees uptime or persistence.** Deployed on Streamlit Community Cloud's free tier: it sleeps after inactivity (needs a manual "wake up" click), and its local filesystem is wiped on every redeploy — which is precisely why the recap feature (§7) was redesigned to need no local storage at all.
- **Not using granular team attack/defence strength data**, even though the FPL API nominally has fields for it (`strength_attack_home/away`, `strength_defence_home/away`) — verified today that **all 20 teams report 0** for those fields this early in the season (FPL's own algorithm hasn't populated them yet). The dynamic clean-sheet model (§7) currently runs on the coarser but always-populated `strength_overall_home/away` instead, with the granular fields wired in to be preferred automatically the moment FPL starts populating them.

## 3. Identity history

The project has been renamed twice in its lifetime, visible in git history and in this engagement:
1. **"Catalan FPL Predictor"** — original coursework name (CU6051NP, London Met)
2. **"ApexFPL"** — first rebrand, generic energy/lightning branding (⚡, neon green `#00FF87` on near-black)
3. **"CatalanPlays"** (current) — tied back to the developer's actual personal handles (`catalansubanta`, `catalanplays`), because "Apex" "didn't link" to the developer at all. Visual identity became **"Blaugrana"**: navy ground (`#0F1420`), garnet accent (`#D64B5E`), gold (`#FFC72C`) reserved specifically for captain/star emphasis — a deliberate nod to Catalan football culture (Barça's colours) rather than a generic dark-mode SaaS palette. Typography: Big Shoulders Display (headings), Source Sans 3 (body), IBM Plex Mono (all numeric stats).

The old Docs/coursework files (Word/PPT submissions, Turnitin report) were deliberately removed from the public repo in a past commit ("Production release: ... removed coursework files") to separate the public product from the academic submission.

## 4. Architecture

```
Live FPL API ──┐
               ├─→ src/data_loader.py (whitelist-trims payload) ─→ src/features.py (process_data)
Fixtures API ──┘                                                         │
                                                                          ▼
                                                          df with engineered features,
                                                    cs_probability, DEFCON, price/ownership, etc.
                                                                          │
                                                                          ▼
                                                     src/predictor.py (make_predictions)
                                                     Linear Regression → raw_xp
                                                     + expert rule layer → final_xp
                                                     + captaincy Ceiling Score
                                                                          │
                                                                          ▼
                                                                       app.py
                                                        (4 tabs, all rendering logic, all UI state)

Separately: vaastav GitHub historical CSVs ─→ src/features.py (fetch_full_history)
            → non-leaky walk-forward `raw_xp` reconstruction
            → used by Player History tab AND the Last Gameweek Recap
```

No database, no backend server beyond Streamlit itself, no message queue, no background jobs. Every "live" number is fetched fresh (subject to caching) on each page load/rerun.

## 5. Data sources — three distinct ones, each with a different trust model

**A. Live FPL REST API** (`src/data_loader.py::fetch_api_data`, `fetch_gameweek_history`, `fetch_player_live_summary`, `fetch_user_team`)
- Endpoints used: `bootstrap-static/` (players, teams, gameweek events), `fixtures/`, `event/{gw}/live/`, `element-summary/{id}/`, `entry/{id}/event/{gw}/picks/`
- The player/team/event/fixture payloads are **deliberately trimmed to an explicit field whitelist** before anything else touches them — stated in-code as reducing both memory and "security exposure." This has bitten the project twice already: two separate live features (DEFCON, then bonus/price/ownership, then penalty/xGI data) were implemented correctly in `features.py`/`predictor.py` but silently produced zeros until someone remembered to also add the new field name to this whitelist in `data_loader.py`.
- Cached 1 hour (`@st.cache_data(ttl=3600)`).

**B. Historical per-gameweek CSVs** (`src/features.py::fetch_full_history`), from `github.com/vaastav/Fantasy-Premier-League` — a well-known community-maintained archive, covering 2022-23 through 2025-26.
- This is the **only current source for the Last Gameweek Recap and the Player History tab**.
- Rolling features (`rolling_3_minutes`, etc.) are computed via `.shift(1).rolling(3)` grouped by player+season — genuinely non-leaky, walk-forward-only features, unlike a since-removed "Backtest" tab that fed today's cumulative stats into a past gameweek's prediction (a real, previously-shipped, now-fixed data-leakage bug).
- A real bug was found and fixed in this pipeline this week: the raw CSV has ~419 duplicate `(name, gameweek)` rows from genuine **double-gameweeks** (a club playing twice in one FPL gameweek), and the code used to silently `drop_duplicates(keep='first')`, discarding one entire fixture's points. Now aggregated (summed) instead.
- Cached 1 hour.

**C. Local committed CSVs** (`Data/Cleaned/model_ready_data.csv` and siblings) — used **only** as the pre-season baseline source, before Gameweek 1 has started, via a 5-tier matching cascade: Opta `code` → normalized full name → normalized first+web name → normalized web name → substring match → positional-median imputation for brand-new/promoted players. This exists because the live API resets cumulative per-gameweek stats at the start of a season, so there's nothing live to compute rolling features from yet.

## 6. The prediction model

- **Algorithm:** `sklearn.linear_model.LinearRegression`, trained offline in `Notebooks/3_Model_Training.ipynb`, serialized to `Models/linear_reg_v1.pkl`, loaded via `joblib`.
- **7 input features**, all rolling-3-gameweek averages plus current price: `rolling_3_minutes`, `rolling_3_ict_index`, `rolling_3_creativity`, `rolling_3_influence`, `rolling_3_threat`, `rolling_3_total_points`, `value`.
- **Training data:** 2021-22 through 2024-25 seasons; tested on early 2025-26.
- **Reported metrics** (`metrics.json`): RMSE 2.16, R² 0.196 — but this was measured on a small, early-season slice of 2025-26. A full-season backtest (documented in this project's own `FUTURE_PLAN.md` from an earlier audit) found R² closer to 0.25–0.27 depending on the split, and — the important caveat — that **`rolling_3_minutes` alone recovers about 91% of that R²**. In plain terms: the statistical model is mostly a sophisticated "is this player actually going to play" detector, not a goal/assist predictor. `test_rules.py` exists specifically to quantify the rule layer's added value against this raw model, run as `python test_rules.py --season <season>`.
- The model's raw output (`raw_xp`) is never shown to the user directly — only `final_xp`, after the rule layer below.

## 7. The expert rule layer (`src/predictor.py::make_predictions`)

Applied per-player, in this order, to turn `raw_xp` into `final_xp`:

1. **Hard availability wipe** — if `status` is `i` (injured) or `s` (suspended), or `chance_of_playing_next_round == 0`: **final_xp = 0.0** immediately, reasoning `"0.0 pts (Injured/Suspended)"`. Status `u` (unavailable — e.g. out on loan) also wipes to 0, labelled `"Unavailable"` separately since "Injured/Suspended" would misdescribe it.
2. **Fixture difficulty bonus — MID/FWD only**: `+1.25` (FDR 1) / `+0.75` (FDR 2) / `−0.5` (FDR 4) / `−0.75` (FDR 5). *Not* applied to DEF/GKP any more — see clean sheet odds below.
3. **Premium attacker multiplier**: `×1.15` if price ≥ £10.0m and position is MID/FWD.
4. **Star captain bonus**: flat `+1.0` if price ≥ £10.0m and fixture difficulty ≤ 3.
5. **Dynamic clean sheet odds — DEF/GKP only** (new this week, replacing a flat "+0.5 if FDR ≤ 2" step function): `cs_xp = cs_probability × 4.0`, where `cs_probability` is computed in `features.py` from each team's `strength_overall_home/away` rating vs. their specific opponent's, clamped to a **0.05–0.55** range around a 0.30 league-average baseline. Reasoning shows e.g. `"+ 1.5 (38% CS Odds)"`.
6. **DEFCON bonus** (2025-26's real new scoring rule): stepped `+2.0 / +1.0 / +0.4` based on `defensive_contribution_per_90` (FPL's own season rate for combined clearances+blocks+interceptions+tackles, or +recoveries for MID/FWD) against the actual rule threshold (10 for DEF/GKP, 12 for MID/FWD). This is the single biggest reason budget defenders can still out-rank premium attackers in the main table (see §13).
7. **Squad-depth rotation tax**: `−25%` applied only to non-nailed (starter probability < 90% and price < £9.0m) MID/FWD players at clubs computed to have 5+ other active attacking assets priced ≥ £5.5m — a dynamic stand-in for "this club rotates its attack," recomputed from live squad data every run rather than a hardcoded list of "big clubs."
8. **Starter-probability multiplier**: a stepped curve on minutes-per-game (100% if ≥60 min/game, 75% if ≥45, 25% if ≥30, 5% otherwise) multiplies the whole subtotal.
9. **Partial-injury-doubt multiplier**: if `chance_of_playing_next_round` is strictly between 0 and 100 (i.e. 25/50/75%), multiplies by that fraction directly (`0.25`/`0.5`/`0.75`).
10. Final result clipped to `[0.0, 18.0]`.

Every one of these steps is echoed into a human-readable `reasoning` string shown in the UI, e.g. `"5.6 pts = 3.2 (base xP) + 0.75 (easy fixture) × 1.15 (premium attacker form) + 1.0 (star captain)"`.

## 8. The captaincy engine — a *separate* ranking, not just top-of-list

`render_captaincy_card` in `app.py` does **not** simply take the top-5 by `final_xp`. It:
- **Filters eligibility** first: MID/FWD only, unless a DEF has real `expected_goal_involvements_per_90 > 0.30` (an actual attacking full-back, not a proxy metric).
- **Ranks by a separate Ceiling Score**, computed in `predictor.py`: `final_xp × (1.25 if price ≥ £9.5m) × (1.15 if penalties_order == 1)`.
- Deliberately does **not** hardcode "always show Haaland/Salah/Saka" — those names surface *because* the formula rewards being a genuine premium penalty-taker, which is more robust than a name list that goes stale the next time a club's penalty taker or a player's price changes.
- This ranking is independent of the pitch-visualizer's own captain/vice-captain badges, which still use plain `final_xp` within that specific optimal-XI selection (a separate, narrower mechanism — not yet unified with the Ceiling Score, since it was out of scope when built).

## 9. Features, tab by tab

**Tab 1 — 🔮 Predictions** (the main tab)
- **Last Gameweek Recap**: predicted-vs-actual table + RMSE/players-tracked/top-performer metrics for the most recently *finished* gameweek, reconstructed entirely from the historical CSV pipeline (§5B) — no local storage, survives redeploys, and honestly reports "recap unavailable" rather than fabricating one if the source data isn't published yet.
- **AI Team of the Week**: an optimal starting XI across all 8 valid FPL formations, selected via a club-cap-aware greedy fill (max 3 players per club — a real bug here was fixed this week: the old version sliced a fixed top-N per position independently and could reject every formation outright if one club dominated several positions at once), rendered as styled player cards on a pitch graphic with gold (C) / silver (V) badges.
- **Top Captain Picks**: the 5-row Ceiling-Score-ranked table described in §8, with star ratings and a plain-English rationale column (home/away, FDR band, penalty-taker flag, price tier, or — for eligible defenders — their specific xGI/90 number).
- **Predicted Points table**: a clean 6-column core view (Player, Team, Opponent, Pos, Predicted Points, Reasoning) with Position/Team filters and a search box; secondary stats (DEFCON/90, Bonus, BPS, Price, Price Δ today, Ownership %, FDR) live in a collapsed "📊 View Underlying Defensive & Bonus Metrics" expander rather than cluttering the main table.
- **5-Gameweek FDR Ticker**: all 20 clubs' next 5 opponents, colour-coded by FPL's own difficulty rating, sorted by easiest run.
- **Rate My Team**: enter an FPL Team ID (now **persisted to the URL** as `?team_id=...` so the dashboard is bookmarkable) to get: total projected starting-XI points, the weakest starting link, a bench-optimization alert (is a benched player projected higher than a starter, formation-legality checked), one recommended sell→buy transfer (budget- and 3-per-club-aware), and the full 15-man squad breakdown with role badges (Captain/Vice/Starter/Bench).

**Tab 2 — 📊 Player History**: pick any player and season, see a predicted-vs-actual line chart and table across their whole history, built from the same non-leaky historical pipeline, with a live-season overlay computed from the FPL element-summary endpoint for the current season's most recent gameweeks.

**Tab 3 — 🏆 Hall of Fame**: **fully static, hand-curated data** (Rooney/Lampard/Salah career totals, Salah/Suárez/Ronaldo single-season records) — not live, not model-derived, not connected to any data source. Purely a content/nostalgia feature.

**Tab 4 — ℹ️ Documentation**: in-app architecture summary, a capabilities/limitations list (the limitations list is honest — it says the expert heuristic layer trades some raw accuracy for better top-20 ranking utility), and credits.

**Sidebar**: system status (which gameweek is active / pre-season), timezone selector (used to localize deadlines and kickoff times across the whole app), model diagnostics (RMSE/R² from `metrics.json`).

## 10. Branding & design system

- Font stack loaded via Google Fonts `<link>`, injected once near the top of `app.py` with CSS custom properties (`--font-display`, `--font-body`, `--font-mono`) and `!important` overrides against Streamlit's own emotion-generated classes.
- A real bug was hit and fixed here: an early version of the font CSS included a blanket `span` selector, which broke Streamlit's icon glyphs (rendered as ligature text in a dedicated icon font) — expander chevrons showed as the literal text `arrow_right`. Fixed by excluding `span` from the override.
- Color tokens: navy ground/surface, garnet accent, gold for captain/star/heading emphasis specifically (not used generically), with **semantic colors kept deliberately separate from the brand palette** — the FDR ticker's green/purple/red/pink bands, and the Sell(red)/Buy(green) transfer tags, were left untouched during the rebrand on purpose, so brand and meaning never collide.
- All custom HTML/CSS blocks route through a single `render_html()` helper that strips leading whitespace per line — necessary because Streamlit's markdown renderer runs CommonMark first, and any line indented 4+ spaces gets treated as a preformatted code block, printed as literal text instead of parsed as HTML. This exact bug (raw CSS dumping onto the page as visible text) was found and fixed in the pitch visualizer, the transfer recommendation card, and the footer.

## 11. Security posture

- `security_audit.py` runs `pip-audit` against `requirements.txt` for known CVEs (a manual/CI script, not run automatically on every deploy).
- Search input is sanitized (`sanitize_search_input` strips to alphanumeric+spaces, and the search itself uses `re.escape`) before being used in a regex `.str.contains()` filter, specifically to prevent regex injection or crashes from adversarial search input.
- The live API payload is trimmed to an explicit field whitelist before any processing touches it (stated purpose: reduce both memory footprint and "security exposure" from an otherwise much larger, unvetted third-party JSON blob).
- `.gitignore` covers `.env`/`.env.*`, `.streamlit/secrets.toml`, `.venv/`, `__pycache__/`, `.DS_Store`, and now new (untracked) `*.csv` files — though the 6 CSVs already committed to the repo remain tracked (a `.gitignore` rule never retroactively untracks a file already in git).
- No authentication, no secrets currently in use by the app (it calls only unauthenticated public FPL endpoints), so there is no credential-handling surface today.

## 12. Bugs found and fixed during this engagement (chronological)

This list matters because several of these were subtle and could easily still be present in a similar codebase — worth knowing what class of bug to watch for:

1. Streamlit markdown-as-code-block bug leaking raw CSS/HTML onto the page as visible text (pitch visualizer, transfer card, footer).
2. Captaincy table showing literal `**bold**` markdown syntax — `st.dataframe` never interprets markdown inside cells.
3. **Gameweek-resolution bug**: `is_next` was checked before `is_current`, so the app jumped to predicting *next* week's gameweek the instant a deadline passed — even while the current gameweek's matches hadn't been played yet. Fixed to prioritize `is_current`. This bug existed in two independent places (`features.py` and a duplicate lookup in `app.py`'s deadline banner) and both needed the fix for the UI to stay internally consistent.
4. **Team-of-the-week club-cap rejection**: greedy top-N-per-position selection could produce >3 players from one club in every one of the 8 candidate formations simultaneously, causing the whole feature to silently render nothing. Fixed with a club-cap-aware greedy fill.
5. Data field whitelist in `data_loader.py` silently dropping newly-added fields (DEFCON, then bonus/price/ownership, then penalty/xGI data) each time a new feature was added elsewhere — three separate incidents of the same root cause.
6. An earlier **local-JSON-snapshot recap design** was built, then deliberately replaced this week after recognizing it would be wiped on every Streamlit Cloud redeploy — rebuilt on the historical-CSV pipeline instead, which needs no persistence at all.
7. **Double-gameweek data loss**: `drop_duplicates(keep='first')` on the historical CSV silently discarded one full fixture's points/minutes whenever a club played twice in one FPL gameweek. Fixed via sum-aggregation.
8. Budget-defender bias in captaincy suggestions (a nailed £4.5m defender with an easy fixture could outrank Haaland) — fixed via the eligibility filter + Ceiling Score in §8. A **related, only partially resolved** version of this bias still exists in the *main* predictions table due to DEFCON's flat scoring (see §13).
9. A font-CSS regression that broke Streamlit's own icon rendering (`span` override clobbering the icon font) — caught and fixed same-session.

## 13. Known limitations, stated honestly

- **DEFCON can still let budget defenders outrank premium attackers in the main table.** Confirmed today: Haaland (5.59 xP) now correctly edges the specific defender (Ajer, 5.57 xP) originally flagged as an example of this bias — but two *other* sub-£5m defenders (Justin 5.97, Muharemović 5.88) still rank above Haaland, because DEFCON's flat `+2.0` cap is exactly as reachable for a £4.5m defender as for anyone. This was not part of the most recent fix's scope (which targeted availability handling and clean-sheet modelling specifically) and remains open.
- **Clean sheet odds run on a coarser team-strength signal than intended**, because FPL hasn't populated the granular attack/defence strength fields yet this season (§2). The code is already written to prefer those fields automatically once they go live — no further change needed when that happens.
- **The pitch-visualizer's captain/vice badges and the Top Captain Picks card can disagree** — they're two different mechanisms (plain `final_xp` for the pitch, Ceiling Score for the captaincy card) that were never unified.
- **The historical recap compares `raw_xp` (the bare statistical model), not `final_xp`** (the full rule-adjusted number shown live) — reconstructing the entire rule layer's historical context (what the fixture difficulty, rotation risk, and DEFCON rate *were* for that specific past gameweek) was judged a larger undertaking than reusing data already fetched for the Player History tab. Stated plainly in the recap's own docstring.
- **Rate My Team's transfer suggestion is single-gameweek and greedy** — it does not consider saving a free transfer, taking a hit, or planning around a future fixture swing.
- **The rule layer's constants are hand-tuned, not fitted.** Thresholds like "£9.5m = premium," "£10.0m = star bonus," "5+ attackers = deep squad" are domain judgment calls, not statistically optimized cutoffs. `test_rules.py` exists to at least measure whether the whole rule layer helps or hurts against the raw model on held-out data.

## 14. Tech stack

Python · Streamlit ≥1.35 · pandas ≥2.2 · numpy ≥1.26 · scikit-learn ≥1.4 · joblib · plotly ≥5.18 · pyarrow ≥15.0 (fast CSV parsing for the historical pipeline, with a plain-pandas fallback if unavailable) · requests · pip-audit (dev/audit only). No JavaScript, no separate frontend framework — Streamlit renders the entire UI server-side.

## 15. Repository layout

```
app.py                  1,342 lines — the entire UI + orchestration
src/
  data_loader.py          195 lines — API fetching, payload trimming, timezone utils
  features.py             399 lines — process_data (live), fetch_full_history (historical)
  predictor.py            210 lines — model + rule layer + captaincy scoring
Models/linear_reg_v1.pkl        — trained LinearRegression, joblib-serialized
Data/Raw/, Data/Cleaned/        — committed historical CSVs (pre-season baseline source only)
Notebooks/                      — cleaning, feature engineering, training, prediction (offline, not run by the app)
metrics.json                    — {rmse, r2, notes} shown in the sidebar
test_rules.py                   — standalone script: does the rule layer help vs. raw_xp?
security_audit.py               — standalone script: pip-audit wrapper
.streamlit/config.toml          — Blaugrana theme tokens
FUTURE_PLAN.md                  — an earlier full audit/roadmap document (pre-dates several fixes above)
```

## 16. Current state (uncommitted)

As of this analysis, `git status` shows **uncommitted changes** to `.gitignore`, `app.py`, `src/data_loader.py`, `src/features.py`, and `src/predictor.py` — this is the strict-availability + dynamic-clean-sheet-odds work from the most recent request, verified working locally but not yet pushed to `main` or deployed. The live app at `apexfpl.streamlit.app` currently reflects commit `d6463b8`, one commit behind this.

## 17. Honest bottom line

This is a real, working, moderately sophisticated hobby/portfolio project with a genuinely interesting hybrid architecture (statistical model + explainable domain-rule layer) and an unusually high density of *fixed, documented, verified* bugs for its size — a sign it's been iterated on carefully rather than left to rot. Its accuracy ceiling is set by a deliberately simple linear model, not by the rule layer, which is honestly disclosed in the app's own Documentation tab. It is not, and doesn't claim to be, a production-grade commercial product: no auth, no persistence, no SLA, unofficial data source, single maintainer.
