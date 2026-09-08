# CatalanPlays — Comprehensive Project Analysis

**As of:** 2026-09-07, branch `feature/multi-model-group-tools` (commit `34e85e0`, PR #1 open against `main`)
**Live app:** `apexfpl.streamlit.app` (still on the pre-PR#1 commit; URL not yet renamed to match the CatalanPlays brand)
**Repo:** `github.com/subantapoudel01/FPL-Predictor`

This document describes the project exactly as it stands today — mechanism by mechanism, not aspirationally. Where something is a heuristic rather than a measured fact, that's said plainly. It fully supersedes the previous version of this file, which described a single-`LinearRegression`, four-tab app that no longer exists in this form.

---

## 1. What this project is

A **Fantasy Premier League (FPL) decision-support web app**: it predicts each Premier League player's expected points ("xP") for the upcoming gameweek, explains *why*, gives an individual manager concrete decisions (captain, transfer, bench), and — new this round — extends that to a **group of managers** (a mini-league) and to **external multi-gameweek planning tools**.

Concretely, it is:
- A **Streamlit single-page app** (`app.py`, 1,641 lines) with **five tabs**.
- Backed by a **swappable prediction architecture**: a per-position ML ensemble is production today; a from-scratch component-based pipeline exists fully built and verified, but is held behind a config flag because it backtested worse on the metrics that matter most (§6–7).
- Reading **live data** from the official (unofficial-to-use) FPL REST API on every page load, plus a **second live data source** now for the Mini-League tab (league standings and every entry's picks).
- Supplemented by a **historical data pipeline** (vaastav's community GitHub archive) that now also incorporates **Understat** (a fourth data source) and a much richer, multi-horizon feature set.
- A **solo, personal project** by Subanta Poudel — university coursework origins, since fully rebranded.

## 2. What this project is *not*

Carried over from the previous audit, with two items now **resolved** and flagged as such:

- **Not primarily a linear-regression product any more.** The 7-feature `LinearRegression` is kept on disk (`Models/linear_reg_v1.pkl`) purely so the backtest harness can compare old against new — the live app now serves a per-position `XGBoost + RandomForest` ensemble (§6).
- **Not a price/transfer-market predictor.** Unchanged — no forecast of future price rises/falls.
- **Not a multi-gameweek squad optimizer itself, and now explicitly delegates that.** "Rate My Team" is still single-gameweek and greedy. But the app now **exports** a multi-gameweek xP CSV (§9, Sidebar) formatted for `solioanalytics/open-fpl-solver` (formerly `sertalpbilal/FPL-Optimization-Tools` — confirmed via GitHub's redirect to be the same repository, renamed, not a fork) — multi-gameweek planning, free-transfer valuation, chip strategy, and point-hit decisions are explicitly left to that external, purpose-built MILP solver rather than rebuilt here.
- **~~Not using third-party expected-goals data~~ — now partially resolved.** Understat is integrated as a real 4th data source (player xG/xA/xGChain/xGBuildup/key passes/shots; team xG/xGA/PPDA/Deep) for historical training and backtesting. Live serving still uses FPL's own native `expected_goals_per_90`/`expected_assists_per_90` fields instead of live-scraping Understat — a deliberate choice (§5D) for reliability, not an oversight.
- **Not accounting for non-Premier-League competitions.** Unchanged.
- **~~Not authenticated or multi-user~~ — now partially resolved.** Still no accounts/login, but the app is no longer single-user-only in effect: the new **Mini-League tab** (§9) lets a whole group see standings, projected points, captaincy, and effective ownership from one shared, bookmarkable URL (`?league_id=...`), alongside the existing `?team_id=...` for an individual.
- **Not commercially licensed.** Unchanged.
- **Not running on infrastructure that guarantees uptime or persistence.** Unchanged — Streamlit Community Cloud free tier, ephemeral filesystem. This constraint was actively checked against every new feature this round (§11).
- **Not using granular team attack/defence strength data** for the *legacy* rule-layer clean-sheet formula — still true for that one formula. The new per-position and component models instead learn from real historical team/opponent rolling goals-for/against, which doesn't have this gap (§6).

## 3. Identity history

Unchanged from the previous audit — see git history. Blaugrana palette, three-name history (Catalan FPL Predictor → ApexFPL → CatalanPlays), coursework files already stripped from the public repo.

## 4. Architecture

```
Live FPL API ──┐
               ├─→ src/data_loader.py (whitelist-trims payload) ─→ src/features.py (process_data)
Fixtures API ──┘                                                         │
League/Entry-Picks API ─→ src/data_loader.py (new: leagues-classic,      ▼
   entry/{id}/event/{gw}/picks, ttl=900, paginated, polite)   df with engineered features,
                                                          cs_probability, DEFCON, squad depth,
                                                          team-rolling goals (live), etc.
                                                                          │
                                                                          ▼
                                                  src/predictor.py::make_predictions
                                                  dispatches on src/config.py::USE_COMPONENT_MODEL
                                                       │                        │
                                       ┌───────────────┘                        └──────────────┐
                                       ▼                                                        ▼
                     _make_predictions_position_ensemble                    _make_predictions_component
                     Models/position_models_v1.pkl                          Models/component_models_v1.pkl
                     (XGBoost+RandomForest per position)                    (src/components.py: 8 scoring
                     + full hand-tuned rule layer                             events summed) + news-only
                     [ACTIVE — flag is False]                                 rule layer [BUILT, NOT ACTIVE]
                                       │                                                        │
                                       └───────────────┐                        ┌──────────────┘
                                                        ▼                        ▼
                                                     final_xp, reasoning, ceiling_score
                                                                          │
                                                                          ▼
                                                                       app.py
                                                      (5 tabs: Predictions, Mini-League,
                                                       Player History, Hall of Fame, Documentation)
                                                                          │
                                                                          ▼
                                                       Sidebar: multi-GW projections.csv export
                                                       (src/projections.py) for external MILP solvers

Separately: vaastav historical CSVs + Understat ─→ src/features.py::fetch_full_history /
            fetch_full_history_with_understat ─→ non-leaky walk-forward features
            ─→ Player History tab, evaluation/backtest.py, training/*.py
```

Still no database, no backend server beyond Streamlit, no message queue, no background jobs. The one new *external* dependency at request time is the Mini-League tab's calls to `leagues-classic/{id}/standings/` and `entry/{id}/event/{gw}/picks/` — both cached and paced (§9).

## 5. Data sources — now **four**, each with a different trust model

**A. Live FPL REST API** (`src/data_loader.py`) — unchanged endpoints plus new ones and new whitelisted fields:
- New: `leagues-classic/{id}/standings/` (paginated, `page_standings=N`), `entry/{id}/event/{gw}/picks/` (used both for the existing single-user Rate My Team and, now, for every entry in a mini-league).
- `ELEMENT_KEYS` grew substantially this round: `starts`, `starts_per_90`, `expected_goals_per_90`, `expected_assists_per_90`, `expected_goals_conceded_per_90`, `saves_per_90`, `clean_sheets_per_90`, `clearances_blocks_interceptions`, `tackles`, `recoveries`, `defensive_contribution`, `yellow_cards`, `red_cards` — all added to the whitelist *before* being wired into `features.py`, per this project's own hard-won rule (the whitelist has silently produced zeros on four separate past occasions when a field was used before being added here).
- `_assert_whitelist_subset()` and `tests/test_whitelist.py` (added this round) turn a fifth such mistake into an immediate, loud `KeyError`/test failure instead of a silent zero.
- Cached 1 hour for bootstrap-static, 60s for fixtures (unchanged); **900s (15 min) for the two new league endpoints** — explicitly requested, since that surface is rate-limited.

**B. Historical per-gameweek CSVs** (`src/features.py::fetch_full_history`), vaastav's archive, now covering **2021-22 through 2025-26** (one season further back than before). Substantially expanded this round:
- Multi-horizon rolling features (windows 1/3/5/10/38, not just 3) for the original 6 stats plus 9 new "component" stats (`expected_goals`, `expected_assists`, `expected_goals_conceded`, `saves`, `bps`, `bonus`, `yellow_cards`, `red_cards`, `starts`).
- Team **and opponent** rolling goals-for/against + a computed league rank — genuine opponent-strength signal beyond FPL's 1–5 FDR integer.
- A **walk-forward Beta-Binomial shrinkage estimator** for DEFCON hit-rate (`defcon_hit_prob`) — a real statistical model (prior blended with each player's own cumulative hit/miss record), not a plain rolling mean, built specifically because DEFCON has no history before 2025-26 and a naive average would either have nothing to average (pre-2025-26) or overreact to 2-3 games of history early in 2025-26.
- The multi-season CSV fetch is now **concurrent and timeout-guarded** (`ThreadPoolExecutor` + `requests.get(timeout=20)` instead of bare `pd.read_csv(url)`) — a real reliability fix found mid-session: an untimed fetch could hang indefinitely on a stalled connection, and a serial fetch let one slow season block every season after it.
- `squad_depth_count` (same-position, same-team teammates who are themselves regular starters) computed identically here and live, for the minutes model.

**C. Understat** (`src/understat.py`, new file, ~510 lines) — the fourth source:
- Team-level (xG, xGA, PPDA, PPDA-allowed, Deep, Deep-allowed) scraped from `understat.com/getLeagueData/EPL/{year}` — the real internal JSON endpoint (found by inspecting network requests; the commonly-documented "`JSON.parse()` embedded in HTML" scraping method no longer works).
- Player-level (xG, xA, xGChain, xGBuildup, key passes, shots) reused from vaastav's own bundled `data/{season}/understat/*.csv` files rather than scraped separately.
- Matching reuses the same name-matching cascade as the pre-season baseline (§5D below), extended with a **6th "mononym" tier** for single-name players (Alisson, Casemiro, etc.), gated on uniqueness to avoid false positives.
- **Verified match rates:** 100% team, 93.6% player (2024-25) — cleared the 90% bar this was built against. Remaining unmatched are genuinely hard cases (name truncations, nicknames, encoding artifacts in the source data) that were deliberately not chased further.
- **Fails soft everywhere**: any Understat outage returns the app to FPL-only data with a logged warning, never a crash.
- `fetch_full_history_with_understat()` is a deliberately **separate** function from `fetch_full_history()` — the live app's own feature computation never gains Understat's network dependency; only the offline training/backtest scripts call it.
- **Not used for live serving** of the attacking component — see §5D.

**D. Live-vs-historical feature parity, and the Understat substitution.** A recurring design constraint this round: several new features only have a genuine walk-forward historical form (from a CSV), not an equivalent computed live from the FPL API alone, without expensive per-player calls. The resolution, applied consistently:
- Where FPL's own live API now exposes a season-aggregate equivalent (e.g. `expected_goals_per_90` mirrors historical rolling `expected_goals`), that's used live, with the historical rolling value used for training — same semantics, different literal source, documented at each call site (`src/components.py::derive_component_inputs`).
- Where no live equivalent exists at all (true match-by-match DEFCON hit/miss history), a different, principled live-only estimate is used instead (a Poisson threshold-crossing probability from the season rate) rather than reusing the historical walk-forward machinery unchanged.
- This is also *why* Understat isn't used for live attacking predictions: it would require live-matching every player every page load, which failed reliability spot-checks earlier in the project's history (rate limiting, multi-minute fetch times) — FPL's own native xG/xA fields cover the same underlying signal without that cost.
- Local committed CSVs (`Data/Cleaned/model_ready_data.csv`, pre-season baseline) — unchanged from the prior audit.

## 6. The prediction models — **two, one active, one built and held back**

### 6a. Production: per-position ensemble (`training/train_position_models.py`, `src/ensemble.py`)

- One `MedianEnsembleRegressor` (XGBoost + RandomForest, median-blended) **per position** (GKP/DEF/MID/FWD) — replaces the single all-positions `LinearRegression`, because a goalkeeper's points come overwhelmingly from saves/clean sheets and a forward's from xG; one linear model averaged across positions mostly just learned "did this player start" (minutes alone recovered ~91% of its R², per the prior audit).
- **OpenFPL-style balanced sample weighting**: `KBinsDiscretizer` bins the target (2 bins for GK, 3 for DEF/FWD, 4 for MID) then `compute_sample_weight("balanced")`, clipped at the 95th percentile, rescaled to unit mean — specifically to stop the model collapsing toward the mean and never predicting a hauler.
- **Team-grouped cross-validation** (`GroupKFold` on club, not a random split) during hyperparameter search, specifically to prevent a team's in-season form from leaking across train/validation folds.
- Trained on 2021-22 through 2024-25, evaluated on 2025-26 (held out, never touched during training/tuning).
- **Backtested result** (`evaluation/results/2025-26_position_models_v1.json`, vs. the legacy LinearRegression): Haulers RMSE 6.034→5.583 (**−7.5%**), Tickers RMSE 1.579→1.316 (**−16.7%**) — the two metrics this was built to improve — at the cost of Zeros RMSE 0.746→1.002 and Blanks RMSE 1.303→1.680 (both worse). This is reported as the **expected, correct trade-off**, not chased back down.
- Two real training-pipeline bugs were found and fixed while building this: `n_jobs=-1` set on both the outer hyperparameter search and the inner estimator caused catastrophic CPU oversubscription (a 32-core search failed to finish GKP's tuning in 30+ minutes); and `RandomForestRegressor`'s `max_depth=None` grew near-one-leaf-per-row trees on the larger positions, the single biggest remaining slowdown once the first fix landed.

### 6b. Built, verified, and deliberately **not** active: the component pipeline (`src/components.py`, `training/train_component_models.py`)

A full restructuring away from "one regression predicts total points" toward predicting **each real scoring event separately and summing through the actual 2026/27 scoring matrix**:

| Component | Method |
|---|---|
| Minutes (P(plays), P(60+)) | Two `LogisticRegression`s on rolling minutes, squad depth, starts rate, position |
| Attacking (xG, xA) | FPL-native `expected_goals`/`expected_assists`, linearly calibrated against real goals/assists, scaled by predicted minutes |
| Clean sheet | `LogisticRegression` on team/opponent rolling goals-for/against (replaces the old hand-tuned `0.30 + delta×0.08` formula) |
| Goals conceded (new — the old model never had this) | Calibrated `expected_goals_conceded` rate × minutes fraction, −1 per 2 conceded, DEF/GKP only |
| Saves | GK-only calibrated rate ÷ 3 |
| DEFCON | `P(hits real threshold) × 2.0` from the walk-forward Beta-Binomial estimator (§5B) historically, a Poisson threshold-crossing estimate live |
| Bonus | Regression on BPS rate **and** DEFCON hit-probability together (not assumed independent — 2026/27 recalibrated BPS specifically to reduce bonus for heavy-DEFCON matches) |
| Cards | Small negative from each player's own rolling card-point rate |

- Produces the exact reasoning-string format requested: `"5.6 = 2.0 appearance + 1.4 attack (0.31 xG, 0.19 xA) + 1.0 CS (38%) + 1.2 DEFCON (60% × 2)"` — every number in it is something a model actually computed, not a rule-layer constant.
- The rule layer for this path is cut to **news overrides only**: injured/suspended/unavailable wipe to zero, plus a manual-override flag. The old fixture-difficulty bonus, premium multiplier, star-captain bonus, and rotation tax are deleted for this path, on the reasoning that the component models already learn that context from data.
- **Multiple real bugs were found and fixed during this build** before any number could be trusted, notably: the attack calibration was initially regressing `expected_goals` against itself (not real goals scored) because the real-outcome columns were never added to the fetch; a per-90 rate conversion blew up to nonsensical values (max 54 "xG per 90") for players with a handful of recent minutes, fixed with a minimum-sample floor and a sane cap; and — the most consequential — **clean-sheet, DEFCON, and bonus points were originally credited to every owning player without discounting for their own chance of actually playing**, so a permanent benchwarmer at a strong defensive team was credited close to a starter's clean-sheet value; fixed by multiplying each of those components by the same minutes-fraction the attack/save components already used.
- **Backtested result** (`evaluation/results/2025-26_component_models_v1.json`, vs. the *production* per-position ensemble, i.e. §6a — not the legacy LinearRegression): better on Zeros (1.002→0.740) and Blanks (1.680→1.386) and overall RMSE/MAE, but **worse on both metrics that were set as the bar**: Tickers RMSE 1.316→1.565 (+18.8%), Haulers RMSE 5.583→5.996 (+7.4%), plus a real drop in Spearman rank correlation (0.693→0.672) and top-10 overlap (1.2→1.0/GW).
- **Consequently held behind `src/config.py::USE_COMPONENT_MODEL = False`.** Working hypothesis for the regression, stated in that file's own comment: the component calibrations are plain linear regressions, which regress toward the mean on a heavy-tailed target — the opposite of what §6a's balanced sample weighting was built to achieve for exactly the Haulers/Tickers buckets. Flipping this flag back on is a one-line change once that's addressed and re-verified — not on the strength of the Zeros/Blanks improvement alone.
- `src/predictor.py::make_predictions()` dispatches between the two paths on this single flag; `app.py` never has to know which one is live.

## 7. The rule layer — now genuinely two different rule layers

- **Position-ensemble path (active)**: essentially the full hand-tuned rule layer from the previous audit — fixture-difficulty bonus, premium multiplier, star-captain bonus, dynamic clean-sheet odds, DEFCON stepped bonus, squad-depth rotation tax, starter-probability and partial-injury multipliers. Unchanged in this round *except* it now sits behind the dispatcher rather than being the only path.
- **Component path (built, inactive)**: cut down to hard availability wipe (injured/suspended/unavailable/manual override) only, described in §6b.

## 8. The captaincy engine

Unchanged mechanism from the previous audit (eligibility filter, separate Ceiling Score ranking, deliberately not a hardcoded name list) — both prediction paths feed the same downstream `ceiling_score`/`ceiling_stars` logic in `src/predictor.py`, so this doesn't fork with the model dispatch.

## 9. Features, tab by tab — now **five** tabs

**Tab 1 — 🔮 Predictions**: Season Points Explorer (replaced the old "Last Gameweek Recap," which compared an unshown internal model output against actuals and was judged not meaningful — this is a live, fully-recomputable, redeploy-safe search-and-drill-down over the current season instead), AI Team of the Week, Top Captain Picks, Predicted Points table, 5-GW FDR Ticker, Rate My Team — all as in the previous audit, now running on whichever model `src/config.py` selects.

**Tab 2 — 🏟️ Mini-League** *(new this round)*:
- League ID input, persisted to `?league_id=...` (mirrors `?team_id=...`'s existing pattern) so a league dashboard is bookmarkable/shareable.
- **Standings** for every entry retrieved (paginated, capped at 500/10 pages with a visible truncation note for anything larger).
- **Per-manager projected points and captain** for the upcoming gameweek, computed by fetching each entry's actual picks and running the app's own predictions against them — capped at the top 60 entries by rank for squad-level detail (with a visible note), to keep load times reasonable against a rate-limited endpoint.
- **Effective ownership**: each owning manager weighted by their real multiplier that gameweek (captain ×2, triple-captain ×3, benched ×0) — explicitly the "useful number" requested, since it can exceed 100% (verified live: Haaland at 93.3% raw ownership showed 190% effective ownership in a real league, because nearly everyone who owns him also captains him) and directly answers "which of my players move me up or down *this* league."
- **Differentials**: the full predicted player pool (not just players someone in the league owns — a 0%-owned gem is exactly the case that matters and would otherwise disappear), filtered to under 30% league ownership, sorted by projected xP.
- A private or nonexistent league returns a clear warning, not a stack trace — verified live against both a real public league and a deliberately invalid ID.
- One real interoperability bug found and fixed while verifying this: initial gameweek-numbering used this app's own "current gameweek" convention (`is_current`-first), which disagrees by one with the real solver's convention (see §9, Sidebar) during a live gameweek — not relevant to this tab specifically, but the same underlying "which gameweek is 'now'" question came up twice this round (see Sidebar entry below).

**Tab 3 — 📊 Player History**, **Tab 4 — 🏆 Hall of Fame**, **Tab 5 — ℹ️ Documentation**: unchanged from the previous audit.

**Sidebar — 📤 Multi-GW Projections Export** *(new this round)*:
- Horizon input (default 5 gameweeks), a "Generate projections" button, then a CSV download button.
- Runs the app's own prediction pipeline once per gameweek in the horizon (fixture/opponent context is genuinely gameweek-specific; form features are frozen at today's values, stated explicitly in the UI caption — "the further out a gameweek is, the more its number is really just current form against that fixture").
- Blank gameweeks (a team with no fixture) are reported as 0, not a fixture-blind guess. Double gameweeks are scored as a single fixture — a stated, not silently absorbed, limitation.
- **Column format was fetched and verified against the real solver's own parser code**, not guessed from its README (which documents solver *settings*, not the CSV layout): `ID`/`Name`/`Team`/`Pos`(single-letter)/`Value` + one `{gw}_Pts`/`{gw}_xMins` pair per gameweek, `gw` being the real absolute FPL gameweek number.
- **Verification went further than reading code**: the actual `solioanalytics/open-fpl-solver` repo was cloned, its real dependencies installed, and its real `prep_data()` **and** `solve_multi_period_fpl()` (the actual MILP solve) were run end-to-end against a CSV this app generated live — with zero manual edits to the file. This caught a real bug (the gameweek-numbering convention mismatch mentioned above) before it shipped: the app's own `is_current`-first convention put the export one gameweek behind what the solver's `is_next`-only convention expected during a live gameweek, producing an actual `ValueError` on the first attempt. Fixed to match the solver's convention specifically for this export.

## 10. Branding & design system

Unchanged from the previous audit.

## 11. Security & deployment posture

- Whitelist-assertion safety net (§5A) and `tests/test_whitelist.py` are new hardening since the previous audit, directly targeting a bug class that had already recurred four times.
- **Deployment-constraint discipline held throughout this round's much larger feature set**: every new cache (Understat, league standings, entry picks) uses `st.cache_data` with an explicit TTL, never a local file write-and-read-back; the historical CSV fetch was made concurrent/timeout-guarded rather than persisted; the two new trained model files (`Models/position_models_v1.pkl`, `Models/component_models_v1.pkl`) are loaded via `joblib` from the repo, same pattern as the pre-existing `linear_reg_v1.pkl`, not regenerated or cached to local disk at runtime.
- No new authentication surface. The Mini-League tab reads only the same public, unauthenticated FPL endpoints as everything else; a private league correctly fails closed (§9) rather than exposing anything.
- `.gitignore` gained a `*.csv` rule and `.env`/`.env.*` this round (existing committed CSVs stay tracked; this only stops *new* untracked CSVs from being added by accident).

## 12. Bugs found and fixed this round (in addition to the 9 already documented from the previous audit)

1. **Attack calibration self-regression** — `expected_goals` was being regressed against itself instead of real `goals_scored`/`assists`, because the real-outcome columns were never pulled from the historical CSV. Fixed by adding them and recalibrating.
2. **Per-90 rate blow-up** — dividing a real stat total by a near-zero recent-minutes sample extrapolated to nonsensical per-90 rates (max observed: 54). Fixed with a minimum-sample floor and a physically-sane cap.
3. **Goals-conceded calibration corrupted by benched-player rows** — a benched player's zero input was paired against their team's real (nonzero) result, dragging the fit toward "just predict the league average." Fixed by filtering the fit to rows where the player actually played.
4. **Clean-sheet/DEFCON/bonus over-crediting bench players** — the single most consequential bug of this round (§6b): none of these three components were scaled by the player's own chance of playing, so a permanent benchwarmer at a strong defensive team scored close to a nailed starter's clean-sheet value. Found via a direct GKP-position sanity check (predicted mean nearly double the actual mean) before it reached the backtest report.
5. **Hyperparameter-search CPU oversubscription** — `n_jobs=-1` on both the outer search and inner estimator turned a tunable search into one that couldn't finish in 30+ minutes on a 32-core machine. Fixed by setting the inner estimator to `n_jobs=1`.
6. **`RandomForestRegressor(max_depth=None)`** growing nearly one leaf per training row on the larger positions — the dominant remaining slowdown after fix #5, and a textbook overfit regardless. Fixed by capping depth.
7. **Untimed, serial multi-season CSV fetch** — `pd.read_csv(url)` has no timeout and can hang indefinitely on a stalled connection; fetching seasons one at a time let one slow season block every season after it. Fixed with `requests.get(timeout=...)` + concurrent fetching.
8. **Solver export gameweek-numbering mismatch** — this app's `is_current`-first "what's the current gameweek" convention disagreed by one with the target solver's `is_next`-only convention during a live gameweek, producing a real `ValueError` verified against the actual solver code. Fixed to match the solver's own convention for that specific export.
9. Two false-positive-masking bugs in the (separate, earlier-in-session) Understat integration: an empty fetch reporting a match rate of `1.0` (perfect) instead of `0.0` (nothing to report), and a duplicate-column `groupby` crash from not dropping the original `team` column before a rename.

## 13. Known limitations, stated honestly

Carried over from the previous audit where still true, plus new ones from this round:

- DEFCON can still let budget defenders outrank premium attackers in the **active** (position-ensemble) rule layer's main table — unchanged, not in scope this round.
- Clean-sheet odds in the **active** rule layer still run on the coarser always-populated team-strength field, not the granular one — unchanged.
- The pitch-visualizer's captain/vice badges and the Top Captain Picks card can still disagree — unchanged.
- Rate My Team's transfer suggestion is still single-gameweek and greedy — **but the app now offers an explicit off-ramp** to a real multi-gameweek solver (§9, Sidebar) rather than only disclosing the limitation.
- **The component pipeline (§6b) is fully built, fully verified, and not live** — it is worse on Haulers/Tickers RMSE than what's actually running. This is the single most important "don't assume" fact about the current state of prediction quality: the more architecturally interesting model is not the one serving predictions.
- **A double gameweek is scored as a single fixture** in both the live app's existing fixture-aware logic and the new solver-export CSV — a team playing twice in one FPL gameweek under-counts that player's true expected total for that gameweek. Stated in the export's UI caption; not fixed this round.
- **Mini-League squad-level detail is capped at the top 60 entries by rank** (standings themselves go up to 500) — a genuinely huge mini-league won't get per-manager projections/ownership for everyone, by design, to keep load times reasonable against a rate-limited endpoint.
- The rule layer's constants in the **active** path are still hand-tuned, not fitted — unchanged; this is precisely the property the component pipeline (§6b) was built to remove, and precisely why its current backtest regression matters.

## 14. Tech stack

Adds to the previous list: **XGBoost**, **scipy** (Beta-Binomial/Poisson estimators, Spearman rank correlation), **GitHub CLI** (used this session for PR creation, not an app dependency). Everything else — Streamlit, pandas, numpy, scikit-learn, joblib, plotly, pyarrow, requests — unchanged.

## 15. Repository layout (current)

```
app.py                          1,641 lines — 5-tab UI + orchestration + sidebar export
src/
  data_loader.py                  418 lines — API fetching (incl. league/picks), whitelist, tz utils
  features.py                     909 lines — process_data (live), fetch_full_history(_with_understat)
  predictor.py                    402 lines — dispatcher + both rule layers + captaincy scoring
  components.py                   441 lines — the 8-component prediction pipeline (inactive)
  ensemble.py                      32 lines — MedianEnsembleRegressor
  config.py                        32 lines — USE_COMPONENT_MODEL flag
  projections.py                  153 lines — multi-GW solver CSV export
  understat.py                    510 lines — 4th data source
training/
  train_position_models.py        277 lines — fits the active per-position ensemble
  train_component_models.py       247 lines — fits the inactive component pipeline
evaluation/
  backtest.py                     637 lines — walk-forward, outcome-bucketed, model-vs-model comparisons
  results/*.json                          — recorded backtest runs, one per model/feature-set
Models/
  linear_reg_v1.pkl                       — legacy, kept only for backtest comparison
  position_models_v1.pkl                  — ACTIVE production model
  component_models_v1.pkl                 — built, verified, not served (see §6b)
tests/
  test_whitelist.py, conftest.py          — live-API whitelist safety net
Data/Raw/, Data/Cleaned/, Notebooks/, metrics.json, .streamlit/config.toml — unchanged from prior audit
```

## 16. Current state

On branch `feature/multi-model-group-tools`, commit `34e85e0`, with **PR #1 open against `main`** (ready for review, not a draft). Working tree is clean — everything described in this document is committed and pushed. The live deployment at `apexfpl.streamlit.app` has not yet been updated to this branch/PR.

## 17. Honest bottom line

The scope of this project roughly quadrupled in one extended session without a corresponding drop in rigor: every new model was backtested (not just built), every new external-format claim was verified against the real target's own code (not its README), and the one model that looked architecturally more honest — predict-each-scoring-event-and-sum, with genuinely computed reasoning strings — was still held back because it measured worse on the metrics that actually matter for this app's purpose. That's the single fact most worth carrying forward: **the more interesting model isn't the one running.** Everything else — Understat, the group/league features, the solver handoff — is additive and shipped. Still not a production-grade commercial product: no auth beyond a shareable URL, no persistence, no SLA, unofficial data sources, single maintainer.
