"""
test_rules.py — Quantitative Rule Layer Validation Script

Evaluates the mathematical impact of the Expert Rule Layer (src/predictor.py)
against the base Linear Regression model on historical FPL test data.

Calculates:
1. RMSE (Root Mean Squared Error) for raw_xp vs final_xp
2. R-squared (R²) for raw_xp vs final_xp
3. Average actual FPL points of top-20 predicted players (per-GW and overall)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Ensure workspace root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.predictor import load_model, make_predictions


def run_validation(season="2025-26", dataset_path=None):
    if dataset_path is None:
        dataset_path = project_root / "Data" / "Cleaned" / "model_ready_data.csv"

    print("=" * 70)
    print(" FPL PREDICTOR -- RULE LAYER QUANTITATIVE VALIDATION")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Target Season: {season}")

    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        return

    # 1. Load Dataset
    df = pd.read_csv(dataset_path)

    features = [
        "rolling_3_minutes",
        "rolling_3_ict_index",
        "rolling_3_creativity",
        "rolling_3_influence",
        "rolling_3_threat",
        "rolling_3_total_points",
        "value",
    ]

    df = df.dropna(subset=features + ["target_next_gw_points"]).copy()

    if season.lower() != "all":
        test_df = df[df["season"] == season].copy()
    else:
        test_df = df.copy()

    if test_df.empty:
        print(
            f"[ERROR] No data found for season '{season}'. Available seasons: {df['season'].unique().tolist()}"
        )
        return

    print(f"Test Samples: {len(test_df)} rows")

    # 2. Prepare Auxiliary Columns for Expert Rules
    test_df["team_name"] = test_df["team"]
    if "next_match_difficulty" not in test_df.columns:
        test_df["next_match_difficulty"] = 3
    if "chance_of_playing" not in test_df.columns:
        test_df["chance_of_playing"] = 100

    test_df["starter_prob"] = test_df["rolling_3_minutes"].apply(
        lambda m: 100 if m >= 60 else (75 if m >= 45 else (25 if m >= 30 else 5))
    )

    # 3. Load Model & Generate Predictions
    model = load_model()
    if model is None:
        print("[ERROR] Could not load model from Models/linear_reg_v1.pkl")
        return

    preds_df = make_predictions(test_df, model)
    actual = preds_df["target_next_gw_points"]

    # 4. Calculate Metrics for raw_xp (Base Model)
    raw_xp = preds_df["raw_xp"]
    rmse_raw = np.sqrt(mean_squared_error(actual, raw_xp))
    r2_raw = r2_score(actual, raw_xp)

    gw_top20_raw = preds_df.groupby("GW").apply(
        lambda g: g.nlargest(20, "raw_xp")["target_next_gw_points"].mean()
    ).mean()

    overall_top20_raw = preds_df.nlargest(20, "raw_xp")["target_next_gw_points"].mean()

    # 5. Calculate Metrics for final_xp (With Expert Rules)
    final_xp = preds_df["final_xp"]
    rmse_final = np.sqrt(mean_squared_error(actual, final_xp))
    r2_final = r2_score(actual, final_xp)

    gw_top20_final = preds_df.groupby("GW").apply(
        lambda g: g.nlargest(20, "final_xp")["target_next_gw_points"].mean()
    ).mean()

    overall_top20_final = preds_df.nlargest(20, "final_xp")["target_next_gw_points"].mean()

    # 6. Display Comparative Table
    print("\n" + "-" * 70)
    print(f"{'Metric':<35} | {'Raw Model (raw_xp)':<14} | {'With Rules (final_xp)':<14}")
    print("-" * 70)
    print(f"{'RMSE (Root Mean Squared Error) v':<35} | {rmse_raw:<14.4f} | {rmse_final:<14.4f}")
    print(f"{'R-squared (R2) ^':<35} | {r2_raw:<14.4f} | {r2_final:<14.4f}")
    print(f"{'Per-GW Top-20 Avg Actual Pts ^':<35} | {gw_top20_raw:<14.2f} | {gw_top20_final:<14.2f}")
    print(f"{'Overall Top-20 Avg Actual Pts ^':<35} | {overall_top20_raw:<14.2f} | {overall_top20_final:<14.2f}")
    print("-" * 70)

    # 7. Mathematical Verdict
    print("\n[SUMMARY] MATHEMATICAL VERDICT:")
    if rmse_final < rmse_raw:
        print("  [+] RMSE: Expert rules REDUCED overall prediction error.")
    else:
        print("  [-] RMSE: Expert rules INCREASED overall prediction error.")

    if r2_final > r2_raw:
        print("  [+] R2: Expert rules IMPROVED explained variance.")
    else:
        print("  [-] R2: Expert rules DECREASED explained variance.")

    if gw_top20_final > gw_top20_raw:
        print("  [+] Top-20 Picks: Expert rules IMPROVED top-20 pick points yield.")
    elif gw_top20_final == gw_top20_raw:
        print("  [=] Top-20 Picks: Expert rules yielded EQUAL top-20 pick points.")
    else:
        print("  [-] Top-20 Picks: Expert rules REDUCED top-20 pick points yield.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate FPL Predictor Expert Rules against Raw Model"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-26",
        help="Season to evaluate (e.g. 2025-26, 2024-25, or all)",
    )
    args = parser.parse_args()

    run_validation(season=args.season)
