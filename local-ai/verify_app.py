"""Execute the Streamlit app with a synthetic API fixture and committed history."""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--api-unavailable", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import pandas as pd
    from streamlit.testing.v1 import AppTest
    import src.data_loader as loader

    if args.api_unavailable:
        with patch("src.data_loader.fetch_api_data", return_value=None):
            app = AppTest.from_file(str(repo / "app.py"), default_timeout=30).run()
        errors = [element.message for element in app.exception]
        if errors:
            print("FAIL API-unavailable recovery", errors)
            return 1
        if not app.error:
            print("FAIL expected a clear API-unavailable UI error")
            return 1
        print("PASS API-unavailable recovery")
        return 0

    if args.live:
        payload = loader.fetch_api_data()
        if payload is None:
            print("FAIL live API fetch")
            return 1
        input_label = "live API snapshot"
    else:
        fixture = repo / "local-ai" / "fixtures" / "api_sample.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        input_label = "labelled synthetic API fixture"

    history = pd.read_csv(repo / "Data/Cleaned/model_ready_data.csv")
    with patch("src.data_loader.fetch_api_data", return_value=payload):
        with patch("src.features.fetch_full_history", side_effect=lambda: history.copy()):
            app = AppTest.from_file(str(repo / "app.py"), default_timeout=90).run()
    errors = [element.message for element in app.exception]
    if errors:
        print("FAIL application execution", errors)
        return 1
    print("PASS main app execution")
    print("Tabs", [element.label for element in app.tabs])
    print("Dataframes", len(app.dataframe))
    print("Scope uses", input_label, "and committed history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
