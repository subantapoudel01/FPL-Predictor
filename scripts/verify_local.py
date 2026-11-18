"""Run from the FPL-Predictor root with the macOS virtual-environment interpreter."""

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))

    for module in ["joblib", "numpy", "pandas", "plotly", "pyarrow", "requests", "scipy", "sklearn", "streamlit"]:
        importlib.import_module(module)

    required = [
        "app.py",
        "src/data_loader.py",
        "src/features.py",
        "src/predictor.py",
        "Models/position_models_v1.pkl",
        "Data/Cleaned/model_ready_data.csv",
    ]
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        raise RuntimeError("Missing " + ", ".join(missing))

    print("Python", platform.python_version(), sys.executable)
    for name in ["numpy", "pandas", "scikit-learn", "streamlit", "pyarrow", "tzdata"]:
        print(name, importlib.metadata.version(name))
    print("Timezone", ZoneInfo("Asia/Kathmandu"))

    from src.predictor import load_active_model

    model = load_active_model()
    if model is None:
        raise RuntimeError("The saved position ensemble could not be loaded.")
    if not isinstance(model, dict) or "positions" not in model:
        raise RuntimeError("The active model is not the position-ensemble bundle.")

    print("PASS source syntax and dependency imports")
    print("PASS active position ensemble", sorted(model["positions"]))
    print("PASS model features", model.get("features"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
