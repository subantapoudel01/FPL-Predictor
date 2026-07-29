@echo off
echo ===================================================
echo Starting FPL Points Predictor AI Application...
echo ===================================================
echo.
echo [Step 1/2] Checking and installing required libraries...
python -m pip install -r requirements.txt
echo.
echo [Step 2/2] Launching Streamlit Dashboard in your browser...
echo.
python -m streamlit run app.py
pause