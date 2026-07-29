# Artificial Intelligence Coursework 2: FPL Points Predictor

**Module:** CU6051NP Artificial Intelligence
**Student Name:** Subanta Poudel
**London Met ID:** 20048736
**College ID:** NP04CP4S210115

---

## 🚀 How to Run the Application (For Tutors/Examiners)

This project has been pre-configured for instant execution. The machine learning model (`linear_reg_v1.pkl`) is pre-trained and saved inside the `Models/` directory to allow for immediate testing.

### Option 1: The 1-Click Method (Windows Recommended)
1. Double-click the **`run_app.bat`** file located in the root folder.
2. The script will automatically verify/install dependencies and open the interactive web dashboard in your default browser at `http://localhost:8501`.

### Option 2: Manual Terminal Execution (Mac/Linux/Windows)
If you prefer running via standard terminal commands, open your terminal inside the project root folder and execute:

**Step 1: Install required libraries**
`pip install -r requirements.txt`

**Step 2: Launch the Streamlit application**
`streamlit run app.py`

---

## 🧠 System Overview
This application functions as a Goal-Based Rational Agent utilizing a **Hybrid Intelligence Architecture**:
1. **Statistical AI Core:** A Multiple Linear Regression model trained on historical player records (Seasons 2021-2025) to predict base Expected Points based on Form, ICT Index, and Minutes.
2. **Expert Domain Rules:** A rule-based heuristic layer that adjusts predictions for tactical realities such as rotation ("Big Club Tax"), fixture difficulty ratings, and strict starter probabilities.