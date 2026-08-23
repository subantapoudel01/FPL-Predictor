# ⚽ Catalan FPL Points Predictor

**Project Established: October 2025** | **Current Engine: Catalan AI Predictor v1.2**

An open-source **Expected Points (xP) Engine** for Fantasy Premier League (FPL). Catalan combines live FPL API data ingestion, a trained Machine Learning model, and domain-expert rule heuristics to provide actionable player performance predictions for upcoming gameweeks.

---

## 💡 Overview

Predicting Fantasy Premier League points requires balancing empirical underlying stats with real-world managerial dynamics (rotation risk, injury status, fixture difficulty). Catalan addresses this with a **Hybrid Intelligence Architecture**:

1. **Statistical ML Baseline:** Predicts raw expected performance based on rolling underlying statistics (ICT Index, Influence, Creativity, Threat, Form, and Minutes).
2. **Domain Expert Heuristics:** Adjusts raw predictions for fixture difficulty ratings (FDR), position-specific clean sheet probabilities, premium asset captaincy potential, and rotation taxes.

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────┐
                                  │   Official FPL Live API   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌──────────────────────────┐      ┌───────────────────────────┐
│   Pre-trained ML Model   ├─────►│    Feature Normalization  │
│  (Models/linear_reg_v1)  │      │     (src/features.py)     │
└──────────────────────────┘      └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Hybrid Prediction Engine│
                                  │    (src/predictor.py)     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Streamlit Web Interface │
                                  │         (app.py)          │
                                  └───────────────────────────┘
```

### Key Components

- **Data Ingestion Layer (`src/data_loader.py`):** Ingests live player metadata, fixture schedules, and injury updates from the official FPL `bootstrap-static` and `fixtures` endpoints with caching and robust error handling.
- **Feature Engineering (`src/features.py`):** Normalizes player statistics across active gameweeks, handles pre-season edge cases safely, and constructs 3-game rolling metrics (`rolling_3_minutes`, `rolling_3_ict_index`, `rolling_3_creativity`, etc.).
- **Prediction Engine (`src/predictor.py`):** 
  - Loads serialized scikit-learn model via `pathlib`.
  - Calculates base expected points (`raw_xp`).
  - Applies expert rules:
    - **Appearance Floor:** Stepped starter probability curve based on average minutes.
    - **Fixture Difficulty:** Heuristic score adjustments (−1.0 to +2.5 pts) according to opponent FDR.
    - **Clean Sheet Boost:** Additional point weighting for DEF/GKP against weak opposition.
    - **Premium Captain Boost:** Bonus points for £10m+ premium assets facing favorable fixtures (FDR ≤ 3).
    - **Big Club Rotation Tax:** Scaled haircut for non-premium (<£9.0m) attackers playing for high-rotation squads.
    - **Availability Filter:** Multiplicative scaling based on official `chance_of_playing` indicators.
- **Dashboard (`app.py`):** Interactive Streamlit web interface featuring position filtering, player search, top-pick visual highlighting, and system status monitors.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/subantapoudel01/FPL-Predictor.git
   cd FPL-Predictor
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your browser at `http://localhost:8501`.

#### Windows 1-Click Launcher
On Windows, you can double-click `run__app.bat` in the root directory to automatically launch the application.

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit web application
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # FPL API client with caching
│   ├── features.py         # Feature engineering & pre-season handling
│   └── predictor.py        # ML model loader & expert rules logic
├── Models/
│   └── linear_reg_v1.pkl   # Serialized Linear Regression model
├── Notebooks/              # Data analysis & model training notebooks
├── requirements.txt        # Python package dependencies
├── run__app.bat            # Windows 1-click batch launcher
└── README.md
```

---

## ⚠️ Current Limitations

- **No Price Forecasting:** Uses current player values (`now_cost`) without modeling future market price fluctuations.
- **Domestic Focus:** Analyzes Premier League fixtures only; does not factor in mid-week European (Champions League/Europa League) or domestic cup fatigue and rotation.
- **Pre-Season Baseline:** Before Gameweek 1 commences, cumulative season statistics are zeroed out until live match data populates.
- **Baseline Model:** Currently utilizes a Multiple Linear Regression base model. The future roadmap includes transitioning to LightGBM/XGBoost and incorporating multi-season rolling lag features.

---

## 👤 Author & Timeline

Developed by **Subanta Poudel**.  
- **Project Established:** October 2025  
- **Current Engine Version:** Catalan AI Predictor v1.2  

Feedback, suggestions, and pull requests are welcome!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).