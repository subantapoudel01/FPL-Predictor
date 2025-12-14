# Fantasy Premier Leagure (FPL) Points Predictor


## Project Overview
This project uses **Artificial Intelligence (Multiple Linear Regression)** to predict the FPL points of players for the upcoming gameweek. Developed for the **Principles of AI** Module.

## Structure
- `data/`: Contains the historical FPL datasets (Sourced from Vaastav Anand).
- `notebooks/`: Jupyter Notebooks for Data Cleaning and Exploratory Data Analysis (EDA).
- `docs/`: Reports

## Dataset
Data sourced from the Vaastav Anand FPL Repository.
- **Training Data:** Seasons 2021-2025
- **Test Data:** Season 2025-26


## Methodology
- **Algorithm:** Linear Regression (Scikit-Learn)
- **Key Features:** ICT Index, Minutes Played, Fixture Difficulty
- **Validation:** Temporal Split (Time-Series Validation)


## How to Run
1. Clone this repository.
2. Open `notebooks/FPL_Analysis.ipynb`.
3. Run all cells to generate the "ICT Index vs Points" visualization.