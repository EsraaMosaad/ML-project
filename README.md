# Greenhouse Gas (GHG) Emissions Predictor

## Overview
This application leverages Machine Learning to predict **Total (Location-Based) GHG Emissions** (Metric Tons CO2e) for buildings in Ontario. Based on the 2024 Broader Public Sector (BPS) energy report data, the model helps stakeholders estimate the environmental impact of facilities using physical and energy consumption attributes.

## Features
- **Batch Processing:** Upload a CSV file with multiple building entries and download the predictions.
- **Single Building Entry:** Interactive form to predict emissions for a specific building with intuitive dropdowns and checkboxes.
- **Impact Insights:** Real-time feedback on whether a building's emissions are **Low**, **Moderate**, or **High** impact based on sector benchmarks.
- **Visual Analytics:** Interactive dashboard showing emissions distributions, sector-wise comparisons, and correlation analysis.

## Technology Stack
- **Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Science:** `pandas`, `numpy`, `scikit-learn`
- **Modeling:** Ensemble Gradient Boosting (XGBoost, HistGradientBoosting) and Random Forest.
- **Visualization:** `matplotlib`, `seaborn`

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML-project
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Data & Model
The model is trained on the clean `bps_2024_model_ready2.csv` dataset. Key features include:
- **Property GFA:** Self-reported floor area.
- **Energy Mix:** Electricity and Natural Gas usage indicators.
- **Sector/Subsector:** Building categorization (e.g., Municipal, Acute Hospital).

The current **VotingEnsemble** model achieves a high **R² score (~0.89)**, ensuring reliable estimates for environmental reporting and planning.

## Project Structure
- `app.py`: Main Streamlit application.
- `notebooks/`: Data preparation and modeling research.
- `outputs/models/`: Trained model pipeline and schema definitions.
- `sample_features.csv`: Example input for batch prediction.
