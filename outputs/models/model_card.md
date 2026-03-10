# Model Card – GHG Emissions Predictor

## Model
VotingEnsemble (RandomForest + HistGradientBoosting + XGBoost)

## Performance on Test Set
| Metric | Value |
|--------|-------|
| MAE    | 60.5178 |
| RMSE   | 231.4315 |
| R²     | 0.889257 |

## Training Data
- Dataset: BPS 2024 (buildings energy & GHG)
- Target: Total (Location-Based) GHG Emissions (Metric Tons CO2e)
- Train / Test split: 80 / 20 (stratified quantile bins)

## Features
{
  "Sector": "object",
  "Subsector": "object",
  "Primary Property Type - Self Selected": "object",
  "Largest Property Use Type": "object",
  "Property GFA - Self-Reported (m\u00b2)": "float64",
  "Number of Buildings": "int64",
  "Has_NaturalGas": "int64",
  "Has_Electricity": "int64"
}

## Notes
- Sub-estimators: RandomForest (100 trees), HistGradientBoosting (300 iters), XGBoost (300 rounds)
- Preprocessing inside each sub-pipeline: OHE for categoricals, median imputation for numerics
