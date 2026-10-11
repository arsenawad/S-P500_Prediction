# Hull Tactical — Compact ML Project v3

This project explores a compact, notebook-driven approach to forecasting S&P 500 market exposure using multiple machine learning models and a risk-aware allocation strategy. The goal is not just to predict returns, but to build a strategy that balances expected return and volatility constraints while producing a practical allocation signal.

## Project Goal

The notebook builds a tactical allocation framework for S&P 500 forecasting:

- Predict daily excess return of the market
- Convert predictions into a continuous allocation signal
- Evaluate models using return, volatility, Sharpe ratio, and risk-adjusted score
- Select the best-performing model or ensemble under a volatility cap
- Generate a final submission allocation for test data

The allocation interpretation is:

- 0 = no exposure
- 1 = normal 100% exposure
- 2 = 200% leveraged exposure

Values between 0 and 2 are also allowed, such as 0.4, 0.9, 1.3, or 1.8.

## Methodology

The project follows a compact but structured workflow:

1. Load and inspect the training and test datasets.
2. Prepare numeric features and handle missing values.
3. Add engineered features such as row statistics, group means, and date-cycle signals.
4. Split the training data chronologically into train/validation sets.
5. Test several candidate model families:
   - Ridge Regression
   - HistGradientBoostingRegressor
   - RandomForestRegressor
   - ExtraTreesRegressor
6. Tune hyperparameters based on validation performance.
7. Compare models using:
   - RMSE
   - overfit ratio
   - annualized strategy return
   - volatility ratio
   - Sharpe ratio
   - Model Score
8. Construct an optimized ensemble from the tuned models.
9. Retrain on the full training set and generate the final allocation submission.

## Why This Project Is Different

The project emphasizes the investment constraint rather than raw predictive accuracy alone. The notebook explicitly evaluates whether the model can produce a strategy that remains within a volatility threshold while improving risk-adjusted performance.

The validation objective is:

- maximize risk-adjusted performance
- respect the volatility constraint
- avoid selecting a model that looks accurate but performs poorly in a strategy context

## Repository Contents

- [hull_tactical_COMPACT_project_v3_tuned.ipynb](hull_tactical_COMPACT_project_v3_tuned.ipynb) — main analysis and modeling notebook
- [train.csv](train.csv) — training dataset
- [test.csv](test.csv) — test dataset
- [submission.csv](submission.csv) — final allocation predictions for submission
- [tuning_results.csv](tuning_results.csv) — parameter tuning outcomes
- [best_model_results.csv](best_model_results.csv) — best model comparison
- [ensemble_weights.csv](ensemble_weights.csv) — optimized ensemble weights
- [final_model_comparison.csv](final_model_comparison.csv) — final model and ensemble results

## Data

The dataset includes financial indicator features and a target column:

- `market_forward_excess_returns`

The target represents the forward daily excess return of the S&P 500 relative to a baseline.

## Modeling Notes

### Feature preprocessing

The workflow includes:

- filtering out numeric columns with excessive missingness
- adding missing-value indicator columns
- forward filling missing values
- filling remaining initial gaps with training medians
- adding engineered group and cyclical features

### Strategy conversion

Model predictions are converted into portfolio allocation using a nonlinear transformation:

- signal = tanh(prediction / std(prediction))
- allocation = clip(1 + 0.65 × signal, 0, 2)

This keeps allocations within a realistic range while allowing dynamic exposure decisions.

### Evaluation criteria

The notebook uses the following metrics as the primary decision framework:

- Strategy return
- Market return
- Strategy volatility
- Volatility ratio
- Sharpe ratio
- Model Score = Sharpe − penalty for volatility and underperformance

The key constraint is:

- volatility ratio ≤ 1.20

## Setup

This project is designed to run in a Python environment with common data-science libraries installed.

Typical dependencies include:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install the main requirements with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Run the Notebook

Open the notebook in Jupyter or VS Code and run the cells in order:

```text
hull_tactical_COMPACT_project_v3_tuned.ipynb
```

The notebook is structured in sections covering:

- data import and inspection
- data preparation
- feature engineering
- train/validation split
- tuning
- model comparison
- ensemble optimization
- final submission output

## Output Interpretation

The main outcomes of the notebook are:

- tuned model ranking by validation score
- optimized ensemble weights
- cumulative return comparison plots
- final submission allocation table for the test period

## Project Status

This is an experimental research and forecasting project focused on a compact, interpretable machine-learning pipeline for market allocation and risk-aware prediction.

## License

This project does not include a formal license file yet. If you plan to publish it publicly, consider adding a LICENSE file that matches your intended usage rights.

## Notes

This repository is intended for analysis and experimentation. Model performance can vary materially depending on dataset version, feature selection, and evaluation assumptions.
