# Air Quality ML — CO Concentration Estimation

Estimate **carbon monoxide (CO) concentration** from same-hour environmental and gas sensor measurements. This is a *contemporaneous estimation* problem; it is **not a forecast of future pollution levels**.

## Data
The included [AirQuality.csv](AirQuality.csv) contains 9,357 hourly records from March 2004 to April 2005. Source: [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) (also mirrored on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set)). Values are historical and geographically specific, not representative of current Saudi air quality.

Dates are parsed as day/month/year; decimal commas are converted; empty entries and `-200` are interpreted as missing. Rows without a CO target are excluded. Missing **features** are imputed using medians learned only from training folds.

## Reproducible evaluation (new)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
pytest -q
```

`train.py` sorts readings chronologically, reserves the **last 20%** of labeled observations as an untouched holdout, and compares a median dummy baseline, Ridge, Random Forest, and Histogram Gradient Boosting on training-only expanding-window `TimeSeriesSplit` cross-validation. Pipelines prevent preprocessing from learning the validation or test distribution. The model with lowest mean CV RMSE is refitted on all training data and evaluated once on the holdout.

Results are written locally to `artifacts/cv_comparison.csv`, `artifacts/metrics.json`, and `artifacts/best_pipeline.joblib`. Generated artifacts are ignored by Git. This repository does **not** claim new holdout scores until they have been reproduced by running the workflow.

**Caveats:** Time-aware evaluation can reveal weaker performance under distribution shift; chronological folds are not a substitute for independent data from other sensors/locations. Same-hour chemical measurements may not be available to a real-world inference system, so deployment requires a clearly specified sensor input contract. This dataset is not appropriate for estimating future values without a different problem formulation, lagged features and a forecast horizon.

## Legacy experiment
The [original script](airQuality_project.py) uses random train/test splitting, drops incomplete rows and compares Linear Regression with Random Forest. Its README originally reported 0.239/0.219 MSE and 0.88/0.89 R² respectively; these are **legacy random-split results** and are not directly comparable to the new time-aware experiment. The original visualizations remain in [results/](results/), including [actual versus predicted](results/actual_vs_predicted.png), [correlation matrix](results/correlation_matrix.png), and [feature importance](results/feature_importance.png).

## Possible next experiments
- Evaluate alternative feature subsets based on which sensors would be available at inference time.
- Measure monthly/seasonal drift and report results for multiple chronological holdout windows.
- Develop a **separate** next-hour forecasting task using lagged features and a strict prediction horizon.

## Stack
Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, pytest.
