# Prospect Conversion Modelling

This project provides a reproducible workflow for training, evaluating, and packaging propensity models that predict whether a HubSpot prospect converts. The codebase is organised into modular pipelines, experiment scripts, and model wrappers so you can iterate in notebooks and then promote the logic into maintainable Python modules.

## Repository Layout

```
.
├── config/                         # YAML configuration (data paths, defaults)
├── data/
│   ├── raw/                        # Provided CSV snapshots (input only)
│   ├── transformed/                # Output of the data-normalisation pipeline
│   └── featured/                   # Feature set + train/holdout parquet files
├── documents/                      # Assessment brief (ignored by git)
├── models/                         # Saved pipelines, metrics, params (gitignored)
├── src/
│   ├── dataprep/                   # Ingestion helpers, transforms, feature utils
│   ├── domain/                     # Dataclass schemas for raw/domain entities
│   ├── experiments/                # CLI entry points (train, split, metrics)
│   ├── models/                     # Trainable wrappers (base, logistic, xgboost)
│   ├── notebooks/                  # Exploratory notebooks
│   └── pipelines/                  # Standalone data-processing scripts
└── pyproject.toml                  # Tooling and dependency definitions
```

## Architecture & Data Flow

1. **Raw ingestion (`src/dataprep/ingest.py`)** – `RawDataSourcesConfig` resolves the CSV locations defined in `config/dataprep.yaml`, and dedicated loaders pull the customers, noncustomers, and usage datasets with basic type coercion.
2. **Domain alignment (`src/dataprep/transform.py`)** – functions convert raw frames into domain dataclasses, standardise categorical values, and fill temporal gaps so downstream joins are reliable.
3. **Pipelines (`src/pipelines/`):**
   - `data_preparation.py` merges customer/noncustomer entities, normalises categories, deduplicates timelines, and writes parquet outputs to `data/transformed/`.
   - `feature_engineering.py` builds rolling window aggregates, one-hot encodes industries, maps employee ranges, creates the `is_won` target, and emits `data/featured/featured_data.parquet`.
   - `split_train_holdout.py` orders records chronologically and picks an 80 % quantile cut-off to produce `train_data.parquet` and `holdout_data.parquet`.
4. **Model wrappers (`src/models/`)** – `BaseModelWrapper` defines a consistent interface (`fine_tune_params`, `train`, `predict`, `feature_importance`, `save_artifacts`). Implementations include:
   - `LogisticRegressionWrapper` – liblinear solver, class-weight search space, coefficient-based feature importance.
   - `XGBoostWrapper` – histogram tree booster with hyperparameter distributions for depth, learning rate, regularisation, and `scale_pos_weight`.
5. **Experiment orchestration (`src/experiments/train_all_features.py`)** – loads the featured train/holdout sets, builds a `ColumnTransformer` (numeric scaling + categorical encoding), computes class-balanced sample weights, hyperparameter-tunes each wrapper, trains, evaluates on the holdout split, and saves artefacts to `models/<wrapper>/<timestamp>/`.
6. **Evaluation (`src/experiments/compute_metrics.py`)** – iterates over every saved pipeline, scores the holdout dataset, prints a full classification report, confusion matrix, and Recall@20 % for both classes using the model’s predicted probabilities.

## Executing the Workflow

Install dependencies (Python ≥ 3.10). `pyarrow` and `xgboost` are optional but recommended:

```bash
pip install -r requirements.txt              # or poetry install
pip install pyarrow xgboost
```

Run the standard sequence:

```bash
# 1. Normalise raw CSVs -> data/transformed/
python -m src.pipelines.data_preparation

# 2. Engineer features + target -> data/featured/featured_data.parquet
python -m src.pipelines.feature_engineering

# 3. Split into train/holdout based on temporal cut-off
python -m src.experiments.split_train_holdout

# 4. Train logistic + XGBoost models; artefacts saved under models/
python -m src.experiments.train_all_features

# 5. Evaluate all saved models on the holdout dataset
python -m src.experiments.compute_metrics
```

Each training run produces:
- `model.joblib` – full preprocessing + estimator pipeline (reload with `joblib.load`).
- `metrics.json` – training/holdout metrics.
- `params.json` – best hyperparameters discovered during tuning.
- `feature_importance.csv` – sorted importance scores per wrapper.

## Handling Class Imbalance

- Training uses `compute_sample_weight(class_weight="balanced")` to weight the rarer positive class.
- Logistic regression explores `class_weight` during hyperparameter tuning.
- XGBoost exposes `scale_pos_weight` in its search grid; the experiment seeds weights with the observed ratio.
- `compute_metrics.py` reports Recall@20 % to understand performance when only the highest-scoring prospects are targeted; adjust thresholds accordingly when deploying.

## Development Notes

- Notebooks in `src/notebooks/` should append `Path.cwd().parent.parent` to `sys.path` so imports like `from src.dataprep import load_raw_files` resolve cleanly.
- Parquet operations require `pyarrow` or `fastparquet`; install one of them if you hit an import error.
- Saved pipelines already bundle feature preprocessing, so inference consists of loading the joblib artefact and calling `.predict()` / `.predict_proba()` with a DataFrame shaped like the training features.
- Extendable points:
  - Add new wrappers (e.g., LightGBM, CatBoost) by subclassing `BaseModelWrapper`.
  - Wire the scripts into an orchestrator (Airflow/Prefect) or CI workflows.
  - Enrich `config/` with environment-specific paths or tracking integrations (MLflow, Weights & Biases).

Keep this README updated when new pipelines, wrappers, or execution paths are introduced so the documentation remains the single source of truth for running the project.
