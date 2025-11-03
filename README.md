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
├── models/                         # Saved model artifacts.
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

1. **Pipelines (`src/pipelines/`):**
   - `data_preparation.py` merges customer/noncustomer entities, normalises categories, deduplicates timelines, and writes parquet outputs to `data/transformed/`.
   - `feature_engineering.py` builds rolling window aggregates, one-hot encodes industries, maps employee ranges, creates the `is_won` target, and emits `data/featured/featured_data.parquet`.
2. **Experiment orchestration (`src/experiments`)** 
    – `train_all_features.py` loads the featured train/holdout sets, computes class-balanced sample weights, hyperparameter-tunes each wrapper and train the models, saves artefacts to `models/<model>_<experiment>_<timestamp>/`.
   - `split_train_holdout.py` orders records chronologically and picks an 80 % quantile cut-off to produce `train_data.parquet` and `holdout_data.parquet`.
3. **Model wrappers (`src/models/`)** – `BaseModelWrapper` defines a consistent interface (`fine_tune_params`, `train`, `predict`, `feature_importance`). Implementations include:
   - `LogisticRegressionWrapper` – liblinear solver, class-weight search space, coefficient-based feature importance.
   - `XGBoostWrapper` – histogram tree booster with hyperparameter distributions for depth, learning rate, regularisation, and `scale_pos_weight`.
4. **Evaluation (`src/experiments/compute_metrics.py`)** – iterates over every saved pipeline, scores the holdout dataset, prints a full classification report, confusion matrix, and Recall@20 % for both classes using the model’s predicted probabilities.

## Executing the Workflow

Install dependencies (Python ≥ 3.10):

```bash
poetry install
```

Run the standard sequence:

```bash
# 1. Normalise raw CSVs -> data/transformed/
poetry run python -m src.pipelines.data_preparation

# 2. Engineer features + target -> data/featured/featured_data.parquet
poetry run python -m src.pipelines.feature_engineering

# 3. Split into train/holdout based on temporal cut-off
poetry run python -m src.experiments.split_train_holdout

# 4. Train logistic + XGBoost models; artefacts saved under models/
poetry run python -m src.experiments.train_all_features

# 5. Evaluate all saved models on the holdout dataset
poetry run python -m src.experiments.compute_metrics
```

## Development Notes

- Raw files need to be located in data/raw. The following folders must be created: data/featured, data/transformed.
- Notebooks in `src/notebooks/` should append `Path.cwd().parent.parent` to `sys.path` so imports like `from src.dataprep import load_raw_files` resolve cleanly.
- Saved pipelines already bundle feature preprocessing, so inference consists of loading the joblib artefact and calling `.predict()` / `.predict_proba()` with a DataFrame shaped like the training features.
- Extendable points:
  - Add new wrappers (e.g., LightGBM, CatBoost) by subclassing `BaseModelWrapper`.
  - Wire the scripts into an orchestrator (Airflow/Prefect) or CI workflows.
  - Enrich `config/` with environment-specific paths or tracking integrations (MLflow, Weights & Biases).

Keep this README updated when new pipelines, wrappers, or execution paths are introduced so the documentation remains the single source of truth for running the project.
