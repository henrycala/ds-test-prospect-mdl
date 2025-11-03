from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import joblib
from sklearn.utils.class_weight import compute_sample_weight

from src.models import LogisticRegressionWrapper, XGBoostWrapper

logging.basicConfig(
    level=logging.INFO,               
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


TARGET_COLUMN = "is_won"
EXCLUDED_COLUMNS = ("id", "when_timestamp", "closedate", "employee_range")


class ExperimentAllFeatures:

    def start(self):
        logger.info("Starting experiment with all features.")
        self.load_train_data()


    def load_train_data(self):
        logger.info("Loading training data.")
        df_train = pd.read_parquet(Path("data/featured/train_data.parquet"))
        logger.info("Training data loaded with shape %s", df_train.shape)

        feature_columns = [col for col in df_train.columns if col not in EXCLUDED_COLUMNS + (TARGET_COLUMN,)]

        self.df_features = df_train[feature_columns].fillna(0)
        self.target = df_train[TARGET_COLUMN]

        # Compute sample weights to handle class imbalance
        self.weights = compute_sample_weight(class_weight="balanced", y=self.target)

        self.run_xgboost_training()


    def run_xgboost_training(self):
        logger.info("Starting XGBoost training.")
        xg_mdl = XGBoostWrapper()
        xg_mdl.train(
            X=self.df_features,
            y=self.target,
            fine_tune_params=True,
            sample_weight=self.weights,
        )
        self.xg_mdl = xg_mdl

        self.save_xgboost_model()


    def save_xgboost_model(self):
        logger.info("Saving trained XGBoost model.")
        root = Path("models")
        run_dir = root / f"xgboost_all_features_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xg_mdl.pipeline_, run_dir / "model.joblib")
        self.run_logistic_training()


    def run_logistic_training(self):
        logger.info("Starting Logistic Regression training.")
        mdl = LogisticRegressionWrapper()
        mdl.train(
            X=self.df_features,
            y=self.target,
            fine_tune_params=True,
            sample_weight=self.weights,
        )
        self.mdl_logistic = mdl

        self.save_logistic_model()


    def save_logistic_model(self):
        logger.info("Saving trained Logistic Regression model.")
        root = Path("models")
        run_dir = root / f"logistic_all_features_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.mdl_logistic.pipeline_, run_dir / "model.joblib")


if __name__ == "__main__":
    pipeline = ExperimentAllFeatures()
    pipeline.start()
