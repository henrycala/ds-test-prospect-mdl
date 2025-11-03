from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, classification_report

logging.basicConfig(
    level=logging.INFO,               
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
TARGET_COLUMN = "is_won"
EXCLUDED_COLUMNS = ("id", "when_timestamp", "closedate", "employee_range")

def iter_saved_models(root: Path):
    for model_file in root.glob("*/**/model.joblib"):
        run_dir = model_file.parent
        yield run_dir.name, model_file, run_dir


def recall_at_fraction(y_true: pd.Series, y_score: np.ndarray, fraction: float) -> float:
    k = int(np.ceil(len(y_score) * fraction))
    top_idx = np.argsort(y_score)[::-1][:k]        # indices of top 20%
    y_pred = np.zeros_like(y_score, dtype=int)
    y_pred[top_idx] = 1
    return recall_score(y_true, y_pred)


class ComputeMetrics:

    def start(self):
        logger.info("Starting metrics computation process.")
        self.load_holdout_data()

    
    def load_holdout_data(self):
        logger.info("Loading holdout data.")
        self.holdout_df = pd.read_parquet("data/featured/holdout_data.parquet")
        self.compute_metrics()


    def compute_metrics(self):
        logger.info("Computing metrics on holdout data.")
        feature_columns = [col for col in self.holdout_df.columns if col not in EXCLUDED_COLUMNS + (TARGET_COLUMN,)]
        df_features = self.holdout_df[feature_columns].fillna(0)
        target = self.holdout_df[TARGET_COLUMN]

        for run_name, model_path, run_dir in iter_saved_models(Path("models")):
            df_pred = df_features.copy()
            print(f"\n=== {run_name} ===")
            pipeline = joblib.load(model_path)
            preds = pipeline.predict(df_pred)

            print(f"Recall at 20%: {recall_at_fraction(target, preds, 0.2)}")
            print(classification_report(target, preds, digits=3))


if __name__ == "__main__":
    cm = ComputeMetrics()
    cm.start()


