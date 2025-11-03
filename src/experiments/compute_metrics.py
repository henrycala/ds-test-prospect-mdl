from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
            probs = pipeline.predict_proba(df_pred)[:, 1]

            print(f"Recall at 20%: {recall_at_fraction(target, preds, 0.2)}")
            print(classification_report(target, preds, digits=3))

            # === 1. Probability Density Plot ===
            plt.figure(figsize=(8, 5))
            sns.kdeplot(probs[target == 0], label="Target = 0", fill=True)
            sns.kdeplot(probs[target == 1], label="Target = 1", fill=True)
            plt.title(f"Probability Density by Target - {run_name}")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{run_name}_prob_density.png")
            plt.close()

            # === 2. Lift Chart ===
            lift_df = pd.DataFrame({
                "target": target.values,
                "proba": probs
            }).sort_values("proba", ascending=False).reset_index(drop=True)

            lift_df["cum_positives"] = lift_df["target"].cumsum()
            lift_df["perc_samples"] = (lift_df.index + 1) / len(lift_df)
            lift_df["perc_positives"] = lift_df["cum_positives"] / lift_df["target"].sum()

            plt.figure(figsize=(8, 5))
            plt.plot(lift_df["perc_samples"], lift_df["perc_positives"], label="Model")
            plt.plot([0, 1], [0, 1], "--", color="gray", label="Random Model")
            plt.title(f"Lift Chart - {run_name}")
            plt.xlabel("Fraction of Samples (sorted by predicted probability)")
            plt.ylabel("Fraction of True Positives Captured")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{run_name}_lift_chart.png")
            plt.close()

            print(f"Plots saved")


if __name__ == "__main__":
    cm = ComputeMetrics()
    cm.start()


