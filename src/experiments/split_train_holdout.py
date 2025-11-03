from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
logging.basicConfig(
    level=logging.INFO,               
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineering:

    def start(self):
        logger.info("Starting feature engineering process.")
        self.load_features()

    
    def load_features(self):
        logger.info("Loading features data.")
        self.customers_df = pd.read_parquet("data/featured/featured_data.parquet")
        self.get_cutoff_date()


    def get_cutoff_date(self):
        logger.info("Calculating cutoff date for splitting data.")
        self.customers_df = self.customers_df.sort_values(by=['id', 'when_timestamp'])

        # Filter only won targets
        won_df = self.customers_df[self.customers_df["is_won"] == 1]

        # Compute the 80% quantile of when_timestamp
        quantile_80 = won_df["when_timestamp"].quantile(0.8)

        logger.info(f"Date when 80 percent of targets are captured: {quantile_80.strftime('%Y-%m-%d %H:%M:%S')}")
        self.cut_off_date = pd.to_datetime(quantile_80)
        self.split_data()

    def split_data(self):
        logger.info("Splitting data into training and holdout sets.")
        train_df = self.customers_df[self.customers_df["when_timestamp"] <= self.cut_off_date]
        holdout_df = self.customers_df[self.customers_df["when_timestamp"] > self.cut_off_date]

        # Save the datasets
        train_path = Path("data/featured/train_data.parquet")
        holdout_path = Path("data/featured/holdout_data.parquet")

        train_df.to_parquet(train_path, index=False)
        holdout_df.to_parquet(holdout_path, index=False)

        logger.info("Data split completed. Training data saved to %s and holdout data saved to %s", train_path, holdout_path)


if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.start()

