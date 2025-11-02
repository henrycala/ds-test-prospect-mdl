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
        self.load_normalized_data()


    def load_normalized_data(self):
        logger.info("Loading normalized data.")
        self.customers_df = pd.read_parquet("data/transformed/customers_normalized.parquet")
        self.usage_df = pd.read_parquet("data/transformed/usage_normalized.parquet")
        self.create_time_features()


    def create_time_features(self):
        from src.dataprep import generate_time_window_features

        logger.info("Creating features.")
        logger.info("Filling missing numerical values with 0.")
        num_columns = [col for col in self.usage_df.columns if col not in ["id", "when_timestamp"]]
        self.usage_df[num_columns] = self.usage_df[num_columns].fillna(0)

        logger.info("Generating rolling window features for 4 weeks.")
        self.usage_df = generate_time_window_features(
            df=self.usage_df,
            group_id="id",
            date_col="when_timestamp",
            numeric_cols=num_columns,
            window_size=4,
        )

        logger.info("Generating rolling window features for 12 weeks.")
        self.usage_df = generate_time_window_features(
            df=self.usage_df,
            group_id="id",
            date_col="when_timestamp",
            numeric_cols=num_columns,
            window_size=12,
        )

        logger.info("Generating rolling window features for 24 weeks.")
        self.usage_df = generate_time_window_features(
            df=self.usage_df,
            group_id="id",
            date_col="when_timestamp",
            numeric_cols=num_columns,
            window_size=24,
        )

        logger.info("Extracting time-based features from when_timestamp.")
        self.usage_df["year"] = self.usage_df["when_timestamp"].dt.year
        self.usage_df["month"] = self.usage_df["when_timestamp"].dt.month
        self.usage_df["day_of_week"] = self.usage_df["when_timestamp"].dt.dayofweek

        self.categorial_encoders()


    def categorial_encoders(self):
        logger.info("Encoding categorical features.")
        self.customers_df = pd.get_dummies(self.customers_df, columns=["industry"], prefix="industry")
        
        employee_size_mapping = {
            'unknown': 0,
            '1': 1,
            '2 to 5': 2,
            '6 to 10': 3,
            '11 to 25': 4,
            '26 to 50': 5,
            '51 to 200': 6,
            '201 to 1000': 7,
            '1001 to 10000': 8,
            '10001 or more': 9
        }

        self.customers_df["employee_size_encoded"] = self.customers_df["employee_range"].map(employee_size_mapping)
        self.merge_datasets()

    
    def merge_datasets(self):
        logger.info("Merging customers and usage datasets.")
        self.merged_df = pd.merge(
            self.usage_df,
            self.customers_df,
            on="id",
            how="left"
        )
        self.create_target()


    def create_target(self):
        from src.dataprep import mark_closest_when_date
        logger.info("Creating target variable.")
        self.merged_df = mark_closest_when_date(
            df=self.merged_df,
            group_col="id",
            when_col="when_timestamp",
            close_col="closedate",
            target_name = "is_won"
        )
        self.save_featured_data()

    
    def save_featured_data(self):
        logger.info("Saving featured data.")
        output_path = Path("data/featured")
        self.merged_df.to_parquet(output_path / "featured_data.parquet", index=False)
        logger.info("Feature engineering process completed.")


if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.start()