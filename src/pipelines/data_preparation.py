from __future__ import annotations
from pathlib import Path
import yaml
import logging
import pandas as pd
logging.basicConfig(
    level=logging.INFO,               
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataNormalization:
    def __init__(self):
        config_path = Path("config/dataprep.yaml")
        with config_path.open("r", encoding="utf-8") as fp:
            self.config = yaml.safe_load(fp)


    def start(self):
        logger.info("Starting data normalization process.")
        self.read_raw_dataframes()


    def read_raw_dataframes(self):
        from src.dataprep import RawDataSourcesConfig, load_raw_files

        logger.info("Loading raw data files based on configuration.")
        config_obj = RawDataSourcesConfig(**self.config["data"])
        self.data_frames = load_raw_files(config_obj)
        self.normalize_dataframes()


    def normalize_dataframes(self):
        from src.dataprep import (transform_customer_to_domain, 
                                  transform_noncustomer_to_domain, 
                                  transform_usage_to_domain,
                                  normalize_categorical_column,
                                  fill_missing_dates)
        
        logger.info("Normalizing data frames.")
        customer_df = self.data_frames["customers"]
        noncustomer_df = self.data_frames["noncustomers"]
        usage_df = self.data_frames["usage"]

        # Normalize customer data
        customer_df = transform_customer_to_domain(customer_df)
        customer_df["is_customer"] = True
        noncustomer_df = transform_noncustomer_to_domain(noncustomer_df)
        noncustomer_df["is_customer"] = False
        usage_df = transform_usage_to_domain(usage_df)
        usage_df = usage_df.drop_duplicates(subset=["id", "when_timestamp"])
        usage_df = fill_missing_dates(usage_df, group_id="id", date_column="when_timestamp")
        self.usage_df = usage_df.reset_index(drop=True)

        # Merge customers and non-customers for unified processing
        logger.info("Combining customer and non-customer data for unified processing.")
        combined_df = pd.concat([customer_df, noncustomer_df], ignore_index=True)
        combined_df = combined_df.sort_values(by=["id", "closedate"])
        combined_df = combined_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Normalize categorical columns
        logger.info("Normalizing categorical columns: industry and employee_range.")
        combined_df = normalize_categorical_column(combined_df, "industry")
        combined_df = normalize_categorical_column(combined_df, "employee_range")
        self.customers_df = combined_df
        self.save_normalized_data()


    def save_normalized_data(self):
        output_dir = Path("data/transformed")

        customer_output_path = output_dir / "customers_normalized.parquet"
        usage_output_path = output_dir / "usage_normalized.parquet"

        logger.info(f"Saving normalized customer data to {customer_output_path}.")
        self.customers_df.to_parquet(customer_output_path, index=False)

        logger.info(f"Saving normalized usage data to {usage_output_path}.")
        self.usage_df.to_parquet(usage_output_path, index=False)

        logger.info("Data normalization process completed successfully.")
        self.end()

    def end(self):
        logger.info("Data normalization pipeline has finished execution.")


if __name__ == "__main__":
    pipeline = DataNormalization()
    pipeline.start()



        