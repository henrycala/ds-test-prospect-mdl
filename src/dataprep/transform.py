from __future__ import annotations
import pandas as pd
import re
import unicodedata
from dataclasses import fields
from src.domain import (
    CustomerMdl,
    NonCustomerMdl,
    UsageActionMdl,
)


def transform_customer_to_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe to domain-specific format.

    This is a placeholder function. Actual transformation logic should be implemented here.

    Args:
        df (pd.DataFrame): Raw dataframe.
    """
    schema_columns = [f.name for f in fields(CustomerMdl)]
    # Rename columns to lowercase for consistency
    df_transformed = df.rename(columns=str.lower)
    df_transformed = df_transformed[schema_columns]

    # Convert data types as per domain model requirements
    df_transformed["closedate"] = pd.to_datetime(df_transformed["closedate"], errors="coerce")
    df_transformed["id"] = df_transformed["id"].astype(int)
    df_transformed["alexa_rank"] = df_transformed["alexa_rank"].astype(float)

    # Validate domain model compliance
    for _, row in df_transformed.iterrows():
        CustomerMdl(**row.to_dict())
    return df_transformed


def transform_noncustomer_to_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe to domain-specific format.

    This is a placeholder function. Actual transformation logic should be implemented here.

    Args:
        df (pd.DataFrame): Raw dataframe.
    """
    schema_columns = [f.name for f in fields(NonCustomerMdl)]
    # Rename columns to lowercase for consistency
    df_transformed = df.rename(columns=str.lower)
    df_transformed = df_transformed[schema_columns]
    # Convert data types as per domain model requirements
    df_transformed["id"] = df_transformed["id"].astype(int)
    df_transformed["alexa_rank"] = df_transformed["alexa_rank"].astype(float)
    # Validate domain model compliance
    for _, row in df_transformed.iterrows():
        NonCustomerMdl(**row.to_dict())
    return df_transformed


def transform_usage_to_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe to domain-specific format.

    This is a placeholder function. Actual transformation logic should be implemented here.

    Args:
        df (pd.DataFrame): Raw dataframe.
    """
    schema_columns = [f.name for f in fields(UsageActionMdl)]
    # Rename columns to lowercase for consistency
    df_transformed = df.rename(columns=str.lower)
    df_transformed = df_transformed[schema_columns]
    # Convert data types as per domain model requirements
    df_transformed["when_timestamp"] = pd.to_datetime(df_transformed["when_timestamp"], errors="coerce")
    df_transformed["id"] = df_transformed["id"].astype(int)
    # Validate domain model compliance
    for _, row in df_transformed.iterrows():
        UsageActionMdl(**row.to_dict())
    return df_transformed


def standardize_text(text: str) -> str:
    if pd.isna(text):
        return None

    # Lowercase
    text = text.lower()

    # Remove accents and diacritics
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    # Replace special characters with spaces
    text = re.sub(r"[_/&-]+", " ", text)

    # Remove punctuation and parentheses
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_categorical_column(df: pd.DataFrame, column_name: str, category_size: int = 15) -> pd.DataFrame:
    df[column_name] = df[column_name].fillna("unknown")
    df[column_name] = df[column_name].replace("", "unknown")
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].apply(standardize_text)

    # Group by the categorical column and filter based on the category size
    category_counts = df[column_name].value_counts()

    # Convert small categories to 'other'
    small_categories = category_counts[category_counts < category_size].index
    df.loc[df[column_name].isin(small_categories), column_name] = "other"
    return df


def fill_missing_dates(df: pd.DataFrame, group_id: str, date_column: str,  freq: str = "7D") -> pd.DataFrame:
    # Create a full date range from min to max
    def _fill_group(group: pd.DataFrame) -> pd.DataFrame:
        full_range = pd.date_range(
            start=group[date_column].min(),
            end=group[date_column].max(),
            freq=freq
        )
        # Reindex to the full range
        group = group.set_index(date_column).reindex(full_range)
        # Fill the group id for all rows
        group[group_id] = group[group_id].iloc[0]
        # Restore the date column name
        group = group.reset_index().rename(columns={"index": date_column})
        return group

    # Apply per group and combine
    filled_df = df.groupby(group_id, group_keys=False).apply(_fill_group).reset_index(drop=True)
    return filled_df
