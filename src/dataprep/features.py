import pandas as pd

def generate_time_window_features(
    df: pd.DataFrame,
    group_id: str,
    date_col: str,
    numeric_cols: list[str],
    window_size: int = 4,
) -> pd.DataFrame:
    """
    Efficiently generate rolling window features (mean, min, max, first, last) 
    for multiple numeric columns within each group.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_id (str): Column name for the group ID (e.g., account_id).
        date_col (str): Timestamp column name.
        numeric_cols (list[str]): List of numeric columns to calculate stats.
        window_size (int): Rolling window size (number of periods).

    Returns:
        pd.DataFrame: Dataframe with new rolling window features added.
    """
    df = df.copy().sort_values(by=[group_id, date_col])
    grouped = df.groupby(group_id, group_keys=False)

    # Preallocate dictionary for new features
    feature_dict = {}

    for col in numeric_cols:
        roll = grouped[col].rolling(window=window_size, min_periods=1)
        feature_dict[f"{col}_mean_{window_size}"] = roll.mean().reset_index(level=0, drop=True)
        feature_dict[f"{col}_min_{window_size}"] = roll.min().reset_index(level=0, drop=True)
        feature_dict[f"{col}_max_{window_size}"] = roll.max().reset_index(level=0, drop=True)
        # Use shift-based approach for first/last (avoids slow .apply())
        feature_dict[f"{col}_first_{window_size}"] = (
            grouped[col].transform(lambda x: x.rolling(window_size, min_periods=1).apply(lambda w: w.iloc[0], raw=False))
        )
        feature_dict[f"{col}_last_{window_size}"] = (
            grouped[col].transform(lambda x: x.rolling(window_size, min_periods=1).apply(lambda w: w.iloc[-1], raw=False))
        )

    # Combine features efficiently (no duplicate columns)
    features = pd.DataFrame(feature_dict, index=df.index)
    return pd.concat([df, features], axis=1)


def mark_closest_when_date(df, group_col, when_col, close_col, max_gap_days=14, target_name="mark"):
    """
    Mark the record with when_date closest (and <=) to closedate, per group.
    Marks 1 only if difference <= max_gap_days; else remains 0.
    If closedate is missing, mark remains 0.
    """
    df = df.copy()
    df[target_name] = 0

    def mark_group(group):
        closedate = group[close_col].iloc[0]

        # Skip if closedate is missing
        if pd.isna(closedate):
            return group

        # Keep only rows before or equal to closedate
        valid = group[group[when_col] <= closedate]
        if valid.empty:
            return group

        # Compute absolute time difference
        time_diffs = (closedate - valid[when_col]).abs()
        min_diff = time_diffs.min()

        # Only mark if difference ≤ max_gap_days
        if min_diff <= pd.Timedelta(days=max_gap_days):
            closest_idx = valid.index[time_diffs.argmin()]
            group.loc[closest_idx, target_name] = 1

        return group

    return df.groupby(group_col, group_keys=False).apply(mark_group)

__all__ = ["generate_time_window_features", "mark_closest_when_date"]
