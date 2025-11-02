from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class RawDataSourcesConfig:
    """Locations for raw data files."""

    customers: str
    noncustomers: str
    usage_actions: str
    base_dir: Path | str
    absolute_paths: bool = False

    def resolve(self, key: str) -> Path:
        base = Path(self.base_dir) if isinstance(self.base_dir, str) else self.base_dir
        mapping: Dict[str, str] = {
            "customers": self.customers,
            "noncustomers": self.noncustomers,
            "usage_actions": self.usage_actions,
        }
        try:
            relative = mapping[key]
        except KeyError as exc:
            raise ValueError(f"Unknown datasource key: {key}") from exc
        return (base / relative).resolve()


def _read_csv(path: Path, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file missing: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def load_customers(config: RawDataSourcesConfig) -> pd.DataFrame:
    path = config.resolve("customers")
    return _read_csv(path, parse_dates=["CLOSEDATE"])


def load_noncustomers(config: RawDataSourcesConfig) -> pd.DataFrame:
    path = config.resolve("noncustomers")
    return _read_csv(path)


def load_usage(config: RawDataSourcesConfig) -> pd.DataFrame:
    path = config.resolve("usage_actions")
    df = _read_csv(path)
    if "WHEN_TIMESTAMP" in df.columns:
        df["WHEN_TIMESTAMP"] = pd.to_datetime(df["WHEN_TIMESTAMP"])
    return df


def load_raw_files(config: RawDataSourcesConfig) -> Dict[str, pd.DataFrame]:
    return {
        "customers": load_customers(config),
        "noncustomers": load_noncustomers(config),
        "usage": load_usage(config),
    }


__all__ = [
    "RawDataSourcesConfig",
    "load_customers",
    "load_noncustomers",
    "load_usage",
    "load_raw_files",
]
