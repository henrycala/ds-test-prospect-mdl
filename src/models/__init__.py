"""Model wrappers for prospect experiments."""

from .base import BaseModelWrapper
from .logistic import LogisticRegressionWrapper

try:  # Optional dependency
    from .xgboost import XGBoostWrapper
except ImportError:  # pragma: no cover
    XGBoostWrapper = None  # type: ignore[assignment]

__all__ = [
    "BaseModelWrapper",
    "LogisticRegressionWrapper",
    "XGBoostWrapper",
]

