"""Logistic regression wrapper."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseModelWrapper


class LogisticRegressionWrapper(BaseModelWrapper):
    """Thin adapter around scikit-learn's logistic regression."""

    def __init__(self) -> None:
        super().__init__(name="logistic_regression")

    def _build_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )

    def param_distributions(self, random_state: int) -> Dict[str, Any]:  # noqa: ARG002
        grid = {
            "model__C": np.logspace(-3, 1, 20),
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
            "model__fit_intercept": [True, False],
        }
        return grid

    def feature_importance(self) -> pd.DataFrame:
        if self.pipeline_ is None:  # pragma: no cover - defensive
            raise RuntimeError("Model has not been trained yet.")
        model: LogisticRegression = self.pipeline_.named_steps["model"]
        coefs = model.coef_.ravel()
        data = {
            "feature": self.feature_names_,
            "coefficient": coefs,
            "importance": np.abs(coefs),
        }
        return pd.DataFrame(data).sort_values(by="importance", ascending=False).reset_index(drop=True)


__all__ = ["LogisticRegressionWrapper"]

