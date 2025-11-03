"""XGBoost classifier wrapper."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    """Adapter around xgboost.XGBClassifier with convenience helpers."""

    def __init__(self) -> None:
        super().__init__(name="xgboost_classifier")

    def _build_estimator(self) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )

    def param_distributions(self, random_state: int) -> Dict[str, Any]:  # noqa: ARG002
        return {
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": np.linspace(0.01, 0.3, 10),
            "model__n_estimators": [200, 400, 600, 800],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0.0, 0.5, 1.0],
            "model__reg_lambda": [0.0, 1.0, 5.0, 10.0],
            "model__reg_alpha": [0.0, 0.5, 1.0],
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.pipeline_ is None:  # pragma: no cover - defensive
            raise RuntimeError("Model has not been trained yet.")
        model: XGBClassifier = self.pipeline_.named_steps["model"]
        importances = model.feature_importances_
        data = {
            "feature": self.feature_names_,
            "importance": importances,
        }
        return pd.DataFrame(data).sort_values(by="importance", ascending=False).reset_index(drop=True)


__all__ = ["XGBoostWrapper"]

