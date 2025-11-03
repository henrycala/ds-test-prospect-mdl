"""Model wrapper abstractions and utilities."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


class BaseModelWrapper(ABC):
    """Common functionality every experiment model should expose."""

    name: str
    pipeline_: Optional[Pipeline]
    feature_names_: list[str]
    params_: Dict[str, Any]

    def __init__(self, name: str) -> None:
        self.name = name
        self.pipeline_ = None
        self.feature_names_ = []
        self.params_ = {}

    # ----- Abstract API -------------------------------------------------
    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return the base estimator for this wrapper."""

    @abstractmethod
    def param_distributions(self, random_state: int) -> Dict[str, Any]:
        """Default hyper-parameter search space."""

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores for the trained model."""

    # ----- Public API ---------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        params: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[np.ndarray] = None,
        fine_tune_params: bool = False,
    ):
        """Fit a pipeline and keep track of metrics on the training data."""
        if fine_tune_params and params is not None:
            params = self.fine_tune_params(X, y)

        estimator = self._build_estimator()
        pipeline = Pipeline(
            steps=[
                ("model", estimator),
            ]
        )
        if params:
            pipeline.set_params(**params)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        pipeline.fit(X, y, **fit_kwargs)

        self.pipeline_ = pipeline
        self.params_ = params or {}
        return 


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:  # pragma: no cover - defensive
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline_.predict(X)


    def fine_tune_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        scoring: str = "roc_auc",
        cv: int = 5,
        n_iter: int = 30,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run a lightweight hyper-parameter search and return the best config."""

        estimator = self._build_estimator()
        pipeline = Pipeline(
            steps=[
                ("model", estimator),
            ]
        )
        distributions = self.param_distributions(random_state)

        search = self._build_search(
            pipeline=pipeline,
            param_distributions=distributions,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        search.fit(X, y)
        best = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
        }
        self.params_ = best["best_params"]
        return best


    # ----- Internal helpers ---------------------------------------------
    def _build_search(
        self,
        *,
        pipeline: Pipeline,
        param_distributions: Dict[str, Any],
        scoring: str,
        cv: int,
        n_iter: int,
        random_state: int,
        n_jobs: int,
    ):
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )

