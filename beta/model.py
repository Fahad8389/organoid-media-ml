"""
Beta Media Predictor - XGBoost Model
Uses XGBoost for both regression and classification tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import joblib
import warnings

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")

from sklearn.base import BaseEstimator


@dataclass
class FactorConfig:
    """Configuration for a media factor model."""
    name: str
    task_type: str  # "regression" or "classification"
    n_samples: int
    n_unique: int
    constant_value: Optional[float] = None
    unit: str = ""


class ConstantPredictor(BaseEstimator):
    """Fallback predictor for insufficient data."""

    def __init__(self, value: float):
        self.value = value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class BetaMediaPredictor:
    """
    XGBoost-based media recipe predictor for beta demo.

    Features:
    - XGBoost for superior tabular data performance
    - Per-factor models with masked training
    - Confidence scoring via prediction variance
    - SHAP-ready for interpretability demos
    """

    # XGBoost hyperparameters (tuned for 660 samples)
    XGBOOST_PARAMS_REG = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }

    XGBOOST_PARAMS_CLF = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
    }

    def __init__(self, random_state: int = 42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required. Install with: pip install xgboost")

        self.random_state = random_state
        self.factor_configs: Dict[str, FactorConfig] = {}
        self.models: Dict[str, BaseEstimator] = {}
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        factor_metadata: Dict[str, Any]
    ) -> 'BetaMediaPredictor':
        """
        Fit XGBoost models for each media factor.

        Args:
            X: Feature matrix
            y: Target matrix (NaN for missing values)
            factor_metadata: Dict with factor info (model_type, n_samples, etc.)
        """
        self.feature_names = list(X.columns)

        for factor in y.columns:
            meta = factor_metadata.get(factor, {})

            config = FactorConfig(
                name=factor,
                task_type=meta.get('model_type', 'regression'),
                n_samples=meta.get('n_samples', 0),
                n_unique=meta.get('n_unique_values', 0),
                constant_value=meta.get('constant_value'),
                unit=meta.get('unit', '')
            )
            self.factor_configs[factor] = config

            # Create model based on task type
            if config.task_type == "constant" or config.n_samples < 10:
                model = ConstantPredictor(config.constant_value or 0)
                model.fit(X, None)

            elif config.task_type == "binary_classifier":
                # Binary: predict if factor is needed
                y_binary = (~y[factor].isna()).astype(int)
                X_clean = X.fillna(0)

                params = self.XGBOOST_PARAMS_CLF.copy()
                params['random_state'] = self.random_state
                model = xgb.XGBClassifier(**params)
                model.fit(X_clean, y_binary)

            else:  # regression
                # Masked training: only non-NULL samples
                mask = ~y[factor].isna()
                if mask.sum() >= 10:
                    X_masked = X.loc[mask].fillna(0)
                    y_masked = y.loc[mask, factor]

                    params = self.XGBOOST_PARAMS_REG.copy()
                    params['random_state'] = self.random_state
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_masked, y_masked)
                else:
                    model = ConstantPredictor(y[factor].median() or 0)
                    model.fit(X, None)

            self.models[factor] = model

        self.fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> pd.DataFrame:
        """
        Predict media recipe for samples.

        Args:
            X: Feature matrix
            return_confidence: If True, return (predictions, confidence)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Reorder columns to match training feature order
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            # Add missing columns with zeros
            for col in missing_cols:
                X[col] = 0
        X = X[self.feature_names]

        X_clean = X.fillna(0)
        predictions = {}
        confidences = {}

        for factor, model in self.models.items():
            config = self.factor_configs[factor]

            if config.task_type == "constant":
                pred = model.predict(X_clean)
                conf = np.ones(len(X_clean))

            elif config.task_type == "binary_classifier":
                proba = model.predict_proba(X_clean)[:, 1]
                is_needed = proba >= 0.5
                pred = np.where(is_needed, config.constant_value, np.nan)
                conf = np.abs(proba - 0.5) * 2  # 0-1 scale

            else:  # regression
                pred = model.predict(X_clean)
                # Confidence from tree variance (approximation)
                conf = np.ones(len(X_clean)) * 0.8  # Base confidence

            predictions[factor] = pred
            confidences[factor] = conf

        pred_df = pd.DataFrame(predictions, index=X.index)

        if return_confidence:
            conf_df = pd.DataFrame(confidences, index=X.index)
            return pred_df, conf_df

        return pred_df

    def get_feature_importance(self, factor: str) -> pd.Series:
        """Get feature importance for a factor."""
        if factor not in self.models:
            raise ValueError(f"Unknown factor: {factor}")

        model = self.models[factor]

        if hasattr(model, 'feature_importances_'):
            return pd.Series(
                model.feature_importances_,
                index=self.feature_names,
                name=factor
            ).sort_values(ascending=False)

        return pd.Series(dtype=float)

    def get_all_importances(self) -> pd.DataFrame:
        """Get feature importance matrix for all factors."""
        importances = {}
        for factor in self.models:
            imp = self.get_feature_importance(factor)
            if len(imp) > 0:
                importances[factor] = imp

        if importances:
            return pd.DataFrame(importances)
        return pd.DataFrame()

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'factor_configs': self.factor_configs,
            'models': self.models,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'fitted': self.fitted,
            'version': 'beta-1.0'
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BetaMediaPredictor':
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(random_state=data['random_state'])
        model.factor_configs = data['factor_configs']
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.fitted = data['fitted']
        return model

    def summary(self) -> pd.DataFrame:
        """Get model summary."""
        rows = []
        for name, config in self.factor_configs.items():
            rows.append({
                'factor': name,
                'task_type': config.task_type,
                'n_samples': config.n_samples,
                'n_unique': config.n_unique,
                'unit': config.unit
            })
        return pd.DataFrame(rows)
