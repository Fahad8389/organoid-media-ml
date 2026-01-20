"""
Media Recipe Generator - Main Model Class
ML Coder Tasks: M1, M2 (Architecture and Masked Training)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.base import BaseEstimator
import joblib
import warnings

# Note: sys.path manipulation removed - package imports work via proper installation

from preprocessing.factor_normalizer import FactorMetadata


@dataclass
class FactorModelConfig:
    """Configuration for a factor-specific model."""
    factor_name: str
    model_type: str  # "regression", "binary_classifier", "constant"
    n_samples: int
    n_unique_values: int
    constant_value: Optional[float]
    unit: str


class ConstantPredictor(BaseEstimator):
    """Predictor that always returns a constant value."""

    def __init__(self, constant: float):
        self.constant = constant

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.constant)

    def predict_proba(self, X):
        # For compatibility with binary classifier interface
        return np.column_stack([
            np.zeros(len(X)),
            np.ones(len(X))
        ])


class MediaRecipeGenerator:
    """
    Multi-output model for predicting organoid media factor concentrations.

    Architecture:
    - Shared feature preprocessing
    - Per-factor models with masked training
    - Three model types based on factor characteristics:
      1. Regression (GradientBoosting) for multi-value factors
      2. Binary Classifier for single-value factors (predicts "is factor needed?")
      3. Constant predictor for insufficient data

    Data Integrity:
    - Each factor model trains ONLY on samples with non-NULL values
    - NULL predictions indicate "factor not needed"
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.factor_configs: Dict[str, FactorModelConfig] = {}
        self.factor_models: Dict[str, BaseEstimator] = {}
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        factor_metadata: Dict[str, FactorMetadata]
    ) -> 'MediaRecipeGenerator':
        """
        Fit per-factor models using masked training.

        Args:
            X: Feature matrix (already preprocessed)
            y: Target matrix (normalized numeric, with NaN for missing)
            factor_metadata: Metadata from FactorNormalizer

        Returns:
            self
        """
        self.feature_names = list(X.columns)

        for factor_name in y.columns:
            meta = factor_metadata.get(factor_name)
            if meta is None:
                warnings.warn(f"No metadata for factor {factor_name}, skipping")
                continue

            # Create factor config
            config = FactorModelConfig(
                factor_name=factor_name,
                model_type=meta.model_type,
                n_samples=meta.n_samples,
                n_unique_values=meta.n_unique_values,
                constant_value=meta.constant_value,
                unit=meta.unit
            )
            self.factor_configs[factor_name] = config

            # Create and fit appropriate model
            model = self._create_model(config)

            if config.model_type == "constant":
                # No fitting needed
                pass
            elif config.model_type == "binary_classifier":
                # Train to predict whether factor is used (non-NULL in original data)
                # y values are NaN where factor not used, numeric where used
                y_binary = (~y[factor_name].isna()).astype(int)
                X_clean = X.fillna(0)  # Handle any feature NaN
                model.fit(X_clean, y_binary)
            else:  # regression
                # Train only on non-NULL samples (masked training)
                mask = ~y[factor_name].isna()
                if mask.sum() > 0:
                    X_masked = X.loc[mask].fillna(0)
                    y_masked = y.loc[mask, factor_name]
                    model.fit(X_masked, y_masked)

            self.factor_models[factor_name] = model

        self.fitted = True
        return self

    def _create_model(self, config: FactorModelConfig) -> BaseEstimator:
        """Create appropriate model based on factor characteristics."""

        if config.model_type == "constant":
            return ConstantPredictor(config.constant_value or 0)

        elif config.model_type == "binary_classifier":
            # Adjust complexity based on sample size
            if config.n_samples < 50:
                return GradientBoostingClassifier(
                    n_estimators=30,
                    max_depth=2,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )
            elif config.n_samples < 100:
                return GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )
            else:
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=10,
                    random_state=self.random_state
                )

        else:  # regression
            if config.n_samples < 30:
                return GradientBoostingRegressor(
                    n_estimators=30,
                    max_depth=2,
                    min_samples_leaf=3,
                    random_state=self.random_state
                )
            elif config.n_samples < 100:
                return GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=10,
                    random_state=self.random_state
                )

    def predict(
        self,
        X: pd.DataFrame,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Predict media recipe for new samples.

        Args:
            X: Feature matrix (preprocessed)
            return_probabilities: If True, return probability of factor being needed
                                  for binary classifier factors

        Returns:
            DataFrame with predicted concentrations (normalized scale)
            NaN indicates "factor not needed"
        """
        if not self.fitted:
            raise ValueError("Must call fit() before predict()")

        X_clean = X.fillna(0)
        predictions = {}

        for factor_name, model in self.factor_models.items():
            config = self.factor_configs[factor_name]

            if config.model_type == "constant":
                # Always predict the constant value
                pred = model.predict(X_clean)
                predictions[factor_name] = pred

            elif config.model_type == "binary_classifier":
                # Predict probability/class of factor being needed
                if return_probabilities:
                    proba = model.predict_proba(X_clean)[:, 1]
                    predictions[factor_name] = proba
                else:
                    is_needed = model.predict(X_clean)
                    # Return constant value where needed, NaN where not
                    pred = np.where(
                        is_needed == 1,
                        config.constant_value,
                        np.nan
                    )
                    predictions[factor_name] = pred

            else:  # regression
                pred = model.predict(X_clean)
                predictions[factor_name] = pred

        return pd.DataFrame(predictions, index=X.index)

    def predict_recipe(
        self,
        X: pd.DataFrame,
        normalizer: Any,  # FactorNormalizer
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate human-readable media recipes.

        Args:
            X: Feature matrix (preprocessed)
            normalizer: FactorNormalizer for inverse transform
            threshold: Probability threshold for binary classifiers

        Returns:
            List of recipe dictionaries, one per sample
        """
        # Get predictions
        pred_normalized = self.predict(X)

        # Inverse transform to original scale
        pred_original = normalizer.inverse_transform(pred_normalized)

        # Format as strings with units
        pred_formatted = normalizer.format_predictions(pred_original)

        # Convert to list of dictionaries
        recipes = []
        for idx in pred_formatted.index:
            recipe = {}
            for factor in pred_formatted.columns:
                value = pred_formatted.loc[idx, factor]
                config = self.factor_configs[factor]

                recipe[factor] = {
                    'concentration': value,
                    'model_type': config.model_type,
                    'unit': config.unit
                }

            recipes.append(recipe)

        return recipes

    def get_feature_importance(self, factor_name: str) -> pd.Series:
        """
        Get feature importance for a specific factor model.

        Args:
            factor_name: Name of the factor

        Returns:
            Series of feature importances
        """
        if factor_name not in self.factor_models:
            raise ValueError(f"Unknown factor: {factor_name}")

        model = self.factor_models[factor_name]
        config = self.factor_configs[factor_name]

        if config.model_type == "constant":
            # No meaningful importance for constant predictor
            return pd.Series(dtype=float)

        # GradientBoosting models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            return pd.Series(
                model.feature_importances_,
                index=self.feature_names,
                name=factor_name
            ).sort_values(ascending=False)

        return pd.Series(dtype=float)

    def get_all_feature_importances(self) -> pd.DataFrame:
        """Get feature importances for all factors as a matrix."""
        importances = {}

        for factor_name in self.factor_models.keys():
            imp = self.get_feature_importance(factor_name)
            if len(imp) > 0:
                importances[factor_name] = imp

        if importances:
            return pd.DataFrame(importances)

        return pd.DataFrame()

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'factor_configs': self.factor_configs,
            'factor_models': self.factor_models,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'fitted': self.fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MediaRecipeGenerator':
        """Load model from disk."""
        data = joblib.load(filepath)
        model = cls(random_state=data['random_state'])
        model.factor_configs = data['factor_configs']
        model.factor_models = data['factor_models']
        model.feature_names = data['feature_names']
        model.fitted = data['fitted']
        return model

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all factor models."""
        rows = []
        for factor_name, config in self.factor_configs.items():
            rows.append({
                'factor': factor_name,
                'model_type': config.model_type,
                'n_samples': config.n_samples,
                'n_unique': config.n_unique_values,
                'constant_value': config.constant_value,
                'unit': config.unit
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test the model
    print("MediaRecipeGenerator module loaded successfully")
    print("\nModel types supported:")
    print("  - Regression (GradientBoosting): For factors with multiple concentration values")
    print("  - Binary Classifier: For factors with single value (predicts 'is needed?')")
    print("  - Constant Predictor: For factors with insufficient data")
