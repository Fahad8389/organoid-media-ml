"""
Recipe Evaluator - Hierarchical metrics for media recipe predictions
Management Agent Tasks: G4 (Success Rate metrics)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)
import json


@dataclass
class FactorMetrics:
    """Metrics for a single factor."""
    factor_name: str
    model_type: str
    n_samples: int
    # Regression metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    discrete_accuracy: Optional[float] = None  # Rounds to nearest valid value
    # Classification metrics
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None


@dataclass
class RecipeMetrics:
    """Overall recipe-level metrics."""
    n_samples: int
    n_factors: int
    # Aggregate regression
    weighted_r2: Optional[float] = None
    mean_mae: Optional[float] = None
    # Aggregate classification
    mean_auc: Optional[float] = None
    mean_f1: Optional[float] = None
    # Recipe-level
    exact_match_rate: float = 0.0
    partial_match_rate: float = 0.0
    weighted_recipe_score: float = 0.0


class RecipeEvaluator:
    """
    Comprehensive evaluation framework for media recipe predictions.

    Evaluation Levels:
    1. Per-Factor Metrics: MAE, RMSE, R² (regression) or Accuracy, F1, AUC (classification)
    2. Aggregate Metrics: Weighted averages across factors
    3. Recipe-Level Metrics: How well complete recipes match
    """

    def __init__(
        self,
        factor_metadata: Dict,
        importance_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            factor_metadata: Metadata from FactorNormalizer
            importance_weights: Optional biological importance weights per factor
        """
        self.factor_metadata = factor_metadata
        self.importance_weights = importance_weights or {}

        # Default weights if not provided
        default_weights = {
            'egf': 1.0,       # Critical growth factor
            'y27632': 0.8,    # ROCK inhibitor, important for survival
            'a83_01': 0.6,    # TGF-beta inhibitor
            'n_acetyl_cysteine': 0.5,
            'sb202190': 0.5,  # p38 MAPK inhibitor
            'fgf2': 0.7,      # FGF signaling
            'cholera_toxin': 0.4,
            'insulin': 0.3
        }

        for factor, weight in default_weights.items():
            if factor not in self.importance_weights:
                self.importance_weights[factor] = weight

    def evaluate_factor(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        factor_name: str
    ) -> FactorMetrics:
        """
        Evaluate predictions for a single factor.

        Args:
            y_true: True values (numeric, with NaN)
            y_pred: Predicted values (numeric, with NaN)
            factor_name: Name of the factor

        Returns:
            FactorMetrics
        """
        meta = self.factor_metadata.get(factor_name)
        if meta is None:
            model_type = "unknown"
            n_samples = 0
        else:
            model_type = meta.model_type
            n_samples = meta.n_samples

        metrics = FactorMetrics(
            factor_name=factor_name,
            model_type=model_type,
            n_samples=n_samples
        )

        if model_type == "regression":
            # Evaluate on samples where both true and pred are non-NULL
            mask = ~(y_true.isna() | y_pred.isna())
            if mask.sum() == 0:
                return metrics

            y_t = y_true[mask].values
            y_p = y_pred[mask].values

            metrics.mae = mean_absolute_error(y_t, y_p)
            metrics.rmse = np.sqrt(mean_squared_error(y_t, y_p))
            metrics.r2 = r2_score(y_t, y_p) if len(y_t) > 1 else None

            # Discrete accuracy: round to nearest valid value
            if meta and meta.unique_values:
                valid_values = np.array(meta.unique_values)
                y_p_discrete = np.array([
                    valid_values[np.argmin(np.abs(valid_values - p))]
                    for p in y_p
                ])
                metrics.discrete_accuracy = np.mean(np.isclose(y_p_discrete, y_t, rtol=0.1))

        elif model_type == "binary_classifier":
            # Convert to binary: 1 if non-NULL (factor used), 0 if NULL
            y_t_binary = (~y_true.isna()).astype(int).values
            y_p_binary = (~y_pred.isna()).astype(int).values

            metrics.accuracy = accuracy_score(y_t_binary, y_p_binary)
            metrics.f1 = f1_score(y_t_binary, y_p_binary, zero_division=0)

            # For AUC, we'd need probability scores (not available here)

        return metrics

    def evaluate_all_factors(
        self,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame
    ) -> Dict[str, FactorMetrics]:
        """Evaluate all factors."""
        results = {}
        for factor in y_true.columns:
            if factor in y_pred.columns:
                results[factor] = self.evaluate_factor(
                    y_true[factor],
                    y_pred[factor],
                    factor
                )
        return results

    def evaluate_recipe(
        self,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        tolerance: float = 0.1
    ) -> Tuple[Dict[str, FactorMetrics], RecipeMetrics]:
        """
        Full evaluation of recipe predictions.

        Args:
            y_true: True values DataFrame
            y_pred: Predicted values DataFrame
            tolerance: Relative tolerance for considering values equal

        Returns:
            Tuple of (per-factor metrics dict, recipe metrics)
        """
        # Per-factor metrics
        factor_metrics = self.evaluate_all_factors(y_true, y_pred)

        # Aggregate metrics
        recipe_metrics = RecipeMetrics(
            n_samples=len(y_true),
            n_factors=len(y_true.columns)
        )

        # Weighted R² for regression factors
        regression_factors = [
            (name, m) for name, m in factor_metrics.items()
            if m.model_type == "regression" and m.r2 is not None
        ]
        if regression_factors:
            r2_values = [m.r2 for _, m in regression_factors]
            sample_counts = [m.n_samples for _, m in regression_factors]
            recipe_metrics.weighted_r2 = np.average(r2_values, weights=sample_counts)

            mae_values = [m.mae for _, m in regression_factors if m.mae is not None]
            if mae_values:
                recipe_metrics.mean_mae = np.mean(mae_values)

        # Mean metrics for classification factors
        classification_factors = [
            (name, m) for name, m in factor_metrics.items()
            if m.model_type == "binary_classifier"
        ]
        if classification_factors:
            f1_values = [m.f1 for _, m in classification_factors if m.f1 is not None]
            if f1_values:
                recipe_metrics.mean_f1 = np.mean(f1_values)

        # Recipe-level metrics
        exact_matches = 0
        partial_match_scores = []
        weighted_scores = []

        for i in range(len(y_true)):
            n_correct = 0
            n_total = 0
            weighted_correct = 0
            total_weight = 0

            for factor in y_true.columns:
                true_val = y_true[factor].iloc[i]
                pred_val = y_pred[factor].iloc[i]
                weight = self.importance_weights.get(factor, 1.0)

                total_weight += weight
                n_total += 1

                # Check if values match
                if pd.isna(true_val) and pd.isna(pred_val):
                    # Both NULL = correct
                    n_correct += 1
                    weighted_correct += weight
                elif pd.isna(true_val) or pd.isna(pred_val):
                    # One NULL, one not = incorrect
                    pass
                elif np.isclose(true_val, pred_val, rtol=tolerance):
                    # Values match within tolerance
                    n_correct += 1
                    weighted_correct += weight

            if n_correct == n_total:
                exact_matches += 1

            partial_match_scores.append(n_correct / n_total if n_total > 0 else 0)
            weighted_scores.append(weighted_correct / total_weight if total_weight > 0 else 0)

        recipe_metrics.exact_match_rate = exact_matches / len(y_true) if len(y_true) > 0 else 0
        recipe_metrics.partial_match_rate = np.mean(partial_match_scores)
        recipe_metrics.weighted_recipe_score = np.mean(weighted_scores)

        return factor_metrics, recipe_metrics

    def generate_report(
        self,
        factor_metrics: Dict[str, FactorMetrics],
        recipe_metrics: RecipeMetrics
    ) -> str:
        """Generate human-readable evaluation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("MEDIA RECIPE GENERATOR - EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append("## Recipe-Level Metrics")
        lines.append(f"  Total samples: {recipe_metrics.n_samples}")
        lines.append(f"  Total factors: {recipe_metrics.n_factors}")
        lines.append(f"  Exact match rate: {recipe_metrics.exact_match_rate:.2%}")
        lines.append(f"  Partial match rate: {recipe_metrics.partial_match_rate:.2%}")
        lines.append(f"  Weighted recipe score: {recipe_metrics.weighted_recipe_score:.2%}")
        lines.append("")

        if recipe_metrics.weighted_r2 is not None:
            lines.append("## Aggregate Regression Metrics")
            lines.append(f"  Weighted R²: {recipe_metrics.weighted_r2:.4f}")
            if recipe_metrics.mean_mae is not None:
                lines.append(f"  Mean MAE: {recipe_metrics.mean_mae:.4f}")
            lines.append("")

        if recipe_metrics.mean_f1 is not None:
            lines.append("## Aggregate Classification Metrics")
            lines.append(f"  Mean F1: {recipe_metrics.mean_f1:.4f}")
            lines.append("")

        lines.append("## Per-Factor Metrics")
        lines.append("-" * 60)

        for factor_name, metrics in sorted(factor_metrics.items()):
            lines.append(f"\n### {factor_name}")
            lines.append(f"  Model type: {metrics.model_type}")
            lines.append(f"  Samples: {metrics.n_samples}")

            if metrics.model_type == "regression":
                if metrics.mae is not None:
                    lines.append(f"  MAE: {metrics.mae:.4f}")
                if metrics.rmse is not None:
                    lines.append(f"  RMSE: {metrics.rmse:.4f}")
                if metrics.r2 is not None:
                    lines.append(f"  R²: {metrics.r2:.4f}")
                if metrics.discrete_accuracy is not None:
                    lines.append(f"  Discrete Accuracy: {metrics.discrete_accuracy:.2%}")

            elif metrics.model_type == "binary_classifier":
                if metrics.accuracy is not None:
                    lines.append(f"  Accuracy: {metrics.accuracy:.2%}")
                if metrics.f1 is not None:
                    lines.append(f"  F1: {metrics.f1:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_json(
        self,
        factor_metrics: Dict[str, FactorMetrics],
        recipe_metrics: RecipeMetrics
    ) -> str:
        """Export metrics to JSON format."""
        data = {
            'recipe_metrics': asdict(recipe_metrics),
            'factor_metrics': {
                name: asdict(m) for name, m in factor_metrics.items()
            }
        }
        return json.dumps(data, indent=2, default=str)


if __name__ == "__main__":
    print("Recipe Evaluator module loaded successfully")
    print("\nEvaluation hierarchy:")
    print("  1. Per-Factor: MAE, RMSE, R² (regression) / Accuracy, F1 (classification)")
    print("  2. Aggregate: Weighted R², Mean F1")
    print("  3. Recipe-Level: Exact match rate, Partial match rate, Weighted score")
