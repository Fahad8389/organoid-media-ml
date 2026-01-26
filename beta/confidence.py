"""
Confidence Scoring System for Beta Model
Provides prediction confidence for demo purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    """Confidence scoring result."""
    overall_confidence: float
    factor_confidences: Dict[str, float]
    reliability_grade: str  # A, B, C, D
    warnings: list


class ConfidenceScorer:
    """
    Confidence scoring for media recipe predictions.

    Factors considered:
    1. Model confidence (prediction probability/variance)
    2. Data coverage (is this sample similar to training data?)
    3. Factor reliability (based on CV scores)
    """

    # CV-based reliability scores (from training)
    DEFAULT_FACTOR_RELIABILITY = {
        'egf': 0.96,           # R² = 0.96
        'y27632': 0.94,        # AUC = 0.94
        'n_acetyl_cysteine': 0.89,  # AUC = 0.89
        'a83_01': 0.96,        # AUC = 0.96
        'sb202190': 0.95,      # AUC = 0.95
        'fgf2': 0.93,          # AUC = 0.93
        'cholera_toxin': 0.89, # AUC = 0.89
        'insulin': 0.91,       # AUC = 0.91
    }

    def __init__(
        self,
        factor_reliability: Optional[Dict[str, float]] = None,
        cv_scores: Optional[Dict[str, float]] = None
    ):
        """
        Initialize confidence scorer.

        Args:
            factor_reliability: Per-factor reliability scores (0-1)
            cv_scores: Cross-validation scores from training
        """
        self.factor_reliability = factor_reliability or self.DEFAULT_FACTOR_RELIABILITY
        self.cv_scores = cv_scores or {}

    def score(
        self,
        predictions: pd.DataFrame,
        model_confidences: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Score confidence for predictions.

        Args:
            predictions: Model predictions
            model_confidences: Per-factor confidence from model

        Returns:
            DataFrame with confidence scores
        """
        results = []

        for idx in predictions.index:
            sample_pred = predictions.loc[idx]
            sample_conf = model_confidences.loc[idx] if model_confidences is not None else None

            result = self._score_sample(sample_pred, sample_conf)
            results.append({
                'overall_confidence': result.overall_confidence,
                'reliability_grade': result.reliability_grade,
                **{f'{f}_confidence': c for f, c in result.factor_confidences.items()}
            })

        return pd.DataFrame(results, index=predictions.index)

    def _score_sample(
        self,
        prediction: pd.Series,
        model_confidence: Optional[pd.Series] = None
    ) -> ConfidenceResult:
        """Score a single sample."""
        factor_confidences = {}
        warnings = []

        for factor in prediction.index:
            # Base reliability from CV
            base_reliability = self.factor_reliability.get(factor, 0.7)

            # Model confidence adjustment
            if model_confidence is not None and factor in model_confidence.index:
                model_conf = model_confidence[factor]
            else:
                model_conf = 0.8  # Default

            # NaN prediction = low confidence
            if pd.isna(prediction[factor]):
                factor_conf = 0.5  # Uncertain about "not needed"
                warnings.append(f"{factor}: predicted as not needed")
            else:
                factor_conf = base_reliability * 0.6 + model_conf * 0.4

            factor_confidences[factor] = round(factor_conf, 3)

        # Overall confidence = weighted average
        valid_confs = [c for c in factor_confidences.values() if not np.isnan(c)]
        overall = np.mean(valid_confs) if valid_confs else 0.5

        # Assign grade
        if overall >= 0.9:
            grade = 'A'
        elif overall >= 0.8:
            grade = 'B'
        elif overall >= 0.7:
            grade = 'C'
        else:
            grade = 'D'

        return ConfidenceResult(
            overall_confidence=round(overall, 3),
            factor_confidences=factor_confidences,
            reliability_grade=grade,
            warnings=warnings
        )

    def get_reliability_report(self) -> pd.DataFrame:
        """Get factor reliability report."""
        rows = []
        for factor, score in self.factor_reliability.items():
            cv_score = self.cv_scores.get(factor, score)
            rows.append({
                'factor': factor,
                'reliability': score,
                'cv_score': cv_score,
                'grade': 'A' if score >= 0.9 else 'B' if score >= 0.8 else 'C'
            })
        return pd.DataFrame(rows)

    def update_from_cv(self, cv_results: Dict[str, float]):
        """Update reliability scores from CV results."""
        for factor, score in cv_results.items():
            self.factor_reliability[factor] = score
            self.cv_scores[factor] = score


def calculate_prediction_intervals(
    predictions: pd.DataFrame,
    factor_stds: Dict[str, float],
    confidence_level: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate prediction intervals for regression factors.

    Args:
        predictions: Point predictions
        factor_stds: Standard deviations from CV
        confidence_level: Confidence level (default 95%)

    Returns:
        (lower_bounds, upper_bounds)
    """
    from scipy import stats

    z = stats.norm.ppf((1 + confidence_level) / 2)

    lower = predictions.copy()
    upper = predictions.copy()

    for factor in predictions.columns:
        std = factor_stds.get(factor, 0)
        lower[factor] = predictions[factor] - z * std
        upper[factor] = predictions[factor] + z * std

    return lower, upper
