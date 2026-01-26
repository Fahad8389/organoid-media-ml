"""
Simple API Interface for Beta Model
For demo purposes - easy prediction interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from .model import BetaMediaPredictor
from .preprocessing import BetaPreprocessor
from .confidence import ConfidenceScorer
from .validators import InputValidator, ValidationResult


@dataclass
class PredictionResult:
    """Result of a media recipe prediction."""
    success: bool
    recipe: Optional[Dict[str, float]]
    confidence: Optional[Dict[str, float]]
    overall_confidence: float
    grade: str  # A, B, C, D
    warnings: List[str]
    error: Optional[str] = None


class BetaAPI:
    """
    Simple API for media recipe prediction.

    Example:
        api = BetaAPI.load('beta_output/')
        result = api.predict_single({
            'primary_site': 'Breast',
            'gender': 'Female',
            'age_at_diagnosis_years': 55
        })
        print(result.recipe)
    """

    def __init__(
        self,
        model: BetaMediaPredictor,
        preprocessor: BetaPreprocessor,
        validator: Optional[InputValidator] = None,
        scorer: Optional[ConfidenceScorer] = None
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.validator = validator or InputValidator(strict=False)
        self.scorer = scorer or ConfidenceScorer()

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> 'BetaAPI':
        """
        Load API from saved model directory.

        Args:
            model_dir: Path to beta_output/ directory

        Returns:
            BetaAPI instance
        """
        model_dir = Path(model_dir)

        model = BetaMediaPredictor.load(str(model_dir / 'beta_model.joblib'))
        preprocessor = BetaPreprocessor.load(str(model_dir / 'beta_preprocessor.joblib'))

        # Load CV scores for confidence if available
        import json
        metrics_path = model_dir / 'beta_metrics.json'
        scorer = ConfidenceScorer()
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            cv_results = metrics.get('cv_results', {})
            cv_scores = {f: r['score'] for f, r in cv_results.items() if 'score' in r}
            scorer.update_from_cv(cv_scores)

        return cls(model, preprocessor, scorer=scorer)

    def predict_single(
        self,
        sample: Dict,
        return_raw: bool = False
    ) -> PredictionResult:
        """
        Predict media recipe for a single sample.

        Args:
            sample: Dictionary with sample features
            return_raw: If True, return raw normalized values

        Returns:
            PredictionResult
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample])

        # Validate
        validation = self.validator.validate(df)
        if not validation.is_valid:
            return PredictionResult(
                success=False,
                recipe=None,
                confidence=None,
                overall_confidence=0,
                grade='F',
                warnings=validation.warnings,
                error='; '.join(validation.errors)
            )

        # Preprocess
        try:
            X, _ = self.preprocessor.transform(validation.cleaned_data)
        except Exception as e:
            return PredictionResult(
                success=False,
                recipe=None,
                confidence=None,
                overall_confidence=0,
                grade='F',
                warnings=validation.warnings,
                error=f"Preprocessing error: {str(e)}"
            )

        # Predict
        predictions, confidences = self.model.predict(X, return_confidence=True)

        # Inverse transform to original scale
        if not return_raw:
            predictions = self.preprocessor.inverse_transform_targets(predictions)

        # Get recipe as dict
        recipe = predictions.iloc[0].to_dict()

        # Clean up NaN values
        recipe = {k: (v if not pd.isna(v) else None) for k, v in recipe.items()}

        # Score confidence
        conf_result = self.scorer._score_sample(
            predictions.iloc[0],
            confidences.iloc[0]
        )

        return PredictionResult(
            success=True,
            recipe=recipe,
            confidence=conf_result.factor_confidences,
            overall_confidence=conf_result.overall_confidence,
            grade=conf_result.reliability_grade,
            warnings=validation.warnings + conf_result.warnings
        )

    def predict_batch(
        self,
        samples: Union[pd.DataFrame, List[Dict]],
        return_raw: bool = False
    ) -> pd.DataFrame:
        """
        Predict media recipes for multiple samples.

        Args:
            samples: DataFrame or list of dicts
            return_raw: If True, return raw normalized values

        Returns:
            DataFrame with predictions
        """
        if isinstance(samples, list):
            df = pd.DataFrame(samples)
        else:
            df = samples.copy()

        # Validate
        validation = self.validator.validate(df)
        if not validation.is_valid:
            raise ValueError(f"Validation failed: {validation.errors}")

        # Preprocess
        X, _ = self.preprocessor.transform(validation.cleaned_data)

        # Predict
        predictions = self.model.predict(X)

        # Inverse transform
        if not return_raw:
            predictions = self.preprocessor.inverse_transform_targets(predictions)

        return predictions

    def get_recipe_string(self, result: PredictionResult) -> str:
        """
        Format prediction result as readable string.

        Args:
            result: PredictionResult from predict_single

        Returns:
            Formatted string
        """
        if not result.success:
            return f"Prediction failed: {result.error}"

        lines = [
            "=" * 50,
            "MEDIA RECIPE PREDICTION",
            "=" * 50,
            f"Confidence: {result.overall_confidence:.1%} (Grade {result.grade})",
            "-" * 50,
        ]

        units = {
            'egf': 'ng/mL',
            'y27632': 'uM',
            'n_acetyl_cysteine': 'mM',
            'a83_01': 'nM',
            'sb202190': 'uM',
            'fgf2': 'ng/mL',
            'cholera_toxin': 'ng/mL',
            'insulin': 'ug/mL',
        }

        for factor, value in result.recipe.items():
            unit = units.get(factor, '')
            conf = result.confidence.get(factor, 0)

            if value is None:
                lines.append(f"  {factor}: NOT NEEDED")
            else:
                lines.append(f"  {factor}: {value:.2f} {unit} (conf: {conf:.0%})")

        if result.warnings:
            lines.extend(["-" * 50, "Warnings:"])
            for w in result.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 50)

        return '\n'.join(lines)

    def demo_predict(self, cancer_type: str) -> PredictionResult:
        """
        Quick demo prediction for a cancer type.

        Args:
            cancer_type: e.g., 'Breast', 'Lung', 'Colon'

        Returns:
            PredictionResult
        """
        sample = {
            'primary_site': cancer_type,
            'gender': 'Unknown',
            'tissue_status': 'Tumor',
            'disease_status': 'Initial CNS Tumor',
        }
        return self.predict_single(sample)


def quick_predict(cancer_type: str, model_dir: str = 'beta_output') -> Dict:
    """
    Quick prediction function for demos.

    Args:
        cancer_type: Cancer type (e.g., 'Breast')
        model_dir: Path to model directory

    Returns:
        Recipe dictionary
    """
    api = BetaAPI.load(model_dir)
    result = api.demo_predict(cancer_type)

    if result.success:
        return result.recipe
    else:
        raise ValueError(result.error)


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m beta.api <cancer_type>")
        print("Example: python -m beta.api Breast")
        sys.exit(1)

    cancer_type = sys.argv[1]

    try:
        api = BetaAPI.load('beta_output')
        result = api.demo_predict(cancer_type)
        print(api.get_recipe_string(result))
    except FileNotFoundError:
        print("ERROR: Model not found. Run 'python -m beta.train' first.")
        sys.exit(1)
