"""
Beta Media Predictor - XGBoost-based model for event demo
Version: beta-1.0
Split: 80/20 with 5-fold CV
"""

from .model import BetaMediaPredictor
from .preprocessing import BetaPreprocessor
from .confidence import ConfidenceScorer
from .validators import InputValidator

__version__ = "beta-1.0"
__all__ = [
    "BetaMediaPredictor",
    "BetaPreprocessor",
    "ConfidenceScorer",
    "InputValidator",
]
