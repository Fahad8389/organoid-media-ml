"""
Cross-Validation Framework
ML Coder Tasks: M3, M4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from dataclasses import dataclass
import warnings

# Note: sys.path manipulation removed - package imports work via proper installation


@dataclass
class FactorCVResults:
    """Cross-validation results for a single factor."""
    factor_name: str
    model_type: str
    n_samples: int
    cv_method: str
    n_folds: int
    # Regression metrics
    mae_scores: Optional[List[float]] = None
    rmse_scores: Optional[List[float]] = None
    r2_scores: Optional[List[float]] = None
    # Classification metrics
    accuracy_scores: Optional[List[float]] = None
    f1_scores: Optional[List[float]] = None
    auc_scores: Optional[List[float]] = None

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {
            'factor': self.factor_name,
            'model_type': self.model_type,
            'n_samples': self.n_samples,
            'cv_method': self.cv_method,
            'n_folds': self.n_folds
        }

        if self.mae_scores:
            summary['mae_mean'] = np.mean(self.mae_scores)
            summary['mae_std'] = np.std(self.mae_scores)
        if self.rmse_scores:
            summary['rmse_mean'] = np.mean(self.rmse_scores)
            summary['rmse_std'] = np.std(self.rmse_scores)
        if self.r2_scores:
            summary['r2_mean'] = np.mean(self.r2_scores)
            summary['r2_std'] = np.std(self.r2_scores)
        if self.accuracy_scores:
            summary['accuracy_mean'] = np.mean(self.accuracy_scores)
            summary['accuracy_std'] = np.std(self.accuracy_scores)
        if self.f1_scores:
            summary['f1_mean'] = np.mean(self.f1_scores)
            summary['f1_std'] = np.std(self.f1_scores)
        if self.auc_scores:
            summary['auc_mean'] = np.mean(self.auc_scores)
            summary['auc_std'] = np.std(self.auc_scores)

        return summary


def get_cv_strategy(n_samples: int, stratify: bool = False):
    """
    Determine appropriate CV strategy based on sample size.

    Args:
        n_samples: Number of samples available
        stratify: Whether to use stratified splits

    Returns:
        CV splitter object and description
    """
    if n_samples < 20:
        return LeaveOneOut(), "LOOCV", n_samples
    elif n_samples < 50:
        n_folds = 3
        if stratify:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42), "3-fold Stratified", n_folds
        return KFold(n_splits=n_folds, shuffle=True, random_state=42), "3-fold", n_folds
    else:
        n_folds = 5
        if stratify:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42), "5-fold Stratified", n_folds
        return KFold(n_splits=n_folds, shuffle=True, random_state=42), "5-fold", n_folds


def cross_validate_factor(
    model,
    X: np.ndarray,
    y: np.ndarray,
    factor_name: str,
    model_type: str,
    cv=None
) -> FactorCVResults:
    """
    Perform cross-validation for a single factor model.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target vector
        factor_name: Name of the factor
        model_type: "regression" or "binary_classifier"
        cv: Cross-validation splitter (auto-determined if None)

    Returns:
        FactorCVResults with all scores
    """
    n_samples = len(y)

    if cv is None:
        cv, cv_method, n_folds = get_cv_strategy(
            n_samples,
            stratify=(model_type == "binary_classifier")
        )
    else:
        cv_method = "custom"
        n_folds = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else "?"

    results = FactorCVResults(
        factor_name=factor_name,
        model_type=model_type,
        n_samples=n_samples,
        cv_method=cv_method,
        n_folds=n_folds
    )

    if model_type == "regression":
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_scores.append(r2_score(y_test, y_pred))

        results.mae_scores = mae_scores
        results.rmse_scores = rmse_scores
        results.r2_scores = r2_scores

    elif model_type == "binary_classifier":
        accuracy_scores = []
        f1_scores_list = []
        auc_scores = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores_list.append(f1_score(y_test, y_pred, zero_division=0))

            # AUC requires probability predictions
            if hasattr(model_clone, 'predict_proba'):
                try:
                    y_proba = model_clone.predict_proba(X_test)[:, 1]
                    if len(np.unique(y_test)) > 1:
                        auc_scores.append(roc_auc_score(y_test, y_proba))
                except:
                    pass

        results.accuracy_scores = accuracy_scores
        results.f1_scores = f1_scores_list
        results.auc_scores = auc_scores if auc_scores else None

    return results


def cross_validate_all_factors(
    generator: Any,  # MediaRecipeGenerator
    X: pd.DataFrame,
    y: pd.DataFrame,
    factor_metadata: Dict
) -> Dict[str, FactorCVResults]:
    """
    Cross-validate all factor models.

    Args:
        generator: MediaRecipeGenerator (unfitted, used as template)
        X: Full feature matrix
        y: Full target matrix (with NaN for missing)
        factor_metadata: Metadata from FactorNormalizer

    Returns:
        Dict mapping factor name to CV results
    """
    results = {}
    X_array = X.fillna(0).values

    for factor_name in y.columns:
        meta = factor_metadata.get(factor_name)
        if meta is None:
            continue

        if meta.model_type == "constant":
            # No CV for constant predictor
            results[factor_name] = FactorCVResults(
                factor_name=factor_name,
                model_type="constant",
                n_samples=meta.n_samples,
                cv_method="N/A",
                n_folds=0
            )
            continue

        # Get non-NULL mask
        if meta.model_type == "binary_classifier":
            # For binary classifier, use all samples
            mask = np.ones(len(y), dtype=bool)
            y_factor = (~y[factor_name].isna()).astype(int).values
        else:
            # For regression, use only non-NULL samples
            mask = ~y[factor_name].isna()
            y_factor = y.loc[mask, factor_name].values

        X_factor = X_array[mask.values] if isinstance(mask, pd.Series) else X_array[mask]

        if len(y_factor) < 3:
            warnings.warn(f"Insufficient samples for CV on {factor_name}")
            continue

        # Create model for CV
        model = generator._create_model(
            generator.factor_configs.get(factor_name) or
            type('Config', (), {
                'model_type': meta.model_type,
                'n_samples': meta.n_samples
            })()
        )

        # Run CV
        results[factor_name] = cross_validate_factor(
            model=model,
            X=X_factor,
            y=y_factor,
            factor_name=factor_name,
            model_type=meta.model_type
        )

    return results


def summarize_cv_results(results: Dict[str, FactorCVResults]) -> pd.DataFrame:
    """Convert CV results to summary DataFrame."""
    rows = [r.get_summary() for r in results.values()]
    return pd.DataFrame(rows)


# Hyperparameter tuning
REGRESSION_PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [3, 5, 10],
    'learning_rate': [0.05, 0.1, 0.2]
}

CLASSIFIER_PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [2, 3, 4],
    'min_samples_leaf': [5, 10],
    'learning_rate': [0.1, 0.2]
}


def tune_hyperparameters(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    cv=None,
    n_jobs: int = -1
) -> Tuple[Any, Dict, float]:
    """
    Tune hyperparameters using GridSearchCV.

    Args:
        model: Base model to tune
        X: Feature matrix
        y: Target vector
        model_type: "regression" or "binary_classifier"
        cv: CV splitter
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    if cv is None:
        cv, _, _ = get_cv_strategy(len(y), stratify=(model_type == "binary_classifier"))

    param_grid = REGRESSION_PARAM_GRID if model_type == "regression" else CLASSIFIER_PARAM_GRID
    scoring = 'neg_mean_squared_error' if model_type == "regression" else 'f1'

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


if __name__ == "__main__":
    print("Cross-validation module loaded successfully")
    print("\nCV strategies by sample size:")
    print("  n < 20: Leave-One-Out CV")
    print("  20 <= n < 50: 3-fold CV")
    print("  n >= 50: 5-fold CV")
