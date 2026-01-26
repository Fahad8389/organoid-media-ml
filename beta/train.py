"""
Beta Training Script
XGBoost model with 80/20 split and 5-fold cross-validation.

Usage:
    python -m beta.train
    python beta/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, f1_score, accuracy_score
)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

from beta.model import BetaMediaPredictor
from beta.preprocessing import BetaPreprocessor, create_split
from beta.confidence import ConfidenceScorer

warnings.filterwarnings('ignore')


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")
    return df


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    factor_metadata: dict,
    n_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run 5-fold cross-validation on training data.

    Returns:
        Dict with CV scores per factor
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({n_folds}-fold)")
    print(f"{'='*60}")

    cv_results = {}

    for factor in y_train.columns:
        meta = factor_metadata.get(factor, {})
        task_type = meta.get('model_type', 'regression')

        # Get valid samples for this factor
        mask = ~y_train[factor].isna()
        X_factor = X_train.loc[mask].fillna(0)
        y_factor = y_train.loc[mask, factor]

        if len(y_factor) < 10:
            print(f"  {factor}: SKIPPED (n={len(y_factor)} < 10)")
            cv_results[factor] = {'score': 0, 'std': 0, 'n_samples': len(y_factor)}
            continue

        # Create stratification for classification
        if task_type == 'binary_classifier':
            y_binary = (~y_train[factor].isna()).astype(int)
            y_cv = y_binary.loc[mask]
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            scoring = 'roc_auc'
        else:
            y_cv = y_factor
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
            scoring = 'r2'

        # Run CV
        try:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
                if task_type == 'binary_classifier' else n_folds

            scores = cross_val_score(model, X_factor, y_cv, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            std_score = scores.std()

            cv_results[factor] = {
                'score': mean_score,
                'std': std_score,
                'n_samples': len(y_factor),
                'task_type': task_type,
                'metric': 'AUC' if task_type == 'binary_classifier' else 'R²'
            }

            print(f"  {factor}: {cv_results[factor]['metric']} = {mean_score:.3f} (+/- {std_score:.3f}) [n={len(y_factor)}]")

        except Exception as e:
            print(f"  {factor}: ERROR - {str(e)[:50]}")
            cv_results[factor] = {'score': 0, 'std': 0, 'n_samples': len(y_factor), 'error': str(e)}

    return cv_results


def evaluate_on_test(
    model: BetaMediaPredictor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    preprocessor: BetaPreprocessor
) -> dict:
    """
    Evaluate model on held-out test set.

    Returns:
        Dict with test metrics per factor
    """
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (20% held out)")
    print(f"{'='*60}")

    # Get predictions
    y_pred = model.predict(X_test)

    test_metrics = {}

    for factor in y_test.columns:
        meta = preprocessor.factor_metadata.get(factor, {})
        task_type = meta.get('model_type', 'regression')

        # Get valid samples
        mask = ~y_test[factor].isna()
        y_true = y_test.loc[mask, factor]
        y_hat = y_pred.loc[mask, factor]

        if len(y_true) < 5:
            print(f"  {factor}: SKIPPED (n={len(y_true)} < 5)")
            continue

        if task_type == 'binary_classifier':
            # For binary, compare predicted vs actual "is needed"
            y_true_binary = mask.loc[mask].astype(int)
            y_pred_binary = (~y_pred[factor].isna()).loc[mask].astype(int)

            acc = accuracy_score(y_true_binary, y_pred_binary)
            test_metrics[factor] = {
                'accuracy': acc,
                'n_test': len(y_true),
                'task_type': task_type
            }
            print(f"  {factor}: Accuracy = {acc:.3f} [n={len(y_true)}]")

        else:  # regression
            # Inverse transform to original scale for meaningful metrics
            y_true_orig = y_true * meta.get('std', 1) + meta.get('mean', 0)
            y_hat_orig = y_hat * meta.get('std', 1) + meta.get('mean', 0)

            r2 = r2_score(y_true_orig, y_hat_orig)
            mae = mean_absolute_error(y_true_orig, y_hat_orig)
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_hat_orig))

            test_metrics[factor] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'n_test': len(y_true),
                'task_type': task_type
            }
            print(f"  {factor}: R² = {r2:.3f}, MAE = {mae:.2f} [n={len(y_true)}]")

    return test_metrics


def save_artifacts(
    output_dir: Path,
    model: BetaMediaPredictor,
    preprocessor: BetaPreprocessor,
    cv_results: dict,
    test_metrics: dict,
    train_size: int,
    test_size: int
):
    """Save all training artifacts."""
    print(f"\n{'='*60}")
    print("SAVING ARTIFACTS")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'beta_model.joblib'
    model.save(str(model_path))
    print(f"  Model: {model_path}")

    # Save preprocessor
    prep_path = output_dir / 'beta_preprocessor.joblib'
    preprocessor.save(str(prep_path))
    print(f"  Preprocessor: {prep_path}")

    # Save metrics
    metrics = {
        'version': 'beta-1.0',
        'timestamp': datetime.now().isoformat(),
        'data_split': {
            'train_size': train_size,
            'test_size': test_size,
            'split_ratio': '80/20',
            'cv_folds': 5
        },
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'factor_metadata': preprocessor.factor_metadata
    }

    metrics_path = output_dir / 'beta_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics: {metrics_path}")

    # Save summary report
    report = generate_report(cv_results, test_metrics, train_size, test_size)
    report_path = output_dir / 'BETA_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report: {report_path}")


def generate_report(
    cv_results: dict,
    test_metrics: dict,
    train_size: int,
    test_size: int
) -> str:
    """Generate markdown report."""
    lines = [
        "# Beta Model Training Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Version:** beta-1.0",
        "",
        "## Data Split",
        "",
        f"- **Training samples:** {train_size} (80%)",
        f"- **Test samples:** {test_size} (20%)",
        f"- **Cross-validation:** 5-fold on training set",
        "",
        "## Cross-Validation Results (Training Set)",
        "",
        "| Factor | Metric | Score | Std | Samples |",
        "|--------|--------|-------|-----|---------|",
    ]

    for factor, result in cv_results.items():
        if 'error' not in result:
            metric = result.get('metric', 'Score')
            score = result.get('score', 0)
            std = result.get('std', 0)
            n = result.get('n_samples', 0)
            lines.append(f"| {factor} | {metric} | {score:.3f} | {std:.3f} | {n} |")

    lines.extend([
        "",
        "## Test Set Results (Held-out 20%)",
        "",
        "| Factor | Metric | Score | Samples |",
        "|--------|--------|-------|---------|",
    ])

    for factor, result in test_metrics.items():
        task = result.get('task_type', 'regression')
        n = result.get('n_test', 0)
        if task == 'binary_classifier':
            score = result.get('accuracy', 0)
            lines.append(f"| {factor} | Accuracy | {score:.3f} | {n} |")
        else:
            r2 = result.get('r2', 0)
            lines.append(f"| {factor} | R² | {r2:.3f} | {n} |")

    lines.extend([
        "",
        "## Model Architecture",
        "",
        "- **Algorithm:** XGBoost",
        "- **Hyperparameters:**",
        "  - n_estimators: 100",
        "  - max_depth: 4",
        "  - learning_rate: 0.1",
        "  - subsample: 0.8",
        "  - colsample_bytree: 0.8",
        "",
        "## Usage",
        "",
        "```python",
        "from beta import BetaMediaPredictor, BetaPreprocessor",
        "",
        "# Load models",
        "model = BetaMediaPredictor.load('beta_output/beta_model.joblib')",
        "prep = BetaPreprocessor.load('beta_output/beta_preprocessor.joblib')",
        "",
        "# Predict",
        "X, _ = prep.transform(new_data)",
        "predictions = model.predict(X)",
        "```",
        "",
        "---",
        "*Beta Model for Event Demo*",
    ])

    return '\n'.join(lines)


def main():
    """Main training pipeline."""
    print("="*60)
    print("BETA MODEL TRAINING PIPELINE")
    print("XGBoost | 80/20 Split | 5-Fold CV")
    print("="*60)

    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_path = project_dir / 'data' / 'processed' / 'master_dataset_v2.csv'
    output_dir = project_dir / 'beta_output'

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run the main data pipeline first to generate master_dataset_v2.csv")
        sys.exit(1)

    # Load data
    df = load_data(data_path)

    # Create 80/20 split
    print(f"\nCreating 80/20 train/test split...")
    train_df, test_df = create_split(df, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Initialize preprocessor and fit on training data only
    print(f"\nFitting preprocessor on training data...")
    preprocessor = BetaPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Targets: {y_train.shape[1]}")

    # Transform test data (using fitted preprocessor)
    X_test, y_test = preprocessor.transform(test_df)

    # Run cross-validation on training set
    cv_results = run_cross_validation(
        X_train, y_train,
        preprocessor.factor_metadata,
        n_folds=5
    )

    # Train final model on full training set
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")

    model = BetaMediaPredictor(random_state=42)
    model.fit(X_train, y_train, preprocessor.factor_metadata)
    print("  Model fitted successfully")
    print(f"\n{model.summary().to_string()}")

    # Evaluate on test set
    test_metrics = evaluate_on_test(model, X_test, y_test, preprocessor)

    # Update confidence scorer with CV results
    scorer = ConfidenceScorer()
    cv_scores = {f: r['score'] for f, r in cv_results.items() if 'score' in r}
    scorer.update_from_cv(cv_scores)

    # Save everything
    save_artifacts(
        output_dir,
        model,
        preprocessor,
        cv_results,
        test_metrics,
        len(train_df),
        len(test_df)
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to: {output_dir}")

    # Print summary
    print("\n--- SUMMARY ---")
    cv_scores_list = [r['score'] for r in cv_results.values() if 'score' in r and r['score'] > 0]
    if cv_scores_list:
        print(f"Mean CV Score: {np.mean(cv_scores_list):.3f}")

    return model, preprocessor, cv_results


if __name__ == "__main__":
    model, preprocessor, cv_results = main()
