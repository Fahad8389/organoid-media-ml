"""
Main Training Script - Media Recipe Generator
Orchestrates the full ML pipeline from data loading to final metrics.

Usage:
    python train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import (
    MediaRecipePreprocessor,
    FactorNormalizer,
    VAFPreprocessor
)
from models import MediaRecipeGenerator, FactorModelConfig
from evaluation import (
    cross_validate_all_factors,
    summarize_cv_results,
    RecipeEvaluator
)
from interpretability import FeatureImportanceAnalyzer, extract_biological_insights

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_data(data_path: str) -> pd.DataFrame:
    """Load the master dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def run_preprocessing(df: pd.DataFrame, preprocessor: MediaRecipePreprocessor):
    """Run the full preprocessing pipeline."""
    print("\n" + "="*60)
    print("PHASE 1: PREPROCESSING")
    print("="*60)

    X, y = preprocessor.fit_transform(df)

    print(f"\nPreprocessing complete:")
    print(f"  Feature matrix X: {X.shape}")
    print(f"  Target matrix y: {y.shape}")
    print(f"  Feature columns: {len(X.columns)}")

    # Show target statistics
    print("\nTarget statistics:")
    for col in y.columns:
        non_null = y[col].notna().sum()
        print(f"  {col}: {non_null} non-NULL samples ({non_null/len(y)*100:.1f}%)")

    return X, y


def train_model(X: pd.DataFrame, y: pd.DataFrame, preprocessor: MediaRecipePreprocessor):
    """Train the MediaRecipeGenerator model."""
    print("\n" + "="*60)
    print("PHASE 2: MODEL TRAINING")
    print("="*60)

    generator = MediaRecipeGenerator(random_state=42)

    generator.fit(X, y, factor_metadata=preprocessor.target_normalizer.factor_metadata)

    print("\nModel training complete:")
    for factor_name, config in generator.factor_configs.items():
        model = generator.factor_models.get(factor_name)
        n_samples = config.n_samples
        print(f"  {factor_name}: {config.model_type} (n={n_samples})")

    return generator


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    generator: MediaRecipeGenerator,
    preprocessor: MediaRecipePreprocessor
):
    """Run cross-validation for all factors."""
    print("\n" + "="*60)
    print("PHASE 3: CROSS-VALIDATION")
    print("="*60)

    # Create a fresh generator for CV (unfitted)
    cv_generator = MediaRecipeGenerator(random_state=42)

    cv_results = cross_validate_all_factors(
        cv_generator, X, y,
        preprocessor.target_normalizer.factor_metadata
    )

    summary = summarize_cv_results(cv_results)

    print("\nCross-validation results:")
    print(summary.to_string())

    return cv_results, summary


def evaluate_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    generator: MediaRecipeGenerator,
    preprocessor: MediaRecipePreprocessor
):
    """Evaluate model predictions."""
    print("\n" + "="*60)
    print("PHASE 4: EVALUATION")
    print("="*60)

    # Get predictions
    y_pred = generator.predict(X)

    # Create evaluator
    evaluator = RecipeEvaluator(
        factor_metadata=preprocessor.target_normalizer.factor_metadata
    )

    # Evaluate
    factor_metrics, recipe_metrics = evaluator.evaluate_recipe(y, y_pred)

    # Generate report
    report = evaluator.generate_report(factor_metrics, recipe_metrics)
    print(report)

    return factor_metrics, recipe_metrics, evaluator


def analyze_feature_importance(
    X: pd.DataFrame,
    generator: MediaRecipeGenerator,
    output_dir: Path
):
    """Analyze and save feature importance."""
    print("\n" + "="*60)
    print("PHASE 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    analyzer = FeatureImportanceAnalyzer(
        generator=generator,
        feature_names=list(X.columns)
    )

    # Get built-in importance
    builtin_importance = analyzer.get_builtin_importance(top_n=30)
    print("\nTop 30 features by mean importance:")
    print(builtin_importance.head(15).to_string())

    # Get gene-media matrix
    gene_media_matrix = analyzer.get_gene_media_matrix(X, method='builtin')

    if not gene_media_matrix.empty:
        # Save gene-media matrix
        matrix_path = output_dir / 'gene_media_matrix.csv'
        gene_media_matrix.to_csv(matrix_path)
        print(f"\nGene-media matrix saved to {matrix_path}")
        print(f"  Shape: {gene_media_matrix.shape}")

        # Extract biological insights
        insights = extract_biological_insights(gene_media_matrix, threshold=0.03)
        print("\nTop gene drivers per factor:")
        for factor, genes in insights.items():
            if genes:
                print(f"  {factor}: {', '.join(genes[:5])}")
    else:
        print("\nNo gene-media matrix generated (no gene columns found)")

    # Try to compute SHAP values
    print("\nComputing SHAP values...")
    shap_values = analyzer.compute_shap_values(X, max_samples=50)

    if shap_values:
        print(f"  SHAP computed for {len(shap_values)} factors")
    else:
        print("  SHAP not available (install with: pip install shap)")

    return analyzer, gene_media_matrix


def save_artifacts(
    output_dir: Path,
    model_dir: Path,
    generator: MediaRecipeGenerator,
    preprocessor: MediaRecipePreprocessor,
    cv_summary: pd.DataFrame,
    factor_metrics: dict,
    recipe_metrics,
    evaluator: RecipeEvaluator
):
    """Save all model artifacts and metrics."""
    print("\n" + "="*60)
    print("PHASE 6: SAVING ARTIFACTS")
    print("="*60)

    # Save models
    model_path = model_dir / 'media_recipe_generator.joblib'
    generator.save(model_path)
    print(f"  Model saved to {model_path}")

    # Save preprocessor metadata
    metadata_path = output_dir.parent / 'data' / 'preprocessing_metadata.json'
    preprocessor.save_metadata(metadata_path)
    print(f"  Preprocessing metadata saved to {metadata_path}")

    # Save CV summary
    cv_path = output_dir / 'cv_results.csv'
    cv_summary.to_csv(cv_path)
    print(f"  CV results saved to {cv_path}")

    # Save final metrics as JSON
    metrics_path = output_dir / 'final_metrics.json'
    metrics_json = evaluator.to_json(factor_metrics, recipe_metrics)
    with open(metrics_path, 'w') as f:
        f.write(metrics_json)
    print(f"  Final metrics saved to {metrics_path}")

    print("\nAll artifacts saved successfully!")


def generate_final_report(
    output_dir: Path,
    cv_summary: pd.DataFrame,
    recipe_metrics,
    gene_media_matrix: pd.DataFrame
):
    """Generate the final markdown report."""
    print("\n" + "="*60)
    print("PHASE 7: GENERATING FINAL REPORT")
    print("="*60)

    report_lines = [
        "# Media Recipe Generator - Final Report",
        "",
        "## Overview",
        "",
        "This report summarizes the performance of the Media Recipe Generator ML system,",
        "which predicts organoid media factor concentrations from clinical and genomic profiles.",
        "",
        "## Model Architecture",
        "",
        "- **Per-factor models**: Each media factor has its own model",
        "- **Regression factors**: egf, cholera_toxin (GradientBoostingRegressor)",
        "- **Binary factors**: y27632, n_acetyl_cysteine, a83_01, sb202190, fgf2 (GradientBoostingClassifier)",
        "- **Constant factors**: insulin (insufficient data, n=5)",
        "",
        "## Cross-Validation Results",
        "",
        "```",
        cv_summary.to_string(),
        "```",
        "",
        "## Recipe-Level Metrics",
        "",
        f"- **Total Samples**: {recipe_metrics.n_samples}",
        f"- **Total Factors**: {recipe_metrics.n_factors}",
        f"- **Exact Match Rate**: {recipe_metrics.exact_match_rate:.2%}",
        f"- **Partial Match Rate**: {recipe_metrics.partial_match_rate:.2%}",
        f"- **Weighted Recipe Score**: {recipe_metrics.weighted_recipe_score:.2%}",
        "",
    ]

    if recipe_metrics.weighted_r2 is not None:
        report_lines.extend([
            "## Regression Performance",
            "",
            f"- **Weighted R²**: {recipe_metrics.weighted_r2:.4f}",
        ])
        if recipe_metrics.mean_mae is not None:
            report_lines.append(f"- **Mean MAE**: {recipe_metrics.mean_mae:.4f}")
        report_lines.append("")

    if recipe_metrics.mean_f1 is not None:
        report_lines.extend([
            "## Classification Performance",
            "",
            f"- **Mean F1 Score**: {recipe_metrics.mean_f1:.4f}",
            "",
        ])

    if not gene_media_matrix.empty:
        report_lines.extend([
            "## Gene-Media Importance Matrix",
            "",
            "Top genes driving each media factor prediction:",
            "",
            "```",
            gene_media_matrix.head(10).to_string(),
            "```",
            "",
        ])

    report_lines.extend([
        "## Files Generated",
        "",
        "- `model_artifacts/`: Trained model files (.joblib)",
        "- `outputs/final_metrics.json`: Detailed metrics in JSON format",
        "- `outputs/cv_results.csv`: Cross-validation results",
        "- `outputs/gene_media_matrix.csv`: Gene-media importance matrix",
        "- `data/preprocessing_metadata.json`: Preprocessing configuration",
        "",
        "## Usage",
        "",
        "```python",
        "from src.models import MediaRecipeGenerator",
        "from src.preprocessing import MediaRecipePreprocessor",
        "",
        "# Load models",
        "generator = MediaRecipeGenerator.load_models('model_artifacts/')",
        "preprocessor = MediaRecipePreprocessor.load_metadata('data/preprocessing_metadata.json')",
        "",
        "# Predict for new sample",
        "X_new = preprocessor.transform(new_patient_data)",
        "recipe = generator.predict(X_new)",
        "```",
        "",
        "---",
        "*Generated by Media Recipe Generator ML Pipeline*",
    ])

    report_path = output_dir.parent / 'FINAL_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Final report saved to {report_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("MEDIA RECIPE GENERATOR - TRAINING PIPELINE")
    print("="*60)

    # Setup paths
    project_dir = Path(__file__).parent
    data_path = project_dir / 'data' / 'master_dataset_v2.csv'
    output_dir = project_dir / 'outputs'
    model_dir = project_dir / 'model_artifacts'

    # Create directories
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Initialize preprocessor
    preprocessor = MediaRecipePreprocessor()

    # Run preprocessing
    X, y = run_preprocessing(df, preprocessor)

    # Train model
    generator = train_model(X, y, preprocessor)

    # Run cross-validation
    cv_results, cv_summary = run_cross_validation(X, y, generator, preprocessor)

    # Evaluate model
    factor_metrics, recipe_metrics, evaluator = evaluate_model(
        X, y, generator, preprocessor
    )

    # Analyze feature importance
    analyzer, gene_media_matrix = analyze_feature_importance(X, generator, output_dir)

    # Save artifacts
    save_artifacts(
        output_dir, model_dir, generator, preprocessor,
        cv_summary, factor_metrics, recipe_metrics, evaluator
    )

    # Generate final report
    generate_final_report(output_dir, cv_summary, recipe_metrics, gene_media_matrix)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

    return generator, preprocessor, cv_results


if __name__ == "__main__":
    generator, preprocessor, cv_results = main()
