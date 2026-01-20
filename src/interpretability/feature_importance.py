"""
Feature Importance Analysis - SHAP-based gene-media relationships
ML Coder Task: M5
Management Agent Task: G5 (Gene-Media Matrix)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance and gene-media relationships.

    Methods:
    1. Built-in feature importance (from tree models)
    2. SHAP values (if available)
    3. Permutation importance
    4. Gene-Media interaction matrix
    """

    def __init__(
        self,
        generator: Any,  # MediaRecipeGenerator
        feature_names: List[str]
    ):
        """
        Args:
            generator: Fitted MediaRecipeGenerator
            feature_names: List of feature names
        """
        self.generator = generator
        self.feature_names = feature_names
        self.shap_values: Dict[str, np.ndarray] = {}
        self.gene_columns = [f for f in feature_names if f.endswith('_vaf')]

    def get_builtin_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get built-in feature importance from all factor models.

        Returns:
            DataFrame with factors as columns, features as rows
        """
        importance_dict = {}

        for factor_name, model in self.generator.factor_models.items():
            config = self.generator.factor_configs[factor_name]

            if config.model_type == "constant":
                continue

            if hasattr(model, 'feature_importances_'):
                importance_dict[factor_name] = pd.Series(
                    model.feature_importances_,
                    index=self.feature_names
                )

        if not importance_dict:
            return pd.DataFrame()

        df = pd.DataFrame(importance_dict)

        # Sort by mean importance across factors
        df['mean_importance'] = df.mean(axis=1)
        df = df.sort_values('mean_importance', ascending=False)
        df = df.drop('mean_importance', axis=1)

        return df.head(top_n)

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        factors: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for factor models.

        Args:
            X: Feature matrix (preprocessed)
            factors: List of factors to analyze (all if None)
            max_samples: Max samples to use (for speed)

        Returns:
            Dict mapping factor name to SHAP values array
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available. Install with: pip install shap")
            return {}

        if factors is None:
            factors = list(self.generator.factor_models.keys())

        X_sample = X.fillna(0)
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=42)

        for factor_name in factors:
            config = self.generator.factor_configs[factor_name]

            if config.model_type == "constant":
                continue

            model = self.generator.factor_models[factor_name]

            try:
                # Use TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_sample.values)

                # For classifiers, get SHAP for positive class
                if config.model_type == "binary_classifier":
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]

                self.shap_values[factor_name] = shap_vals
            except Exception as e:
                warnings.warn(f"Could not compute SHAP for {factor_name}: {e}")

        return self.shap_values

    def get_gene_media_matrix(
        self,
        X: pd.DataFrame,
        method: str = 'builtin'
    ) -> pd.DataFrame:
        """
        Build gene-media importance matrix.

        Shows which genes are important for predicting each media factor.

        Args:
            X: Feature matrix (for SHAP computation if needed)
            method: 'builtin' or 'shap'

        Returns:
            DataFrame with genes as rows, factors as columns
        """
        if method == 'shap':
            if not self.shap_values:
                self.compute_shap_values(X)

            if not self.shap_values:
                warnings.warn("No SHAP values available, falling back to builtin")
                method = 'builtin'

        if method == 'builtin':
            # Get builtin importance
            all_importance = self.get_builtin_importance(top_n=len(self.feature_names))

            # Filter to gene columns only
            gene_importance = all_importance.loc[
                all_importance.index.isin(self.gene_columns)
            ]

            return gene_importance

        else:  # shap
            gene_importance_dict = {}

            for factor_name, shap_vals in self.shap_values.items():
                # Mean absolute SHAP value per feature
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)

                # Create Series and filter to genes
                importance = pd.Series(mean_abs_shap, index=self.feature_names)
                gene_importance = importance[importance.index.isin(self.gene_columns)]

                gene_importance_dict[factor_name] = gene_importance

            return pd.DataFrame(gene_importance_dict)

    def get_top_gene_drivers(
        self,
        factor_name: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top gene drivers for a specific factor.

        Args:
            factor_name: Factor to analyze
            top_n: Number of top genes to return

        Returns:
            DataFrame with gene names and importance scores
        """
        model = self.generator.factor_models.get(factor_name)
        if model is None:
            raise ValueError(f"Unknown factor: {factor_name}")

        config = self.generator.factor_configs[factor_name]
        if config.model_type == "constant":
            return pd.DataFrame()

        if not hasattr(model, 'feature_importances_'):
            return pd.DataFrame()

        importance = pd.Series(
            model.feature_importances_,
            index=self.feature_names
        )

        # Filter to gene columns
        gene_importance = importance[importance.index.isin(self.gene_columns)]
        gene_importance = gene_importance.sort_values(ascending=False)

        return gene_importance.head(top_n).to_frame(name='importance')

    def plot_feature_importance(
        self,
        factor_name: str,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot feature importance for a factor.

        Args:
            factor_name: Factor to plot
            top_n: Number of top features
            figsize: Figure size

        Returns:
            matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available")
            return None

        model = self.generator.factor_models.get(factor_name)
        if model is None or not hasattr(model, 'feature_importances_'):
            return None

        importance = pd.Series(
            model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=True)

        top_features = importance.tail(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        top_features.plot(kind='barh', ax=ax)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance: {factor_name}')
        plt.tight_layout()

        return fig

    def plot_shap_summary(
        self,
        factor_name: str,
        X: pd.DataFrame,
        max_display: int = 20
    ):
        """
        Generate SHAP summary plot for a factor.

        Args:
            factor_name: Factor to plot
            X: Feature matrix
            max_display: Max features to display

        Returns:
            matplotlib Figure or None
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            warnings.warn("SHAP or Matplotlib not available")
            return None

        if factor_name not in self.shap_values:
            self.compute_shap_values(X, factors=[factor_name])

        if factor_name not in self.shap_values:
            return None

        shap_vals = self.shap_values[factor_name]

        # Get corresponding X data
        X_sample = X.fillna(0)
        if len(X_sample) > len(shap_vals):
            X_sample = X_sample.iloc[:len(shap_vals)]

        plt.figure()
        shap.summary_plot(
            shap_vals,
            X_sample.values,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Summary: {factor_name}')
        fig = plt.gcf()
        plt.tight_layout()

        return fig

    def save_gene_media_matrix(
        self,
        filepath: str,
        X: pd.DataFrame,
        method: str = 'builtin'
    ):
        """Save gene-media importance matrix to CSV."""
        matrix = self.get_gene_media_matrix(X, method=method)
        matrix.to_csv(filepath)
        print(f"Gene-media matrix saved to {filepath}")
        return matrix


def extract_biological_insights(
    gene_media_matrix: pd.DataFrame,
    threshold: float = 0.05
) -> Dict[str, List[str]]:
    """
    Extract biological insights from gene-media matrix.

    Args:
        gene_media_matrix: Importance matrix (genes × factors)
        threshold: Minimum importance to consider significant

    Returns:
        Dict mapping each factor to its top driver genes
    """
    insights = {}

    for factor in gene_media_matrix.columns:
        factor_importance = gene_media_matrix[factor].sort_values(ascending=False)
        significant_genes = factor_importance[factor_importance > threshold]

        # Clean gene names (remove _vaf suffix)
        gene_names = [g.replace('_vaf', '') for g in significant_genes.index]

        insights[factor] = gene_names

    return insights


if __name__ == "__main__":
    print("Feature Importance Analysis module loaded successfully")
    print(f"\nSHAP available: {SHAP_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print("\nAnalysis methods:")
    print("  1. Built-in feature importance (from GradientBoosting)")
    print("  2. SHAP values (TreeSHAP for tree models)")
    print("  3. Gene-Media importance matrix")
