"""
VAF Preprocessor - Handle NULL values in genomic VAF features
Bioinformatician Tasks: B3, B4 (feature engineering)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class VAFPreprocessor:
    """
    Preprocessor for Variant Allele Frequency (VAF) genomic features.

    Strategy for NULL handling:
    - Indicator encoding: For each VAF column, create a companion "is_sequenced" flag
    - NULL values are imputed to 0 (wild-type assumption for non-sequenced)
    - This allows the model to distinguish between:
      - Wild-type (sequenced, VAF=0)
      - Mutated (sequenced, VAF>0)
      - Unknown (not sequenced, flag=0)
    """

    def __init__(self, add_aggregate_features: bool = True):
        """
        Args:
            add_aggregate_features: Whether to add mutation_count, tumor_burden, etc.
        """
        self.vaf_columns: List[str] = []
        self.add_aggregate_features = add_aggregate_features
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> 'VAFPreprocessor':
        """
        Identify VAF columns in the dataframe.

        Args:
            X: Feature DataFrame

        Returns:
            self
        """
        # Find all columns ending with '_vaf'
        self.vaf_columns = [col for col in X.columns if col.endswith('_vaf')]
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform VAF columns with indicator encoding.

        For each VAF column:
        1. Impute NULL to 0 (wild-type assumption)
        2. Create binary indicator column (_is_sequenced)

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with transformed VAF columns and indicators
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        result = X.copy()

        # Process each VAF column
        for vaf_col in self.vaf_columns:
            if vaf_col not in result.columns:
                continue

            # Create indicator: 1 if sequenced (non-NULL), 0 if not
            indicator_col = vaf_col.replace('_vaf', '_is_sequenced')
            result[indicator_col] = (~result[vaf_col].isna()).astype(int)

            # Impute NULL to 0 (wild-type assumption)
            result[vaf_col] = result[vaf_col].fillna(0)

        # Add aggregate features if requested
        if self.add_aggregate_features:
            result = self._add_aggregate_features(result)

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _add_aggregate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived aggregate features from VAF data."""

        # Mutation count: number of genes with VAF > 0
        vaf_cols = [col for col in X.columns if col.endswith('_vaf')]
        if vaf_cols:
            X['mutation_count'] = (X[vaf_cols] > 0).sum(axis=1)

            # Check if we have sequencing indicator columns
            indicator_cols = [col for col in X.columns if col.endswith('_is_sequenced')]
            if indicator_cols:
                # Has sequencing data: at least one gene was sequenced
                X['has_sequencing_data'] = (X[indicator_cols].sum(axis=1) > 0).astype(int)

                # Tumor mutation burden: mean VAF among mutated genes (only for sequenced)
                def calc_tmb(row):
                    vaf_values = row[vaf_cols]
                    mutated = vaf_values[vaf_values > 0]
                    return mutated.mean() if len(mutated) > 0 else 0

                X['tumor_mutation_burden'] = X.apply(calc_tmb, axis=1)

        return X

    def get_feature_names(self, include_indicators: bool = True) -> List[str]:
        """Get list of output feature names."""
        names = list(self.vaf_columns)
        if include_indicators:
            names.extend([col.replace('_vaf', '_is_sequenced') for col in self.vaf_columns])
        if self.add_aggregate_features:
            names.extend(['mutation_count', 'has_sequencing_data', 'tumor_mutation_burden'])
        return names


# Pathway aggregation features (Bioinformatician Task B4)
GENE_PATHWAYS = {
    'p53_pathway': ['TP53', 'MDM2', 'MDM4', 'CDKN2A', 'CDKN2B', 'ATM', 'ATR', 'CHEK1', 'CHEK2'],
    'wnt_pathway': ['APC', 'CTNNB1', 'AXIN1', 'AXIN2', 'GSK3B', 'TCF7L2', 'LEF1'],
    'ras_pathway': ['KRAS', 'NRAS', 'HRAS', 'BRAF', 'NF1', 'RAF1', 'MAP2K1', 'MAPK1'],
    'pi3k_pathway': ['PIK3CA', 'PIK3CB', 'PIK3R1', 'PTEN', 'AKT1', 'AKT2', 'MTOR'],
    'tgfb_pathway': ['SMAD2', 'SMAD3', 'SMAD4', 'TGFBR1', 'TGFBR2', 'ACVR2A'],
    'chromatin_remodeling': ['ARID1A', 'ARID1B', 'ARID2', 'SMARCA4', 'SMARCB1', 'KMT2D', 'KMT2C'],
}


def add_pathway_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add pathway activity scores based on mutation presence.

    For each pathway, calculate:
    - pathway_has_mutation: 1 if any gene in pathway is mutated
    - pathway_mutation_count: count of mutated genes in pathway
    - pathway_max_vaf: maximum VAF among pathway genes

    Args:
        X: DataFrame with VAF columns (gene_vaf format)

    Returns:
        DataFrame with additional pathway columns
    """
    result = X.copy()

    for pathway_name, genes in GENE_PATHWAYS.items():
        # Find VAF columns for pathway genes
        pathway_vaf_cols = [f"{gene}_vaf" for gene in genes if f"{gene}_vaf" in X.columns]

        if not pathway_vaf_cols:
            continue

        # Has any mutation in pathway
        result[f'{pathway_name}_has_mutation'] = (X[pathway_vaf_cols] > 0).any(axis=1).astype(int)

        # Count of mutations in pathway
        result[f'{pathway_name}_mutation_count'] = (X[pathway_vaf_cols] > 0).sum(axis=1)

        # Max VAF in pathway
        result[f'{pathway_name}_max_vaf'] = X[pathway_vaf_cols].max(axis=1)

    return result


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame({
        'TP53_vaf': [0.5, 0.0, None, 0.3, 0.0],
        'KRAS_vaf': [0.4, 0.0, None, 0.0, 0.6],
        'APC_vaf': [0.0, 0.3, None, 0.0, 0.0],
        'other_feature': [1, 2, 3, 4, 5]
    })

    print("Original Data:")
    print(test_data)

    preprocessor = VAFPreprocessor(add_aggregate_features=True)
    transformed = preprocessor.fit_transform(test_data)

    print("\nTransformed Data:")
    print(transformed)

    # Add pathway features
    with_pathways = add_pathway_features(transformed)
    print("\nWith Pathway Features:")
    print(with_pathways.columns.tolist())
