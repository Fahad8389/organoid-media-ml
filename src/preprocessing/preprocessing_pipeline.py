"""
Full Preprocessing Pipeline
Data Analyst Tasks: D3, D4, D5
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from .factor_normalizer import FactorNormalizer
from .vaf_preprocessor import VAFPreprocessor, add_pathway_features


# Define feature categories
CLINICAL_CATEGORICAL = [
    'primary_site',
    'gender',
    'tissue_status',
    'disease_status',
    'vital_status',
    'histological_grade',
    'model_type',
]

CLINICAL_NUMERICAL = [
    'age_at_acquisition_years',
    'age_at_diagnosis_years',
]

# Media factor columns (targets)
MEDIA_FACTORS = [
    'egf',
    'y27632',
    'n_acetyl_cysteine',
    'a83_01',
    'sb202190',
    'fgf2',
    'cholera_toxin',
    'insulin',
]

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    'model_name',
    'case_id',
    'link_model_details',
    'link_sequencing_data',
    'link_maf',
    'link_proteomics',
    'date_created',
    'date_of_availability',
    'molecular_characterizations',
    'distributor',
    'licensing_required',
    'has_multiple_models',
    'neoadjuvant_therapy',
    'chemo_drug_list_available',
    'therapy',
    'clinical_tumor_diagnosis',
    'histological_subtype',
    'acquisition_site',
    'race',
    'tnm_stage',
    'clinical_stage_grouping',
    'expansion_status',
    'mutated_genes_1',
    'mutated_genes_2',
    'research_somatic_variants',
    'clinical_variants',
    'histopath_biomarkers',
    'split_ratio',
    'doubling_time',
    'time_to_split',
    'formulation_number',  # Exclude - we're learning chemistry not formulations
    # Also exclude computed fields from master_dataset_v2
    'has_tp53_mutation',
    'has_kras_mutation',
    'has_apc_mutation',
    'mutation_count_top10',
    'data_completeness_score',
    'has_sequencing_data',
    # Factors with no data
    'noggin', 'gastrin', 'nicotinamide', 'b27', 'n2', 'fgf7', 'fgf10',
    'rspondin', 'wnt3a', 'heparin', 'hydrocortisone', 'prostaglandin_e2',
    'primocin', 'forskolin', 'heregulin', 'neuregulin',
]


class MediaRecipePreprocessor:
    """
    Complete preprocessing pipeline for Media Recipe Generator.

    Handles:
    - Clinical categorical encoding (OneHot)
    - Clinical numerical scaling (StandardScaler)
    - VAF NULL handling (indicator encoding)
    - Pathway feature engineering
    - Target normalization
    """

    def __init__(self):
        self.clinical_encoder: Optional[ColumnTransformer] = None
        self.vaf_preprocessor: Optional[VAFPreprocessor] = None
        self.target_normalizer: Optional[FactorNormalizer] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'MediaRecipePreprocessor':
        """
        Fit all preprocessing components.

        Args:
            df: Full dataset with all columns

        Returns:
            self
        """
        X, y = self._split_features_targets(df)

        # Fit target normalizer
        self.target_normalizer = FactorNormalizer()
        self.target_normalizer.fit(y)

        # Fit VAF preprocessor
        vaf_cols = [col for col in X.columns if col.endswith('_vaf')]
        if vaf_cols:
            self.vaf_preprocessor = VAFPreprocessor(add_aggregate_features=True)
            self.vaf_preprocessor.fit(X)

        # Identify clinical columns present in data
        cat_cols = [c for c in CLINICAL_CATEGORICAL if c in X.columns]
        num_cols = [c for c in CLINICAL_NUMERICAL if c in X.columns]

        # Build clinical encoder
        transformers = []
        if cat_cols:
            transformers.append((
                'categorical',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                cat_cols
            ))
        if num_cols:
            transformers.append((
                'numerical',
                StandardScaler(),
                num_cols
            ))

        if transformers:
            self.clinical_encoder = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            # Fit on a sample to initialize
            X_clinical = X[cat_cols + num_cols].copy()
            # Handle categorical NaN
            for col in cat_cols:
                X_clinical[col] = X_clinical[col].fillna('Unknown')
            # Handle numerical columns - convert to numeric first, then fill with median
            for col in num_cols:
                numeric_col = pd.to_numeric(X_clinical[col], errors='coerce')
                median_val = numeric_col.median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback if all values are non-numeric
                X_clinical[col] = numeric_col.fillna(median_val)
            self.clinical_encoder.fit(X_clinical)

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        return_arrays: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data for model training.

        Args:
            df: Full dataset
            return_arrays: If True, return numpy arrays instead of DataFrames

        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        X, y = self._split_features_targets(df)

        # Transform targets
        y_transformed = self.target_normalizer.transform(y)

        # Transform VAF features
        if self.vaf_preprocessor is not None:
            X = self.vaf_preprocessor.transform(X)
            # Add pathway features
            X = add_pathway_features(X)

        # Transform clinical features
        cat_cols = [c for c in CLINICAL_CATEGORICAL if c in X.columns]
        num_cols = [c for c in CLINICAL_NUMERICAL if c in X.columns]

        if self.clinical_encoder is not None and (cat_cols or num_cols):
            X_clinical = X[cat_cols + num_cols].copy()
            # Handle categorical NaN
            for col in cat_cols:
                X_clinical[col] = X_clinical[col].fillna('Unknown')
            # Handle numerical columns
            for col in num_cols:
                numeric_col = pd.to_numeric(X_clinical[col], errors='coerce')
                X_clinical[col] = numeric_col.fillna(0)

            X_clinical_encoded = self.clinical_encoder.transform(X_clinical)

            # Get feature names from encoder
            feature_names = self.clinical_encoder.get_feature_names_out()

            # Create DataFrame with encoded clinical features
            X_clinical_df = pd.DataFrame(
                X_clinical_encoded,
                index=X.index,
                columns=feature_names
            )

            # Get non-clinical columns (VAF and derived)
            other_cols = [c for c in X.columns if c not in cat_cols + num_cols]
            X_other = X[other_cols].copy()

            # Combine
            X_transformed = pd.concat([X_clinical_df, X_other], axis=1)
        else:
            X_transformed = X

        if return_arrays:
            return X_transformed.values, y_transformed.values

        return X_transformed, y_transformed

    def fit_transform(
        self,
        df: pd.DataFrame,
        return_arrays: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, return_arrays=return_arrays)

    def _split_features_targets(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into features and targets."""
        # Target columns
        target_cols = [c for c in MEDIA_FACTORS if c in df.columns]
        self.target_columns = target_cols

        # Feature columns (everything except targets and excluded)
        exclude = set(EXCLUDE_COLUMNS + target_cols)
        feature_cols = [c for c in df.columns if c not in exclude]
        self.feature_columns = feature_cols

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        return X, y

    def inverse_transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Convert normalized predictions back to original scale."""
        return self.target_normalizer.inverse_transform(y)

    def format_predictions(self, y: pd.DataFrame) -> pd.DataFrame:
        """Format predictions as strings with units."""
        return self.target_normalizer.format_predictions(y)

    def save(self, filepath: str):
        """Save preprocessor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MediaRecipePreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(filepath)

    def save_metadata(self, filepath: str):
        """Save preprocessor metadata to JSON."""
        import json

        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'fitted': self.fitted,
        }

        # Add factor normalizer metadata
        if self.target_normalizer is not None:
            factor_meta = {}
            for name, meta in self.target_normalizer.factor_metadata.items():
                factor_meta[name] = {
                    'model_type': meta.model_type,
                    'n_samples': meta.n_samples,
                    'unique_values': meta.unique_values,
                    'unit': meta.unit,
                    'scaler_mean': meta.scaler_mean,
                    'scaler_std': meta.scaler_std,
                }
            metadata['factor_metadata'] = factor_meta

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_metadata(cls, filepath: str) -> dict:
        """Load preprocessor metadata from JSON."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by: str = 'primary_site',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split.

    Args:
        df: Full dataset
        test_size: Fraction for test set
        stratify_by: Column to stratify by
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    if stratify_by in df.columns:
        # Handle rare categories by grouping them
        stratify_col = df[stratify_by].copy()
        value_counts = stratify_col.value_counts()
        rare_categories = value_counts[value_counts < 5].index
        stratify_col = stratify_col.replace(rare_categories, 'Other')

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

    return train_df, test_df


def get_factor_masks(y: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Get boolean masks for non-NULL values in each factor.

    Used for masked training where each factor model trains
    only on samples with non-NULL values for that factor.

    Args:
        y: Target DataFrame (normalized numeric values)

    Returns:
        Dict mapping factor name to boolean mask Series
    """
    return {col: ~y[col].isna() for col in y.columns}


if __name__ == "__main__":
    # Test the pipeline
    import os

    # Load data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from config.paths import PROCESSED_DATA_DIR
    data_path = str(PROCESSED_DATA_DIR / "master_dataset_v2.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows")

        # Create preprocessor
        preprocessor = MediaRecipePreprocessor()

        # Fit and transform
        X, y = preprocessor.fit_transform(df)

        print(f"\nFeature shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        print("\nFeature columns (first 20):")
        print(X.columns[:20].tolist())

        print("\nTarget summary:")
        print(preprocessor.target_normalizer.get_factor_summary())

        print("\nFactor masks (non-NULL counts):")
        masks = get_factor_masks(y)
        for factor, mask in masks.items():
            print(f"  {factor}: {mask.sum()} samples")
    else:
        print(f"Data file not found: {data_path}")
