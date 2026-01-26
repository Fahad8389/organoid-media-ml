"""
Beta Preprocessing Pipeline
Simplified preprocessing for demo purposes.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib


# Feature definitions
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

# Target media factors
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

# Columns to exclude
EXCLUDE_COLUMNS = [
    'model_name', 'case_id', 'link_model_details', 'link_sequencing_data',
    'link_maf', 'link_proteomics', 'date_created', 'date_of_availability',
    'molecular_characterizations', 'distributor', 'licensing_required',
    'has_multiple_models', 'neoadjuvant_therapy', 'chemo_drug_list_available',
    'therapy', 'clinical_tumor_diagnosis', 'histological_subtype',
    'acquisition_site', 'race', 'tnm_stage', 'clinical_stage_grouping',
    'expansion_status', 'mutated_genes_1', 'mutated_genes_2',
    'research_somatic_variants', 'clinical_variants', 'histopath_biomarkers',
    'split_ratio', 'doubling_time', 'time_to_split', 'formulation_number',
    'has_tp53_mutation', 'has_kras_mutation', 'has_apc_mutation',
    'mutation_count_top10', 'data_completeness_score', 'has_sequencing_data',
    'noggin', 'gastrin', 'nicotinamide', 'b27', 'n2', 'fgf7', 'fgf10',
    'rspondin', 'wnt3a', 'heparin', 'hydrocortisone', 'prostaglandin_e2',
    'primocin', 'forskolin', 'heregulin', 'neuregulin',
]


class BetaPreprocessor:
    """
    Simplified preprocessing for beta demo.

    Pipeline:
    1. Split features/targets
    2. Encode categorical (OneHot)
    3. Scale numerical (StandardScaler)
    4. Handle VAF NULL values
    5. Normalize targets
    """

    def __init__(self):
        self.encoder: Optional[ColumnTransformer] = None
        self.factor_metadata: Dict[str, Dict] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'BetaPreprocessor':
        """Fit preprocessing pipeline."""
        X, y = self._split_features_targets(df)

        # Analyze targets for metadata
        self._analyze_targets(y)

        # Identify columns
        cat_cols = [c for c in CLINICAL_CATEGORICAL if c in X.columns]
        num_cols = [c for c in CLINICAL_NUMERICAL if c in X.columns]

        # Build encoder
        transformers = []
        if cat_cols:
            transformers.append((
                'cat',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                cat_cols
            ))
        if num_cols:
            transformers.append((
                'num',
                StandardScaler(),
                num_cols
            ))

        if transformers:
            self.encoder = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )

            # Prepare data for fitting
            X_fit = X[cat_cols + num_cols].copy()
            for col in cat_cols:
                X_fit[col] = X_fit[col].fillna('Unknown')
            for col in num_cols:
                X_fit[col] = pd.to_numeric(X_fit[col], errors='coerce').fillna(0)

            self.encoder.fit(X_fit)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data."""
        if not self.fitted:
            raise ValueError("Call fit() first")

        X, y = self._split_features_targets(df)

        # Transform targets (normalize continuous, keep binary as-is)
        y_transformed = self._transform_targets(y)

        # Identify columns
        cat_cols = [c for c in CLINICAL_CATEGORICAL if c in X.columns]
        num_cols = [c for c in CLINICAL_NUMERICAL if c in X.columns]

        if self.encoder and (cat_cols or num_cols):
            # Prepare data
            X_enc = X[cat_cols + num_cols].copy()
            for col in cat_cols:
                X_enc[col] = X_enc[col].fillna('Unknown')
            for col in num_cols:
                X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce').fillna(0)

            # Transform
            X_encoded = self.encoder.transform(X_enc)
            feature_names = self.encoder.get_feature_names_out()

            X_enc_df = pd.DataFrame(
                X_encoded,
                index=X.index,
                columns=feature_names
            )

            # Get other columns (VAF, etc.)
            other_cols = [c for c in X.columns if c not in cat_cols + num_cols]
            X_other = X[other_cols].fillna(0)

            # Handle VAF columns - add NULL indicators
            vaf_cols = [c for c in X_other.columns if c.endswith('_vaf')]
            for col in vaf_cols:
                null_col = f"{col}_is_null"
                X_other[null_col] = X[col].isna().astype(int)

            X_transformed = pd.concat([X_enc_df, X_other], axis=1)
        else:
            X_transformed = X.fillna(0)

        return X_transformed, y_transformed

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit and transform."""
        return self.fit(df).transform(df)

    def _split_features_targets(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into features and targets."""
        target_cols = [c for c in MEDIA_FACTORS if c in df.columns]
        self.target_columns = target_cols

        exclude = set(EXCLUDE_COLUMNS + target_cols)
        feature_cols = [c for c in df.columns if c not in exclude]
        self.feature_columns = feature_cols

        return df[feature_cols].copy(), df[target_cols].copy()

    def _extract_numeric(self, series: pd.Series) -> pd.Series:
        """Extract numeric values from strings like '50 ng/mL'."""
        def extract(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            # Extract first number from string
            import re
            match = re.search(r'[\d.]+', str(val))
            if match:
                return float(match.group())
            return np.nan
        return series.apply(extract)

    def _analyze_targets(self, y: pd.DataFrame):
        """Analyze targets to determine model type."""
        for col in y.columns:
            # Extract numeric values first
            numeric_col = self._extract_numeric(y[col])
            non_null = numeric_col.dropna()
            n_samples = len(non_null)
            unique_vals = non_null.unique()
            n_unique = len(unique_vals)

            # Determine model type
            if n_samples < 10:
                model_type = "constant"
                constant_val = non_null.median() if n_samples > 0 else 0
            elif n_unique <= 2:
                model_type = "binary_classifier"
                constant_val = unique_vals[0] if n_unique == 1 else non_null.mode().iloc[0]
            else:
                model_type = "regression"
                constant_val = None

            self.factor_metadata[col] = {
                'model_type': model_type,
                'n_samples': n_samples,
                'n_unique_values': n_unique,
                'constant_value': constant_val,
                'unit': self._get_unit(col),
                'mean': non_null.mean() if n_samples > 0 else 0,
                'std': non_null.std() if n_samples > 1 else 1,
            }

    def _transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Normalize targets."""
        y_transformed = y.copy()

        for col in y.columns:
            # First extract numeric values
            y_transformed[col] = self._extract_numeric(y[col])

            meta = self.factor_metadata.get(col, {})
            if meta.get('model_type') == 'regression':
                mean = meta.get('mean', 0)
                std = meta.get('std', 1)
                if std > 0:
                    y_transformed[col] = (y_transformed[col] - mean) / std

        return y_transformed

    def inverse_transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Convert predictions back to original scale."""
        y_inv = y.copy()

        for col in y.columns:
            meta = self.factor_metadata.get(col, {})
            if meta.get('model_type') == 'regression':
                mean = meta.get('mean', 0)
                std = meta.get('std', 1)
                y_inv[col] = y[col] * std + mean

        return y_inv

    def _get_unit(self, factor: str) -> str:
        """Get unit for factor."""
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
        return units.get(factor, '')

    def save(self, path: str):
        """Save preprocessor."""
        joblib.dump({
            'encoder': self.encoder,
            'factor_metadata': self.factor_metadata,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'fitted': self.fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BetaPreprocessor':
        """Load preprocessor."""
        data = joblib.load(path)
        prep = cls()
        prep.encoder = data['encoder']
        prep.factor_metadata = data['factor_metadata']
        prep.feature_columns = data['feature_columns']
        prep.target_columns = data['target_columns']
        prep.fitted = data['fitted']
        return prep


def create_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: str = 'primary_site',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create 80/20 train/test split.

    Args:
        df: Full dataset
        test_size: Test fraction (default 0.2 = 20%)
        stratify_col: Column for stratification
        random_state: Random seed

    Returns:
        (train_df, test_df)
    """
    if stratify_col in df.columns:
        # Group rare categories
        strat = df[stratify_col].copy()
        counts = strat.value_counts()
        rare = counts[counts < 5].index
        strat = strat.replace(rare, 'Other')

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=strat,
            random_state=random_state
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

    return train_df, test_df
