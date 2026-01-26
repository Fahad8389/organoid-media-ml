"""
Input Validators for Beta API
Validates input data before prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None


class InputValidator:
    """
    Validates input data for media recipe prediction.

    Checks:
    1. Required columns present
    2. Data types correct
    3. Values in expected ranges
    4. No critical missing data
    """

    # Required columns for prediction
    REQUIRED_COLUMNS = ['primary_site']

    # Expected categorical values
    VALID_PRIMARY_SITES = [
        'Breast', 'Lung', 'Colon', 'Pancreas', 'Ovary',
        'Stomach', 'Liver', 'Kidney', 'Prostate', 'Bladder',
        'Esophagus', 'Head and Neck', 'Skin', 'Brain',
        'Colorectal', 'Gastric', 'Other'
    ]

    VALID_GENDERS = ['Male', 'Female', 'Unknown']

    VALID_TISSUE_STATUS = ['Tumor', 'Normal', 'Metastatic', 'Unknown']

    # Numerical ranges
    AGE_RANGE = (0, 120)
    VAF_RANGE = (0, 1)

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, errors on warnings. If False, auto-fix issues.
        """
        self.strict = strict

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input DataFrame

        Returns:
            ValidationResult with status and cleaned data
        """
        errors = []
        warnings = []
        df = data.copy()

        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                cleaned_data=None
            )

        # Validate and clean primary_site
        if 'primary_site' in df.columns:
            df, site_warnings = self._validate_primary_site(df)
            warnings.extend(site_warnings)

        # Validate gender
        if 'gender' in df.columns:
            df, gender_warnings = self._validate_gender(df)
            warnings.extend(gender_warnings)

        # Validate age
        for age_col in ['age_at_acquisition_years', 'age_at_diagnosis_years']:
            if age_col in df.columns:
                df, age_warnings = self._validate_age(df, age_col)
                warnings.extend(age_warnings)

        # Validate VAF columns
        vaf_cols = [c for c in df.columns if c.endswith('_vaf')]
        for col in vaf_cols:
            df, vaf_warnings = self._validate_vaf(df, col)
            warnings.extend(vaf_warnings)

        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("Input data is empty")

        is_valid = len(errors) == 0
        if self.strict and warnings:
            is_valid = False
            errors.extend([f"Warning treated as error: {w}" for w in warnings])

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_data=df if is_valid else None
        )

    def _validate_primary_site(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and normalize primary_site."""
        warnings = []

        # Map common variations
        site_map = {
            'breast': 'Breast',
            'lung': 'Lung',
            'colon': 'Colon',
            'colorectal': 'Colorectal',
            'pancreas': 'Pancreas',
            'pancreatic': 'Pancreas',
            'ovary': 'Ovary',
            'ovarian': 'Ovary',
            'stomach': 'Stomach',
            'gastric': 'Gastric',
            'liver': 'Liver',
            'kidney': 'Kidney',
            'renal': 'Kidney',
            'prostate': 'Prostate',
            'bladder': 'Bladder',
            'esophagus': 'Esophagus',
            'head and neck': 'Head and Neck',
            'skin': 'Skin',
            'brain': 'Brain',
        }

        def normalize(val):
            if pd.isna(val):
                return 'Unknown'
            val_lower = str(val).lower().strip()
            return site_map.get(val_lower, str(val).title())

        original = df['primary_site'].copy()
        df['primary_site'] = df['primary_site'].apply(normalize)

        # Check for unknown sites
        unknown_sites = df[~df['primary_site'].isin(self.VALID_PRIMARY_SITES + ['Unknown'])]
        if len(unknown_sites) > 0:
            unique_unknown = unknown_sites['primary_site'].unique()
            warnings.append(f"Unknown primary sites mapped to 'Other': {list(unique_unknown)}")
            df.loc[~df['primary_site'].isin(self.VALID_PRIMARY_SITES), 'primary_site'] = 'Other'

        return df, warnings

    def _validate_gender(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and normalize gender."""
        warnings = []

        gender_map = {
            'male': 'Male',
            'm': 'Male',
            'female': 'Female',
            'f': 'Female',
            'unknown': 'Unknown',
            '': 'Unknown',
        }

        def normalize(val):
            if pd.isna(val):
                return 'Unknown'
            return gender_map.get(str(val).lower().strip(), 'Unknown')

        df['gender'] = df['gender'].apply(normalize)

        return df, warnings

    def _validate_age(
        self,
        df: pd.DataFrame,
        col: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate age column."""
        warnings = []

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check range
        invalid_ages = df[(df[col] < self.AGE_RANGE[0]) | (df[col] > self.AGE_RANGE[1])]
        if len(invalid_ages) > 0:
            warnings.append(f"{col}: {len(invalid_ages)} values out of range, set to NaN")
            df.loc[(df[col] < self.AGE_RANGE[0]) | (df[col] > self.AGE_RANGE[1]), col] = np.nan

        return df, warnings

    def _validate_vaf(
        self,
        df: pd.DataFrame,
        col: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate VAF column."""
        warnings = []

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check range (0-1)
        invalid_vaf = df[(df[col] < self.VAF_RANGE[0]) | (df[col] > self.VAF_RANGE[1])]
        if len(invalid_vaf) > 0:
            # VAF might be in percentage (0-100), convert
            if (df[col] > 1).any():
                warnings.append(f"{col}: Converting from percentage to fraction")
                df.loc[df[col] > 1, col] = df.loc[df[col] > 1, col] / 100

            # Clip remaining invalid
            df[col] = df[col].clip(0, 1)

        return df, warnings

    def validate_single_sample(self, sample: Dict) -> ValidationResult:
        """Validate a single sample dictionary."""
        df = pd.DataFrame([sample])
        return self.validate(df)


def quick_validate(data: pd.DataFrame) -> bool:
    """Quick validation check - returns True/False."""
    validator = InputValidator(strict=False)
    result = validator.validate(data)
    return result.is_valid
