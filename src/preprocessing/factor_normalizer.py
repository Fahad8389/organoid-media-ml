"""
Factor Normalizer - Unit parsing and per-factor normalization
Bioinformatician Tasks: B1, B2
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, asdict


@dataclass
class FactorMetadata:
    """Metadata for a single media factor."""
    factor_name: str
    unit: str
    n_samples: int
    n_unique_values: int
    unique_values: list
    constant_value: Optional[float]
    model_type: str  # "regression", "binary_classifier", "constant"
    scaler_mean: Optional[float] = None
    scaler_std: Optional[float] = None


def parse_concentration(value: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a concentration string into numeric value and unit.

    Examples:
        "10 ng/mL" -> (10.0, "ng/mL")
        "500 nM" -> (500.0, "nM")
        "1 mM" -> (1.0, "mM")
        "10 uM" -> (10.0, "uM")
        "20 ug/mL" -> (20.0, "ug/mL")

    Returns:
        Tuple of (numeric_value, unit) or (None, None) if parsing fails
    """
    if pd.isna(value) or value is None or str(value).strip() == '':
        return None, None

    value_str = str(value).strip()

    # Pattern to match: number (optional decimal) followed by unit
    pattern = r'^(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|uM|µM|um|μM|nM|mM|ug/mL|µg/mL)$'
    match = re.match(pattern, value_str, re.IGNORECASE)

    if match:
        numeric = float(match.group(1))
        unit = match.group(2)
        # Normalize unit spelling
        unit = unit.replace('µM', 'uM').replace('μM', 'uM').replace('um', 'uM')
        unit = unit.replace('ng/ml', 'ng/mL').replace('µg/mL', 'ug/mL')
        return numeric, unit

    return None, None


class FactorNormalizer:
    """
    Handles unit parsing and per-factor normalization for media factors.

    Strategy:
    - Parse concentration strings to numeric values
    - Apply Z-score normalization within each factor (preserve biological scale)
    - Store metadata for inverse transform
    """

    def __init__(self):
        self.factor_metadata: Dict[str, FactorMetadata] = {}
        self.scalers: Dict[str, Optional[StandardScaler]] = {}

    def fit(self, y: pd.DataFrame) -> 'FactorNormalizer':
        """
        Analyze and fit normalizers for all media factors.

        Args:
            y: DataFrame with media factor columns (string concentrations)

        Returns:
            self
        """
        for factor_name in y.columns:
            self._fit_factor(factor_name, y[factor_name])

        return self

    def _fit_factor(self, factor_name: str, series: pd.Series):
        """Fit normalizer for a single factor."""
        # Parse all values
        parsed = series.apply(lambda x: parse_concentration(x)[0])
        non_null = parsed.dropna()

        # Determine factor characteristics
        n_samples = len(non_null)
        unique_values = sorted(non_null.unique().tolist()) if n_samples > 0 else []
        n_unique = len(unique_values)

        # Extract unit from first non-null value
        for val in series:
            _, unit = parse_concentration(val)
            if unit is not None:
                break
        else:
            unit = "unknown"

        # Determine model type
        if n_samples < 10:
            model_type = "constant"
        elif n_unique == 1:
            model_type = "binary_classifier"
        else:
            model_type = "regression"

        # Create scaler for regression factors
        scaler = None
        scaler_mean = None
        scaler_std = None

        if model_type == "regression" and n_samples > 1:
            scaler = StandardScaler()
            scaler.fit(non_null.values.reshape(-1, 1))
            scaler_mean = float(scaler.mean_[0])
            scaler_std = float(scaler.scale_[0])

        # Store metadata
        self.factor_metadata[factor_name] = FactorMetadata(
            factor_name=factor_name,
            unit=unit,
            n_samples=n_samples,
            n_unique_values=n_unique,
            unique_values=unique_values,
            constant_value=unique_values[0] if n_unique >= 1 else None,
            model_type=model_type,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std
        )
        self.scalers[factor_name] = scaler

    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Transform media factor columns to normalized numeric values.

        Args:
            y: DataFrame with media factor columns (string concentrations)

        Returns:
            DataFrame with normalized numeric values (NaN preserved)
        """
        result = pd.DataFrame(index=y.index)

        for factor_name in y.columns:
            if factor_name not in self.factor_metadata:
                raise ValueError(f"Factor {factor_name} not fitted. Call fit() first.")

            # Parse to numeric
            parsed = y[factor_name].apply(lambda x: parse_concentration(x)[0])

            # Apply scaling if regression factor
            meta = self.factor_metadata[factor_name]
            if meta.model_type == "regression" and self.scalers[factor_name] is not None:
                # Scale non-null values
                mask = ~parsed.isna()
                scaled = parsed.copy()
                if mask.any():
                    scaled.loc[mask] = self.scalers[factor_name].transform(
                        parsed.loc[mask].values.reshape(-1, 1)
                    ).flatten()
                result[factor_name] = scaled
            else:
                result[factor_name] = parsed

        return result

    def fit_transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Convert normalized predictions back to original scale.

        Args:
            y: DataFrame with normalized numeric predictions

        Returns:
            DataFrame with original scale numeric values
        """
        result = pd.DataFrame(index=y.index)

        for factor_name in y.columns:
            if factor_name not in self.factor_metadata:
                raise ValueError(f"Factor {factor_name} not fitted.")

            meta = self.factor_metadata[factor_name]
            values = y[factor_name].copy()

            # Inverse scale if regression factor
            if meta.model_type == "regression" and self.scalers[factor_name] is not None:
                mask = ~values.isna()
                if mask.any():
                    values.loc[mask] = self.scalers[factor_name].inverse_transform(
                        values.loc[mask].values.reshape(-1, 1)
                    ).flatten()

            result[factor_name] = values

        return result

    def format_predictions(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Format predictions as strings with units (e.g., "10.0 ng/mL").

        Args:
            y: DataFrame with numeric predictions (original scale)

        Returns:
            DataFrame with formatted string predictions
        """
        result = pd.DataFrame(index=y.index)

        for factor_name in y.columns:
            meta = self.factor_metadata[factor_name]

            def format_value(val):
                if pd.isna(val):
                    return None
                return f"{val:.1f} {meta.unit}"

            result[factor_name] = y[factor_name].apply(format_value)

        return result

    def save_metadata(self, filepath: str):
        """Save factor metadata to JSON file."""
        data = {
            name: asdict(meta)
            for name, meta in self.factor_metadata.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_metadata(self, filepath: str):
        """Load factor metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.factor_metadata = {
            name: FactorMetadata(**meta_dict)
            for name, meta_dict in data.items()
        }

        # Recreate scalers from metadata
        for name, meta in self.factor_metadata.items():
            if meta.model_type == "regression" and meta.scaler_mean is not None:
                scaler = StandardScaler()
                scaler.mean_ = np.array([meta.scaler_mean])
                scaler.scale_ = np.array([meta.scaler_std])
                scaler.var_ = np.array([meta.scaler_std ** 2])
                scaler.n_features_in_ = 1
                self.scalers[name] = scaler
            else:
                self.scalers[name] = None

    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all factors."""
        rows = []
        for name, meta in self.factor_metadata.items():
            rows.append({
                'factor': name,
                'unit': meta.unit,
                'n_samples': meta.n_samples,
                'n_unique': meta.n_unique_values,
                'model_type': meta.model_type,
                'constant_value': meta.constant_value
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame({
        'egf': ['10 ng/mL', '20 ng/mL', '50 ng/mL', None, '20 ng/mL'],
        'y27632': ['10 uM', '10 uM', None, '10 uM', '10 uM'],
        'cholera_toxin': ['9.0 ng/mL', '25 ng/mL', None, None, '9.0 ng/mL']
    })

    normalizer = FactorNormalizer()
    normalized = normalizer.fit_transform(test_data)

    print("Factor Summary:")
    print(normalizer.get_factor_summary())
    print("\nOriginal Data:")
    print(test_data)
    print("\nNormalized Data:")
    print(normalized)
