# Beta Model Training Report

**Generated:** 2026-01-27 01:33
**Version:** beta-1.0

## Data Split

- **Training samples:** 528 (80%)
- **Test samples:** 132 (20%)
- **Cross-validation:** 5-fold on training set

## Cross-Validation Results (Training Set)

| Factor | Metric | Score | Std | Samples |
|--------|--------|-------|-----|---------|
| egf | R² | 0.932 | 0.038 | 207 |
| insulin | Score | 0.000 | 0.000 | 5 |

## Test Set Results (Held-out 20%)

| Factor | Metric | Score | Samples |
|--------|--------|-------|---------|
| egf | R² | 0.993 | 49 |
| y27632 | Accuracy | 0.690 | 42 |
| n_acetyl_cysteine | Accuracy | 0.485 | 33 |
| a83_01 | Accuracy | 0.485 | 33 |
| sb202190 | Accuracy | 0.588 | 17 |
| fgf2 | Accuracy | 0.938 | 16 |

## Model Architecture

- **Algorithm:** XGBoost
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 4
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8

## Usage

```python
from beta import BetaMediaPredictor, BetaPreprocessor

# Load models
model = BetaMediaPredictor.load('beta_output/beta_model.joblib')
prep = BetaPreprocessor.load('beta_output/beta_preprocessor.joblib')

# Predict
X, _ = prep.transform(new_data)
predictions = model.predict(X)
```

---
*Beta Model for Event Demo*