# Beta Publish Model - Spin-off Pipeline Plan

## Goal
Create a separate, publishable beta model with honest limitations, confidence scoring, and clear scope - independent from main development.

---

## Current State Analysis

### Data Reality
| Metric | Value | Implication |
|--------|-------|-------------|
| Total samples | 660 | Decent for beta |
| With media data | 317 (48%) | Limits training |
| EGF samples | 256 | Good for regression |
| Cholera toxin | 25 | Too sparse - exclude |
| Insulin | 5 | Too sparse - exclude |

### Cancer Type Coverage
- **Strong (>50 samples):** Colon (126), Pancreas (101), Esophagus (72), Brain (61)
- **Moderate (20-50):** Skin (46), Rectum (42), Stomach (32), Ovary (24)
- **Weak (<20):** Everything else - return "insufficient data"

### Current Gaps
1. No prediction confidence scores
2. No cancer type stratification
3. No out-of-distribution detection
4. Suspiciously high R² (0.995) - needs investigation

---

## Beta Model Design

### Principle: Honest Over Impressive

**Include only what we can defend:**
- Factors with >50 training samples
- Cancer types with >20 samples
- Confidence scores on every prediction
- Clear "insufficient data" responses

### Scope Limits

**Factors to Include (6 of 8):**
| Factor | Samples | Model Type | Include |
|--------|---------|------------|---------|
| egf | 256 | Regression | YES |
| y27632 | 231 | Binary | YES |
| n_acetyl_cysteine | 176 | Binary | YES |
| a83_01 | 176 | Binary | YES |
| sb202190 | 87 | Binary | YES |
| fgf2 | 75 | Binary | YES |
| cholera_toxin | 25 | Regression | NO - too sparse |
| insulin | 5 | Constant | NO - no variance |

**Cancer Types to Support (8 of 41):**
- Colon, Pancreas, Esophagus, Brain, Skin, Rectum, Stomach, Ovary
- All others: Return "insufficient training data for this cancer type"

---

## Architecture

### New Files (Isolated from Main)

```
organoid-media-ml/
├── beta/                           # NEW: Isolated beta module
│   ├── __init__.py
│   ├── config.py                   # Beta-specific settings
│   ├── model.py                    # BetaMediaPredictor class
│   ├── confidence.py               # Confidence scoring system
│   ├── validators.py               # Input validation + scope checks
│   └── api.py                      # Simple prediction interface
│
├── scripts/
│   └── train_beta_model.py         # NEW: Beta training script
│
├── model_artifacts/
│   └── beta_v1.joblib              # NEW: Beta model artifact
│
└── outputs/
    └── beta_validation_report.json # NEW: Honest metrics
```

### BetaMediaPredictor Class

```python
class BetaMediaPredictor:
    """
    Publishable beta model with:
    - Scoped predictions (supported factors/cancer types only)
    - Confidence scores on every prediction
    - Clear "insufficient data" responses
    """

    SUPPORTED_CANCER_TYPES = [
        'Colon', 'Pancreas', 'Esophagus', 'Brain',
        'Skin', 'Rectum', 'Stomach', 'Ovary'
    ]

    SUPPORTED_FACTORS = [
        'egf', 'y27632', 'n_acetyl_cysteine',
        'a83_01', 'sb202190', 'fgf2'
    ]

    def predict(self, sample: dict) -> BetaPrediction:
        # 1. Validate input
        validation = self.validate_input(sample)
        if not validation.is_supported:
            return BetaPrediction(
                status="unsupported",
                reason=validation.reason,
                recipe=None
            )

        # 2. Get predictions with confidence
        recipe = {}
        for factor in self.SUPPORTED_FACTORS:
            pred, confidence = self._predict_factor(sample, factor)
            recipe[factor] = {
                'value': pred,
                'confidence': confidence,  # 0.0 - 1.0
                'confidence_level': self._confidence_label(confidence)
            }

        # 3. Overall confidence
        overall_confidence = self._compute_overall_confidence(recipe)

        return BetaPrediction(
            status="success",
            cancer_type=sample['primary_site'],
            recipe=recipe,
            overall_confidence=overall_confidence,
            warnings=self._generate_warnings(sample, recipe)
        )
```

### Confidence Scoring System

```python
class ConfidenceScorer:
    """
    Compute prediction confidence from multiple signals:
    1. Model uncertainty (tree variance for GB models)
    2. Data coverage (how many similar samples in training)
    3. Feature completeness (% of features present)
    """

    def score(self, sample, factor, prediction) -> float:
        # Weight components
        model_conf = self._model_confidence(factor, sample)      # 40%
        coverage_conf = self._coverage_confidence(sample)        # 35%
        completeness_conf = self._completeness_confidence(sample) # 25%

        return (0.40 * model_conf +
                0.35 * coverage_conf +
                0.25 * completeness_conf)

    def _model_confidence(self, factor, sample):
        """Extract uncertainty from model internals"""
        if factor in self.regression_factors:
            # Use tree variance from GradientBoosting
            predictions = [tree.predict(sample) for tree in model.estimators_]
            variance = np.var(predictions)
            # Convert variance to confidence (lower variance = higher confidence)
            return 1 / (1 + variance)
        else:
            # Binary classifier: use predict_proba distance from 0.5
            proba = model.predict_proba(sample)[0]
            return abs(proba[1] - 0.5) * 2  # Scale to 0-1

    def _coverage_confidence(self, sample):
        """How well represented is this sample in training data?"""
        cancer_type = sample['primary_site']
        n_similar = self.training_counts.get(cancer_type, 0)
        # More samples = higher confidence, diminishing returns
        return min(1.0, n_similar / 50)

    def _completeness_confidence(self, sample):
        """What fraction of expected features are present?"""
        n_present = sum(1 for v in sample.values() if v is not None)
        n_expected = len(self.expected_features)
        return n_present / n_expected
```

### Confidence Labels

| Score | Label | User Guidance |
|-------|-------|---------------|
| 0.8 - 1.0 | HIGH | Prediction well-supported by training data |
| 0.5 - 0.8 | MODERATE | Use with caution, consider literature validation |
| 0.3 - 0.5 | LOW | Limited data support, verify experimentally |
| 0.0 - 0.3 | EXPERIMENTAL | Insufficient data, use as starting point only |

### Output Format: Actual Values from Database

**Critical Rule:** All predicted values must exist in the training database. No interpolation or extrapolation.

```python
# For regression factors (egf):
# Prediction must be one of the observed values in training data
OBSERVED_EGF_VALUES = [5, 10, 20, 25, 50, 100]  # From database

def _constrain_to_observed(prediction, observed_values):
    """Snap prediction to nearest observed value"""
    return min(observed_values, key=lambda x: abs(x - prediction))

# Output example:
{
    'egf': {
        'value': 50,           # Actual value from database
        'unit': 'ng/mL',
        'confidence': 0.82,
        'confidence_level': 'HIGH',
        'source': 'database'   # Always 'database', never 'interpolated'
    }
}
```

This ensures predictions are always defensible - every value has been used successfully in real organoid cultures in the training data.

---

## Implementation Steps

### Phase 1: Create Beta Module Structure
1. Create `beta/` directory with `__init__.py`
2. Create `beta/config.py` with scope limits (supported factors/cancer types)
3. Create `beta/validators.py` with input validation

### Phase 2: Build Confidence System
1. Create `beta/confidence.py` with ConfidenceScorer class
2. Implement model uncertainty extraction (tree variance)
3. Implement coverage and completeness scoring
4. Add confidence label mapping

### Phase 3: Build Beta Predictor
1. Create `beta/model.py` with BetaMediaPredictor class
2. Wrap existing MediaRecipeGenerator with scope checks
3. Add confidence scoring to predictions
4. Implement warning generation

### Phase 4: Training Script
1. Create `scripts/train_beta_model.py`
2. Filter training data to supported scope
3. Train with explicit validation split (not just CV)
4. Generate honest validation metrics
5. Save to `model_artifacts/beta_v1.joblib`

### Phase 5: Simple API
1. Create `beta/api.py` with simple interface
2. Single function: `predict_media_recipe(sample_dict) -> dict`
3. Include example usage in docstring

### Phase 6: Documentation & Validation Report
1. Generate `outputs/beta_validation_report.json` with:
   - Per-factor metrics on held-out test set
   - Cancer type stratified performance
   - Confidence calibration analysis
2. Create `beta/README.md` with:
   - Supported scope
   - Limitations
   - Usage examples
   - Citation request

---

## Files to Modify/Create

### New Files
| File | Purpose |
|------|---------|
| `beta/__init__.py` | Module init, export main classes |
| `beta/config.py` | Scope limits, thresholds |
| `beta/validators.py` | Input validation |
| `beta/confidence.py` | Confidence scoring |
| `beta/model.py` | BetaMediaPredictor |
| `beta/api.py` | Simple prediction interface |
| `beta/README.md` | Usage documentation |
| `scripts/train_beta_model.py` | Training script |

### No Modifications to Existing Files
The beta module is completely isolated - main development continues unaffected.

---

## Verification Plan

### 1. Unit Tests
```bash
python -m pytest beta/tests/ -v
```

### 2. Validate Scope Enforcement
```python
# Should return "unsupported" for rare cancer
result = predictor.predict({'primary_site': 'Appendix', ...})
assert result.status == "unsupported"

# Should return prediction for supported cancer
result = predictor.predict({'primary_site': 'Colon', ...})
assert result.status == "success"
assert 'confidence' in result.recipe['egf']
```

### 3. Confidence Calibration Check
```python
# High confidence predictions should be more accurate
high_conf = [p for p in predictions if p.overall_confidence > 0.8]
low_conf = [p for p in predictions if p.overall_confidence < 0.5]
assert accuracy(high_conf) > accuracy(low_conf)
```

### 4. End-to-End Test
```bash
python scripts/train_beta_model.py
python -c "from beta.api import predict_media_recipe; print(predict_media_recipe({'primary_site': 'Colon', 'age_at_diagnosis': 55}))"
```

---

## Success Criteria

1. Beta model only predicts for 8 supported cancer types
2. Every prediction includes confidence score (0-1)
3. Validation report shows honest metrics (no cherry-picking)
4. API is simple: one function, one dict in, one dict out
5. Documentation clearly states limitations
6. No changes to main codebase - fully isolated

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Structure | Small |
| Phase 2: Confidence | Medium |
| Phase 3: Predictor | Medium |
| Phase 4: Training | Small |
| Phase 5: API | Small |
| Phase 6: Docs | Small |

