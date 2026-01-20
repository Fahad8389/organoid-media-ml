# Technical Log - Organoid Media ML

## System Architecture

### Data Flow
1. **Raw Data** -> HCMI clinical metadata + GDC genomic MAF files
2. **ETL Pipeline** -> `scripts/data_pipeline/` extracts and transforms
3. **Database** -> SQLite `organoid_data.db` stores all data
4. **Training Data** -> `master_dataset_v2.csv` (660 samples)
5. **ML Pipeline** -> `src/` modules for preprocessing and modeling
6. **Outputs** -> Trained models, metrics, feature importance

### Database Tables
- `media_protocols` - Raw media text per case_id
- `media_factors_structured` - Extracted factor concentrations
- `master_dataset_v2` - Final joined dataset

### ML Pipeline Components

**Preprocessing (`src/preprocessing/`):**
- `factor_normalizer.py` - Parse "10 ng/mL" -> numeric + unit
- `vaf_preprocessor.py` - Handle missing genomic data
- `preprocessing_pipeline.py` - Full feature engineering

**Models (`src/models/`):**
- `media_recipe_generator.py` - Per-factor model ensemble

**Evaluation (`src/evaluation/`):**
- `cross_validation.py` - K-fold CV framework
- `recipe_evaluator.py` - Recipe-level metrics

### Key Design Decisions
1. **Masked training** - Don't train on NULL targets
2. **NULL indicators** - Explicit missing data handling
3. **Pathway features** - Aggregate mutations by biological pathway
4. **Per-factor models** - Independent model per media component

---

## Directory Structure

```
organoid-media-ml/
├── config/
│   ├── __init__.py
│   └── paths.py              # Centralized path configuration
├── data/
│   ├── raw/                  # Original data files
│   ├── processed/            # Cleaned datasets
│   │   └── master_dataset_v2.csv
│   └── cache/                # Temporary cached data
├── database/
│   └── organoid_data.db      # SQLite database (4.5GB)
├── docs/
│   ├── DATA_DICTIONARY.md    # Auto-generated schema reference
│   └── TROUBLESHOOTING.md    # Common errors and fixes
├── examples/
│   └── predict_sample.py     # Usage example
├── logs/                     # Training and processing logs
├── model_artifacts/
│   └── media_recipe_generator.joblib
├── notebooks/                # Jupyter notebooks for exploration
├── outputs/                  # Training outputs, metrics
├── scripts/
│   ├── data_pipeline/        # ETL scripts
│   ├── docs/                 # Documentation generators
│   ├── scrapers/             # Data collection scripts
│   └── verification/         # Data validation scripts
├── src/
│   ├── evaluation/           # Model evaluation modules
│   ├── models/               # ML model implementations
│   └── preprocessing/        # Data preprocessing modules
├── CLAUDE.md                 # AI assistant guide
├── PROJECT_PLAN.md           # Progress tracker
├── TECHNICAL_LOG.md          # This file
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
└── setup.py                  # Package installation
```

---

## Database Schema

### Core Tables

**media_protocols**
- `case_id` (TEXT, PRIMARY KEY) - HCMI case identifier
- `media_raw` (TEXT) - Raw media protocol text

**media_factors_structured**
- `case_id` (TEXT, FOREIGN KEY)
- `egf` (REAL) - EGF concentration in ng/mL
- `y27632` (INTEGER) - Binary presence/absence
- `n_acetyl_cysteine` (INTEGER) - Binary
- `a83_01` (INTEGER) - Binary
- `sb202190` (INTEGER) - Binary
- `fgf2` (INTEGER) - Binary
- `cholera_toxin` (REAL) - Concentration in ng/mL
- `insulin` (INTEGER) - Binary

**master_dataset_v2**
- Clinical features (primary_site, age_at_diagnosis, etc.)
- 50 VAF columns (TP53_vaf, KRAS_vaf, etc.)
- 8 media factor target columns
- Pathway aggregate features

---

## Model Architecture

### MediaRecipeGenerator

Per-factor model ensemble that trains independent models for each media component:

```python
class MediaRecipeGenerator:
    def __init__(self):
        self.models = {}  # One model per factor
        self.factor_types = {
            'egf': 'regression',
            'cholera_toxin': 'regression',
            'y27632': 'classification',
            # ... etc
        }

    def fit(self, X, y_dict):
        for factor, y in y_dict.items():
            # Mask: only train on non-NULL samples
            mask = y.notna()
            self.models[factor].fit(X[mask], y[mask])

    def predict(self, X):
        return {f: model.predict(X) for f, model in self.models.items()}
```

### Feature Engineering

1. **Clinical Features**
   - One-hot encoded: primary_site, gender, race
   - Numeric: age_at_diagnosis

2. **Genomic Features**
   - Top 50 genes by mutation frequency
   - VAF (Variant Allele Frequency) values
   - Binary mutation indicators
   - NULL indicators for missing data

3. **Pathway Features**
   - Aggregated mutation counts by biological pathway
   - Examples: DNA_repair_pathway, cell_cycle_pathway

---

## Training Pipeline

```
1. Load Data
   └── Read master_dataset_v2.csv

2. Preprocess
   ├── Handle missing values (NULL indicators)
   ├── Normalize VAF values
   ├── One-hot encode categoricals
   └── Create pathway features

3. Split Data
   └── Stratified train/test split (80/20)

4. Train Models
   ├── For each media factor:
   │   ├── Mask NULL targets
   │   ├── Select appropriate model type
   │   └── Fit with cross-validation
   └── Save ensemble to joblib

5. Evaluate
   ├── Per-factor metrics (R2, accuracy)
   ├── Recipe-level evaluation
   └── Feature importance analysis

6. Save Artifacts
   ├── model_artifacts/media_recipe_generator.joblib
   ├── outputs/metrics.json
   └── outputs/feature_importance.csv
```

---

## Important Code Patterns

### Path Configuration
Always use centralized paths from `config/paths.py`:
```python
from config.paths import DB_PATH, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR
```

### Database Connection
```python
import sqlite3
from config.paths import DB_PATH

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM master_dataset_v2", conn)
conn.close()
```

### Loading Trained Model
```python
from src.models import MediaRecipeGenerator
from config.paths import MODEL_ARTIFACTS_DIR

model = MediaRecipeGenerator.load(MODEL_ARTIFACTS_DIR / "media_recipe_generator.joblib")
predictions = model.predict(X_new)
```
