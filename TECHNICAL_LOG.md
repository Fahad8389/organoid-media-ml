# Technical Log - Organoid Media ML

## System Architecture

### Data Flow
1. **Raw Data** -> HCMI clinical metadata + GDC genomic MAF files
2. **ETL Pipeline** -> `scripts/data_pipeline/` extracts and transforms
3. **Database** -> SQLite `organoid_data.db` stores all data
4. **Training Data** -> `master_dataset_v2.csv` (660 samples)
5. **ML Pipeline** -> `src/` modules for preprocessing and modeling
6. **Outputs** -> Trained models, metrics, feature importance

### Database Tables (v3.0)

**Core Tables:**
- `clinical_data_full` - 660 rows, patient metadata
- `media_protocols` - 387 rows, raw ATCC media text
- `gene_expression` - 29.7M rows, RNA-seq TPM values
- `mutations` - 182K rows, somatic variants

**V3.0 Tables (Added 2026-01-22):**
- `media_factors_v3` - 660 × 60, 13 factors with audit trail
- `master_dataset_v3` - 660 × 206, unified dataset
- `gene_expression_top1000` - 490 × 1003, top variable genes (raw TPM)
- `gene_expression_pathways` - 490 × 29, pathway aggregates
- `gene_expression_markers` - 490 × 26, curated marker genes
- `top_variable_genes` - 1000 rows, genes ranked by variance
- `data_cleaning_log` - Audit trail for data cleaning
- `outlier_flags` - Flagged outliers for review

**Legacy Tables:**
- `media_factors_structured` - 317 rows (superseded by v3)
- `master_dataset_v2` - 660 rows (superseded by v3)

### ML Pipeline Components

**Preprocessing (`src/preprocessing/`):**
- `factor_normalizer.py` - Parse "10 ng/mL" -> numeric + unit
- `vaf_preprocessor.py` - Handle missing genomic data
- `preprocessing_pipeline.py` - Full feature engineering

**Models (`src/models/`):**
- `media_recipe_generator.py` - Per-factor model ensemble

**Beta Pipeline (`beta/`):**
- `model.py` - BetaMediaPredictor with XGBoost
- `preprocessing.py` - BetaPreprocessor
- `confidence.py` - ConfidenceScorer
- `validators.py` - InputValidator
- `train.py` - Training script with 5-fold CV
- `api.py` - Simple API for demo predictions

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
├── beta/                     # Beta model pipeline (XGBoost)
│   ├── model.py
│   ├── preprocessing.py
│   ├── confidence.py
│   ├── validators.py
│   ├── train.py
│   └── api.py
├── beta_output/              # Beta model artifacts
│   ├── beta_model.joblib
│   ├── beta_preprocessor.joblib
│   └── beta_metrics.json
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
│   ├── verification/         # Data validation scripts
│   └── download_database.py  # Download DB from Google Drive
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

---

## Session Log

### 2026-01-20: Infrastructure Setup Complete

**Accomplishments:**
1. Created session continuity documentation system:
   - `CLAUDE.md` - AI assistant guide with golden rules and workflow protocol
   - `TECHNICAL_LOG.md` - System architecture documentation (this file)
   - `PROJECT_PLAN.md` - Progress tracker with stop points

2. Created supporting documentation:
   - `docs/DATA_DICTIONARY.md` - Auto-generated database schema (11 tables)
   - `docs/TROUBLESHOOTING.md` - Common errors and fixes
   - `examples/predict_sample.py` - Usage example script
   - `scripts/docs/generate_data_dict.py` - Schema auto-generation tool

3. GitHub repository setup:
   - Repository: https://github.com/Fahad8389/organoid-media-ml (private)
   - 3 commits pushed to main branch

**Database Summary (from DATA_DICTIONARY.md):**
| Table | Rows | Purpose |
|-------|------|---------|
| master_dataset_v2 | 660 | Final training dataset |
| clinical_data_full | 660 | HCMI clinical metadata |
| gene_expression | 29.7M | RNA-seq TPM values |
| mutations | 182K | Somatic variants from MAF |
| mutations_top50_pivot | 632 | VAF for top 50 genes |
| media_protocols | 387 | Raw ATCC media text |
| media_factors_structured | 317 | Parsed media components |

**Stop Point:** Phase 5 Complete: Infrastructure & Governance system fully established
**Next Session:** Review system integrity and plan the next research phase

---

### 2026-01-21: Database Distribution Setup

**Accomplishments:**
1. Reviewed repository completeness:
   - Confirmed `organoid_data.db` is the only file excluded from Git
   - Verified all source code, models, and documentation are present

2. Set up database distribution via Google Drive:
   - Uploaded organoid_data.db (4.8GB) to Google Drive
   - File ID: `1B-E9pScJRukGVa9Tckc-JxMWDCw_AhhB`
   - Public link: https://drive.google.com/file/d/1B-E9pScJRukGVa9Tckc-JxMWDCw_AhhB/view

3. Created automated download workflow:
   - Added `scripts/download_database.py` using gdown library
   - Added `gdown>=5.0` to requirements.txt
   - Updated README.md with simplified Quick Start

4. Pushed to GitHub (commit 6c7ceb8)

**New User Workflow:**
```bash
git clone https://github.com/Fahad8389/organoid-media-ml.git
cd organoid-media-ml
pip install -r requirements.txt
python scripts/download_database.py  # Downloads 4.8GB from Google Drive
python scripts/verification/verify_db_link.py
```

**Stop Point:** Phase 6 Complete: Database distribution system established
**Next Session:** Data Management

---

### 2026-01-22: Database Upgrade v3.0

**Accomplishments:**

1. **Phase 1 - Formulation Mapping Analysis:**
   - Analyzed ATCC formulation references in raw text
   - Discovered concentrations NOT in raw text (only formulation #s)
   - Decision: Option B - use only existing data, NULL for missing

2. **Phase 2 - Media Factors v3:**
   - Created `media_factors_v3` table (660 rows × 60 cols)
   - Parsed 5 existing factors: EGF (39%), Y-27632 (35%), A83-01 (27%), N-Acetylcysteine (27%), SB202190 (13%)
   - Marked 8 missing factors as NULL: Wnt3a, R-spondin, Noggin, CHIR99021, B27, N2, Nicotinamide, Gastrin
   - Applied unit normalization (ng/mL, uM, mM, %)
   - Added audit trail columns

3. **Phase 3 - Gene Expression Processing:**
   - Created `top_variable_genes` (1000 genes ranked by variance)
   - Created `gene_expression_top1000` (490 × 1003) - wide format raw TPM
   - Created `gene_expression_pathways` (490 × 29) - 9 pathway aggregates
   - Created `gene_expression_markers` (490 × 26) - 25 curated markers

4. **Phase 4 - Master Dataset Integration:**
   - Created `master_dataset_v3` (660 × 206 columns)
   - Integrated: clinical + media factors + pathways + markers + mutations
   - DNA (VAF) and RNA (TPM) data side-by-side for same genes

5. **Phase 4.5 - Data Cleaning:**
   - Standardized missing values (Unknown, N/A, -- → NULL)
   - Cleaned categorical values (gender, tissue_status)
   - Created `data_cleaning_log` table (9 entries)

6. **Phase 4.6 - Outlier Detection:**
   - Created `outlier_flags` table (8 flagged cases)
   - Detected age inconsistencies (acquisition < diagnosis)

7. **Deployment:**
   - Updated `download_database.py` with v3.0 version checking
   - Created `database/db_metadata.json` for version tracking
   - Pushed to GitHub (commit afac91b)
   - Uploaded updated database to Google Drive
   - Verified cloud download has all v3.0 tables

**New Database Schema (v3.0):**
| Table | Rows | Columns | Purpose |
|-------|------|---------|---------|
| media_factors_v3 | 660 | 60 | 13 factors with audit trail |
| top_variable_genes | 1000 | 6 | Genes ranked by variance |
| gene_expression_top1000 | 490 | 1003 | Wide format raw TPM |
| gene_expression_pathways | 490 | 29 | 9 pathway aggregates |
| gene_expression_markers | 490 | 26 | 25 curated markers |
| master_dataset_v3 | 660 | 206 | Unified ML-ready dataset |
| data_cleaning_log | 9 | 8 | Cleaning audit trail |
| outlier_flags | 8 | 10 | Flagged outliers |

**Data Coverage Summary:**
- Total organoid models: 660
- With media factors: 287 (43.5%)
- With gene expression: 507 (76.8%)
- With mutation data: 522 (79.1%)

**Stop Point:** Phase 7 Complete: Database v3.0 deployed
**Next Session:** ML model retraining with new features

---

### 2026-01-24: Beta Publish Model Planning

**Accomplishments:**
1. Analyzed data weaknesses:
   - Only 317/660 samples have media data (48%)
   - Sparse factors: cholera_toxin (25), insulin (5)
   - Cancer type imbalance (Colon/Pancreas/Esophagus = 45%)

2. Designed Beta Publish Model pipeline:
   - Isolated `beta/` module (won't affect main development)
   - Scope limited to 8 supported cancer types (>20 samples each)
   - Scope limited to 6 factors (>50 training samples each)
   - Confidence scoring system (model variance + coverage + completeness)
   - Predictions constrained to actual database values only

3. Documented plan in `docs/BETA_PUBLISH_PLAN.md`

**Key Design Decisions:**
- Exclude cholera_toxin (25 samples) and insulin (5 samples) - too sparse
- Predictions snap to nearest observed value in database (no interpolation)
- Every prediction includes confidence score (0.0-1.0)
- Unsupported cancer types return "insufficient data" instead of guessing

**Stop Point:** Beta publish plan documented, ready for implementation
**Next Session:** Implement Beta Publish Pipeline per docs/BETA_PUBLISH_PLAN.md

---

### 2026-01-27: Beta Model Implementation Complete

**Accomplishments:**
1. Created complete `beta/` module:
   - `beta/__init__.py` - Package exports
   - `beta/model.py` - BetaMediaPredictor with XGBoost
   - `beta/preprocessing.py` - BetaPreprocessor with 80/20 split
   - `beta/confidence.py` - ConfidenceScorer for prediction reliability
   - `beta/validators.py` - InputValidator for data validation
   - `beta/train.py` - Training script with 5-fold CV
   - `beta/api.py` - Simple API for demo predictions

2. Trained beta model:
   - 660 samples total (528 train / 132 test)
   - XGBoost with tuned hyperparameters
   - 5-fold cross-validation on training set

3. Model performance results:
   | Factor | Type | CV Score | Test Score |
   |--------|------|----------|------------|
   | EGF | Regression | R² = 0.932 | R² = 0.993 |
   | Y-27632 | Binary | - | Acc = 0.690 |
   | FGF2 | Binary | - | Acc = 0.938 |
   | N-acetyl-cysteine | Binary | - | Acc = 0.485 |
   | A83-01 | Binary | - | Acc = 0.485 |
   | SB202190 | Binary | - | Acc = 0.588 |

4. Generated documentation:
   - `beta_output/BETA_REPORT.md` - Training report
   - Desktop exports for lab sharing:
     - Onoids_beta_model.docx (results report)
     - Onoids data base explain.docx (data protocol)
     - master_dataset_v2.csv (training data)

**Key Technical Decisions:**
- Used XGBoost instead of GradientBoosting for better tabular performance
- 80/20 split with stratification by primary_site
- Binary classifiers had CV issues due to class imbalance (single constant values)
- EGF is the only true regression target with variance

**Known Issues:**
- Binary classifiers show near-random accuracy for some factors (class imbalance)
- Cholera toxin and insulin excluded (n < 10 samples)
- 16 media factors have zero data coverage

**Stop Point:** Beta model trained and documented, ready for event demo
**Next Session:** Address class imbalance or deployment planning
