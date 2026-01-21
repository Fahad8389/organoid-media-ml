# Organoid Media Recipe Generator

ML system that predicts organoid culture media formulations from clinical and genomic profiles.

## Overview

This project uses machine learning to predict optimal culture media compositions for organoid models based on:
- Clinical metadata (tissue type, disease status, patient demographics)
- Genomic features (Top 50 cancer gene VAF values)

## Quick Start

```bash
# Clone and install
git clone https://github.com/Fahad8389/organoid-media-ml.git
cd organoid-media-ml
pip install -r requirements.txt
pip install -e .

# Download database (~4.8 GB)
python scripts/download_database.py

# Verify setup
python scripts/verification/verify_db_link.py

# Run training
python train.py
```

## Project Structure

```
organoid-media-ml/
├── config/
│   └── paths.py                 # Centralized path configuration
│
├── data/
│   ├── raw/                     # Original data files
│   ├── processed/               # Cleaned datasets
│   └── cache/                   # Downloaded MAF files
│
├── database/
│   └── organoid_data.db         # SQLite database (not in git)
│
├── scripts/
│   ├── data_pipeline/           # ETL scripts
│   ├── scrapers/                # Web scrapers
│   └── verification/            # Data validation
│
├── src/                         # ML Package
│   ├── preprocessing/           # Feature engineering
│   ├── models/                  # Model definitions
│   ├── evaluation/              # Cross-validation
│   └── interpretability/        # Feature importance
│
├── notebooks/                   # Jupyter notebooks
├── model_artifacts/             # Trained models
├── outputs/                     # Results and metrics
├── logs/                        # Log files
│
├── train.py                     # Main training script
├── FINAL_REPORT.md              # Results summary
└── requirements.txt             # Dependencies
```

## Results Summary

| Target | Model Type | Metric | Score |
|--------|-----------|--------|-------|
| EGF | Regression | R² | 0.96 |
| Y-27632 | Binary | AUC | 0.94 |
| N-acetyl-cysteine | Binary | AUC | 0.89 |
| A83-01 | Binary | AUC | 0.96 |
| SB202190 | Binary | AUC | 0.95 |
| FGF2 | Binary | AUC | 0.93 |
| Cholera Toxin | Binary | AUC | 0.89 |
| Insulin | Binary | AUC | 0.91 |

See `FINAL_REPORT.md` for detailed analysis.

## Data Sources

- **HCMI**: Human Cancer Models Initiative clinical/genomic data
- **ATCC**: Media formulation protocols
- **GDC**: Masked somatic MAF files for VAF extraction

## Key Features

- **Masked Training**: Each factor model trains only on samples with non-NULL values
- **NULL Indicator Encoding**: Handles missing genomic data explicitly
- **Pathway Features**: Aggregated mutation burden by cancer pathway
- **Factor Normalization**: Converts diverse units to standardized numeric values

## License

[Add your license here]
