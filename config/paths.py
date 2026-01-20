"""
Centralized Path Configuration
==============================

All paths are relative to PROJECT_ROOT.
This file is the single source of truth for all file paths in the project.
"""

from pathlib import Path

# Project root (works anywhere the project is cloned)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Database
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "organoid_data.db"

# Source data files
TSV_PATH = RAW_DATA_DIR / "model-table.tsv"
MAF_CACHE_DIR = CACHE_DIR / "maf_cache"

# Outputs
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
                     DATABASE_DIR, LOGS_DIR, MODEL_ARTIFACTS_DIR, OUTPUTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
