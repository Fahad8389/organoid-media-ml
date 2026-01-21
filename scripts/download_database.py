#!/usr/bin/env python3
"""
Download organoid_data.db from Google Drive.

Usage:
    pip install gdown
    python scripts/download_database.py

Database Version: 3.0.0 (2026-01-22)
- Includes 8 new tables for ML training
- See database/db_metadata.json for full schema
"""

import sys
import json
import sqlite3
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: gdown not installed. Run: pip install gdown")
    sys.exit(1)

# Google Drive file ID (UPDATE THIS AFTER UPLOADING NEW VERSION)
FILE_ID = "1B-E9pScJRukGVa9Tckc-JxMWDCw_AhhB"  # TODO: Update after cloud upload
EXPECTED_VERSION = "3.0.0"
OUTPUT_DIR = Path(__file__).parent.parent / "database"
OUTPUT_PATH = OUTPUT_DIR / "organoid_data.db"
METADATA_PATH = OUTPUT_DIR / "db_metadata.json"

# Tables that should exist in v3.0
REQUIRED_TABLES_V3 = [
    "master_dataset_v3",
    "media_factors_v3",
    "gene_expression_top1000",
    "gene_expression_pathways",
    "gene_expression_markers",
    "top_variable_genes",
    "data_cleaning_log",
    "outlier_flags"
]


def verify_database_version():
    """Check if database has the expected v3.0 tables."""
    if not OUTPUT_PATH.exists():
        return False, "Database file not found"

    try:
        conn = sqlite3.connect(str(OUTPUT_PATH))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        missing = [t for t in REQUIRED_TABLES_V3 if t not in tables]
        if missing:
            return False, f"Missing tables: {missing}"

        return True, f"Database v3.0 verified ({len(tables)} tables)"
    except Exception as e:
        return False, f"Verification error: {e}"


def main():
    print("=" * 60)
    print("Downloading organoid_data.db from Google Drive")
    print("=" * 60)
    print(f"File ID: {FILE_ID}")
    print(f"Expected Version: {EXPECTED_VERSION}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: ~4.8 GB (this may take a while)")
    print()

    # Create directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already exists and is correct version
    if OUTPUT_PATH.exists():
        size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
        print(f"Database already exists ({size_gb:.2f} GB)")

        is_valid, msg = verify_database_version()
        if is_valid:
            print(f"✓ {msg}")
            print("Database is up to date. No download needed.")
            return
        else:
            print(f"⚠ {msg}")
            print("Database needs update.")

        response = input("Download new version? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Download
    print("\nStarting download...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(OUTPUT_PATH), quiet=False)

    # Verify download
    if OUTPUT_PATH.exists():
        size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
        print(f"\n✓ Downloaded {size_gb:.2f} GB")

        # Verify database version
        is_valid, msg = verify_database_version()
        if is_valid:
            print(f"✓ {msg}")
        else:
            print(f"⚠ WARNING: {msg}")
            print("  The downloaded database may be an older version.")
            print("  Check if Google Drive file has been updated.")
    else:
        print("\n✗ Error: Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
