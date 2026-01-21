#!/usr/bin/env python3
"""
Download organoid_data.db from Google Drive.

Usage:
    pip install gdown
    python scripts/download_database.py
"""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: gdown not installed. Run: pip install gdown")
    sys.exit(1)

# Google Drive file ID
FILE_ID = "1B-E9pScJRukGVa9Tckc-JxMWDCw_AhhB"
OUTPUT_DIR = Path(__file__).parent.parent / "database"
OUTPUT_PATH = OUTPUT_DIR / "organoid_data.db"


def main():
    print("=" * 50)
    print("Downloading organoid_data.db from Google Drive")
    print("=" * 50)
    print(f"File ID: {FILE_ID}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: ~4.8 GB (this may take a while)")
    print()

    # Create directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if OUTPUT_PATH.exists():
        size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
        print(f"Database already exists ({size_gb:.2f} GB)")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Download
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(OUTPUT_PATH), quiet=False)

    # Verify
    if OUTPUT_PATH.exists():
        size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
        print(f"\nSuccess! Downloaded {size_gb:.2f} GB")
        print(f"Location: {OUTPUT_PATH}")
    else:
        print("\nError: Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
