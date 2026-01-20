#!/usr/bin/env python3
"""
VAF (Variant Allele Frequency) Update Script
=============================================

Updates the mutations table with depth data (t_depth, t_ref_count, t_alt_count)
by re-downloading and parsing MAF files from GDC.

Calculates VAF = t_alt_count / t_depth

Usage:
    python update_vaf_data.py
"""

import sqlite3
import requests
import gzip
import pandas as pd
import io
import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH, TSV_PATH, MAF_CACHE_DIR, LOGS_DIR
LOG_FILE = str(LOGS_DIR / f"vaf_update_{datetime.now():%Y%m%d_%H%M%S}.log")

# GDC API
GDC_DATA_URL = "https://api.gdc.cancer.gov/data/"
REQUEST_TIMEOUT = 60
DELAY_BETWEEN_REQUESTS = 0.5

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Accept': '*/*',
}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 60)
    logging.info("VAF UPDATE SCRIPT STARTED")
    logging.info("=" * 60)


# =============================================================================
# SCHEMA UPDATE
# =============================================================================

def add_vaf_columns(conn: sqlite3.Connection) -> bool:
    """Add t_depth, t_ref_count, t_alt_count columns if they don't exist."""
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(mutations)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    columns_to_add = [
        ('t_depth', 'INTEGER'),
        ('t_ref_count', 'INTEGER'),
        ('t_alt_count', 'INTEGER'),
    ]

    added = []
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE mutations ADD COLUMN {col_name} {col_type}")
            added.append(col_name)

    conn.commit()

    if added:
        logging.info(f"Added columns: {', '.join(added)}")
    else:
        logging.info("VAF columns already exist")

    return True


# =============================================================================
# TSV PARSING - Get MAF UUIDs
# =============================================================================

def get_maf_uuids_from_tsv() -> Dict[str, str]:
    """
    Parse TSV to get case_id -> MAF file UUID mapping.

    Returns: {case_id: maf_file_uuid}
    """
    df = pd.read_csv(TSV_PATH, sep='\t')

    mapping = {}
    for _, row in df.iterrows():
        model_name = row['Name']
        maf_link = row.get('Link To Masked Somatic MAF', '')

        if pd.isna(maf_link) or maf_link == '--' or not maf_link:
            continue

        # Extract case_id from model name (e.g., HCM-CSHL-0056-C18 -> HCM-CSHL-0056)
        parts = model_name.split('-')
        if len(parts) >= 3:
            case_id = '-'.join(parts[:3])
        else:
            case_id = model_name

        # Extract UUID from URL
        # Format: https://portal.gdc.cancer.gov/files/dd8d5df7-d822-4c6b-b323-5e3d6cb2eede
        if '/files/' in maf_link:
            uuid = maf_link.split('/files/')[-1].split('?')[0]
            mapping[case_id] = uuid

    logging.info(f"Found {len(mapping)} case_id -> MAF UUID mappings")
    return mapping


# =============================================================================
# MAF FILE OPERATIONS
# =============================================================================

def get_cached_maf_path(uuid: str) -> str:
    """Get path for cached MAF file."""
    return os.path.join(MAF_CACHE_DIR, f"{uuid}.maf.gz")


def download_maf_file(uuid: str) -> Optional[pd.DataFrame]:
    """
    Download MAF file from GDC and parse it.
    Caches locally for future use.

    Returns DataFrame with mutation data or None on failure.
    """
    cache_path = get_cached_maf_path(uuid)

    # Check cache first
    if os.path.exists(cache_path):
        logging.info(f"  Using cached: {uuid}")
        try:
            with gzip.open(cache_path, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
            return df
        except Exception as e:
            logging.warning(f"  Cache read failed, re-downloading: {e}")

    # Download from GDC
    url = f"{GDC_DATA_URL}{uuid}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            logging.error(f"  HTTP {response.status_code} for {uuid}")
            return None

        # Decompress and parse
        content = gzip.decompress(response.content)
        df = pd.read_csv(io.BytesIO(content), sep='\t', comment='#', low_memory=False)

        # Cache for future use
        os.makedirs(MAF_CACHE_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(response.content)

        return df

    except requests.exceptions.Timeout:
        logging.error(f"  Timeout downloading {uuid}")
        return None
    except Exception as e:
        logging.error(f"  Error downloading {uuid}: {e}")
        return None


# =============================================================================
# DATABASE UPDATE
# =============================================================================

def get_cases_needing_update(conn: sqlite3.Connection) -> Set[str]:
    """Get case_ids that have mutations but no depth data."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT case_id
        FROM mutations
        WHERE t_depth IS NULL
    """)
    return {row[0] for row in cursor.fetchall()}


def update_mutations_with_depth(conn: sqlite3.Connection, case_id: str,
                                 maf_df: pd.DataFrame) -> int:
    """
    Update mutations table with depth data from MAF DataFrame.

    Matches on: case_id + hugo_symbol + variant_classification + protein_change

    Returns: number of rows updated
    """
    cursor = conn.cursor()

    # Verify MAF has required columns
    required_cols = {'Hugo_Symbol', 'Variant_Classification', 'HGVSp_Short',
                     't_depth', 't_ref_count', 't_alt_count'}

    if not required_cols.issubset(set(maf_df.columns)):
        missing = required_cols - set(maf_df.columns)
        logging.warning(f"  MAF missing columns: {missing}")
        return 0

    updated = 0

    for _, row in maf_df.iterrows():
        hugo = row['Hugo_Symbol']
        var_class = row['Variant_Classification']
        protein = row['HGVSp_Short'] if pd.notna(row['HGVSp_Short']) else None

        t_depth = int(row['t_depth']) if pd.notna(row['t_depth']) else None
        t_ref = int(row['t_ref_count']) if pd.notna(row['t_ref_count']) else None
        t_alt = int(row['t_alt_count']) if pd.notna(row['t_alt_count']) else None

        # Update matching mutation
        cursor.execute("""
            UPDATE mutations
            SET t_depth = ?, t_ref_count = ?, t_alt_count = ?
            WHERE case_id = ?
              AND hugo_symbol = ?
              AND variant_classification = ?
              AND (protein_change = ? OR (protein_change IS NULL AND ? IS NULL))
        """, (t_depth, t_ref, t_alt, case_id, hugo, var_class, protein, protein))

        if cursor.rowcount > 0:
            updated += cursor.rowcount

    conn.commit()
    return updated


# =============================================================================
# VAF VIEW
# =============================================================================

def create_vaf_view(conn: sqlite3.Connection):
    """Create a view that includes calculated VAF."""
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS mutations_with_vaf")
    cursor.execute("""
        CREATE VIEW mutations_with_vaf AS
        SELECT
            id,
            case_id,
            hugo_symbol,
            variant_classification,
            protein_change,
            t_depth,
            t_ref_count,
            t_alt_count,
            CASE
                WHEN t_depth > 0 THEN ROUND(CAST(t_alt_count AS REAL) / t_depth, 4)
                ELSE NULL
            END AS vaf
        FROM mutations
    """)
    conn.commit()
    logging.info("Created view: mutations_with_vaf (includes VAF calculation)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging()

    conn = sqlite3.connect(DB_PATH)

    # Step 1: Add columns
    logging.info("\n[Step 1] Adding VAF columns to schema...")
    add_vaf_columns(conn)

    # Step 2: Get MAF UUIDs from TSV
    logging.info("\n[Step 2] Loading MAF UUID mappings from TSV...")
    maf_uuids = get_maf_uuids_from_tsv()

    # Step 3: Get cases needing update
    logging.info("\n[Step 3] Finding cases needing VAF data...")
    cases_needing_update = get_cases_needing_update(conn)
    logging.info(f"   {len(cases_needing_update)} cases need VAF data")

    # Filter to cases we have MAF files for
    cases_to_process = cases_needing_update & set(maf_uuids.keys())
    logging.info(f"   {len(cases_to_process)} cases have MAF files available")

    if not cases_to_process:
        logging.info("No cases to process!")
        conn.close()
        return

    # Step 4: Download and update
    logging.info("\n[Step 4] Downloading MAF files and updating database...")
    logging.info("=" * 60)

    total_updated = 0
    success_count = 0
    fail_count = 0

    for idx, case_id in enumerate(sorted(cases_to_process), 1):
        uuid = maf_uuids[case_id]
        logging.info(f"[{idx}/{len(cases_to_process)}] {case_id} (UUID: {uuid[:8]}...)")

        # Download/load MAF file
        maf_df = download_maf_file(uuid)

        if maf_df is None:
            fail_count += 1
            continue

        # Update mutations
        updated = update_mutations_with_depth(conn, case_id, maf_df)
        total_updated += updated
        success_count += 1
        logging.info(f"  Updated {updated} mutations")

        # Polite delay
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Step 5: Create VAF view
    logging.info("\n[Step 5] Creating VAF view...")
    create_vaf_view(conn)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Cases processed successfully: {success_count}")
    logging.info(f"Cases failed: {fail_count}")
    logging.info(f"Total mutations updated: {total_updated}")

    # Verification
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM mutations WHERE t_depth IS NOT NULL")
    with_depth = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM mutations")
    total = cursor.fetchone()[0]

    logging.info(f"\nMutations with depth data: {with_depth}/{total} ({100*with_depth/total:.1f}%)")

    # Sample output
    logging.info("\n" + "=" * 60)
    logging.info("SAMPLE VAF DATA (5 rows)")
    logging.info("=" * 60)
    cursor.execute("""
        SELECT case_id, hugo_symbol, variant_classification,
               t_depth, t_alt_count,
               ROUND(CAST(t_alt_count AS REAL) / t_depth, 4) as vaf
        FROM mutations
        WHERE t_depth IS NOT NULL AND t_depth > 0
        ORDER BY RANDOM()
        LIMIT 5
    """)
    for row in cursor.fetchall():
        case, gene, var, depth, alt, vaf = row
        logging.info(f"  {case} | {gene} | {var} | depth={depth} alt={alt} VAF={vaf}")

    conn.close()
    logging.info(f"\n[DONE] Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
