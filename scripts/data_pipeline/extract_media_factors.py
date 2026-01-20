#!/usr/bin/env python3
"""
Media Factors Structured Extraction
====================================

Extracts growth factors, supplements, and reagents from media_raw_text
and creates a structured table with dynamic columns.

Usage:
    python extract_media_factors.py
"""

import sqlite3
import re
from collections import defaultdict
from typing import Dict, Set, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH

# =============================================================================
# KNOWN FACTORS AND SYNONYMS
# =============================================================================

# Map of synonyms to canonical column names
# Format: 'pattern_to_match': 'canonical_column_name'
FACTOR_SYNONYMS = {
    # Growth Factors
    'egf': 'egf',
    'epidermal growth factor': 'egf',
    'fgf': 'fgf',
    'fgf-10': 'fgf_10',
    'fgf10': 'fgf_10',
    'fgf-2': 'fgf_2',
    'fgf2': 'fgf_2',
    'bfgf': 'fgf_2',
    'basic fgf': 'fgf_2',
    'fgf-7': 'fgf_7',
    'fgf7': 'fgf_7',
    'kgf': 'fgf_7',
    'hgf': 'hgf',
    'hepatocyte growth factor': 'hgf',
    'noggin': 'noggin',
    'wnt': 'wnt',
    'wnt-3a': 'wnt_3a',
    'wnt3a': 'wnt_3a',
    'r-spondin': 'r_spondin',
    'rspondin': 'r_spondin',
    'rspo': 'r_spondin',
    'r-spondin-1': 'r_spondin_1',
    'rspondin1': 'r_spondin_1',

    # Supplements
    'b-27': 'b27',
    'b27': 'b27',
    'n-2': 'n2',
    'n2': 'n2',
    'glutamax': 'glutamax',
    'l-glutamine': 'l_glutamine',
    'glutamine': 'l_glutamine',
    'hepes': 'hepes',
    'pen/strep': 'pen_strep',
    'penicillin': 'pen_strep',
    'streptomycin': 'pen_strep',
    'antibiotic': 'antibiotic',
    'primocin': 'primocin',
    'gentamicin': 'gentamicin',

    # Small Molecules / Inhibitors
    'a83-01': 'a83_01',
    'a 83-01': 'a83_01',
    'a-83-01': 'a83_01',
    'sb202190': 'sb202190',
    'sb 202190': 'sb202190',
    'sb-202190': 'sb202190',
    'y-27632': 'y27632',
    'y27632': 'y27632',
    'rock inhibitor': 'y27632',
    'chir99021': 'chir99021',
    'chir 99021': 'chir99021',
    'gsk3 inhibitor': 'chir99021',
    'ldn193189': 'ldn193189',
    'ldn-193189': 'ldn193189',
    'sb431542': 'sb431542',
    'sb-431542': 'sb431542',
    'forskolin': 'forskolin',
    'dexamethasone': 'dexamethasone',
    'hydrocortisone': 'hydrocortisone',

    # Other Components
    'gastrin': 'gastrin',
    'nicotinamide': 'nicotinamide',
    'n-acetylcysteine': 'n_acetylcysteine',
    'n-acetyl-cysteine': 'n_acetylcysteine',
    'nac': 'n_acetylcysteine',
    'prostaglandin': 'prostaglandin',
    'pge2': 'prostaglandin_e2',
    'prostaglandin e2': 'prostaglandin_e2',
    'cholera toxin': 'cholera_toxin',
    'insulin': 'insulin',
    'transferrin': 'transferrin',
    'selenite': 'selenite',
    'its': 'its',  # Insulin-Transferrin-Selenite
    'heparin': 'heparin',
    'bsa': 'bsa',
    'bovine serum albumin': 'bsa',

    # Serum/Media
    'fbs': 'fbs',
    'fetal bovine serum': 'fbs',
    'serum': 'serum',

    # ECM
    'matrigel': 'matrigel',
    'basement membrane': 'basement_membrane',
    'ecm': 'ecm',
    'collagen': 'collagen',

    # Base Media (just presence)
    'dmem': 'dmem',
    'dmem/f12': 'dmem_f12',
    'advanced dmem': 'advanced_dmem',
    'rpmi': 'rpmi',
    'neurobasal': 'neurobasal',
}

# Factors to actively search for (regex patterns)
FACTOR_PATTERNS = [
    # Growth factors with concentrations
    (r'(\d+\.?\d*)\s*(ng/mL|ng/ml)\s+(?:recombinant\s+)?(?:human\s+)?(EGF|Noggin|FGF[\-\s]?\d*|HGF|Wnt[\-\s]?\d*[aA]?|R-?spondin[\-\s]?\d*)', 'growth_factor'),
    (r'(EGF|Noggin|FGF[\-\s]?\d*|HGF|Wnt[\-\s]?\d*[aA]?|R-?spondin[\-\s]?\d*)\s*[\(\[]?\s*(\d+\.?\d*)\s*(ng/mL|ng/ml)', 'growth_factor'),

    # Small molecules with concentrations
    (r'(\d+\.?\d*)\s*(uM|µM|nM|mM)\s+(A[\s\-]?83[\s\-]?01|SB[\s\-]?202190|Y[\s\-]?27632|CHIR[\s\-]?99021|LDN[\s\-]?193189|SB[\s\-]?431542|Forskolin|Dexamethasone|Hydrocortisone)', 'small_molecule'),
    (r'(A[\s\-]?83[\s\-]?01|SB[\s\-]?202190|Y[\s\-]?27632|ROCK\s+Inhibitor|CHIR[\s\-]?99021|LDN[\s\-]?193189|SB[\s\-]?431542|Forskolin|Dexamethasone|Hydrocortisone)\s*[\(\[]?\s*(\d+\.?\d*)\s*(uM|µM|nM|mM)', 'small_molecule'),

    # Supplements with concentrations
    (r'(\d+\.?\d*)\s*(ng/mL|ug/mL|µg/mL|uM|µM|nM|mM|%|x|X)\s+(Gastrin|Nicotinamide|N-?Acetyl[\s\-]?Cysteine|NAC|Prostaglandin|PGE2|Cholera\s+Toxin|Insulin|Transferrin|Selenite|Heparin|BSA)', 'supplement'),
    (r'(Gastrin|Nicotinamide|N-?Acetyl[\s\-]?Cysteine|NAC|Prostaglandin|PGE2|Cholera\s+Toxin|Insulin|Transferrin|Selenite|Heparin|BSA)\s*[\(\[]?\s*(\d+\.?\d*)\s*(ng/mL|ug/mL|µg/mL|uM|µM|nM|mM|%)', 'supplement'),

    # B-27 and N-2 (usually with x or %)
    (r'(\d+\.?\d*)\s*(x|X|%)\s+(B[\s\-]?27|N[\s\-]?2)', 'supplement_kit'),
    (r'(B[\s\-]?27|N[\s\-]?2)\s*[\(\[]?\s*(\d+\.?\d*)\s*(x|X|%)', 'supplement_kit'),

    # FBS and serum
    (r'(\d+\.?\d*)\s*(%)\s+(FBS|Fetal\s+Bovine\s+Serum|Serum)', 'serum'),
    (r'(FBS|Fetal\s+Bovine\s+Serum)\s*[\(\[]?\s*(\d+\.?\d*)\s*(%)', 'serum'),

    # HEPES
    (r'(\d+\.?\d*)\s*(mM|uM)\s+(HEPES)', 'buffer'),
    (r'(HEPES)\s*[\(\[]?\s*(\d+\.?\d*)\s*(mM|uM)', 'buffer'),

    # Glutamax/Glutamine
    (r'(\d+\.?\d*)\s*(mM|x|X|%)\s+(GlutaMAX|L-?Glutamine|Glutamine)', 'amino_acid'),
    (r'(GlutaMAX|L-?Glutamine)\s*[\(\[]?\s*(\d+\.?\d*)\s*(mM|x|X|%)', 'amino_acid'),
]


def normalize_factor_name(name: str) -> str:
    """Convert factor name to canonical column name."""
    name_lower = name.lower().strip()

    # Try multiple normalization variants for synonym matching
    variants = [
        name_lower,
        re.sub(r'[\s\-]+', ' ', name_lower),   # spaces for hyphens: "y-27632" -> "y 27632"
        re.sub(r'[\s\-]+', '-', name_lower),   # hyphens: "y 27632" -> "y-27632"
        re.sub(r'[\s\-]+', '', name_lower),    # no separators: "y-27632" -> "y27632"
    ]

    # Check synonyms against all variants
    for variant in variants:
        for pattern, canonical in FACTOR_SYNONYMS.items():
            if pattern == variant or variant == pattern:
                return canonical
            if pattern in variant or variant in pattern:
                return canonical

    # Default: clean the name for column use
    clean = re.sub(r'[^\w]', '_', name_lower)
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_')


def extract_factors_from_text(text: str) -> Dict[str, str]:
    """
    Extract all factors and their concentrations from text.

    Returns dict: {canonical_factor_name: 'concentration unit' or 'Present'}
    """
    if not text:
        return {}

    found_factors = {}
    text_lower = text.lower()

    # Apply all patterns
    for pattern, category in FACTOR_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) == 3:
                # Could be (value, unit, name) or (name, value, unit)
                if match[0].replace('.', '').isdigit():
                    value, unit, name = match
                else:
                    name, value, unit = match

                canonical = normalize_factor_name(name)
                conc = f"{value} {unit}"
                found_factors[canonical] = conc

    # Also check for presence without concentration
    presence_patterns = [
        r'\b(EGF|Noggin|FGF|HGF|Wnt|R-?spondin|Gastrin|Nicotinamide)\b',
        r'\b(B-?27|N-?2|GlutaMAX|HEPES|Matrigel)\b',
        r'\b(A[\s\-]?83[\s\-]?01|SB[\s\-]?202190|Y[\s\-]?27632|ROCK\s+Inhibitor)\b',
        r'\b(DMEM|DMEM/F12|Advanced\s+DMEM|RPMI|Neurobasal)\b',
        r'\b(Pen/?Strep|Penicillin|Primocin|Gentamicin)\b',
    ]

    for pattern in presence_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            canonical = normalize_factor_name(match)
            if canonical not in found_factors:
                found_factors[canonical] = 'Present'

    return found_factors


def discover_all_factors(conn: sqlite3.Connection) -> Set[str]:
    """
    Phase 1: Scan all records to discover unique factors.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT case_id, media_raw_text FROM media_protocols WHERE LENGTH(media_raw_text) > 100")

    all_factors = set()

    for case_id, text in cursor.fetchall():
        factors = extract_factors_from_text(text)
        all_factors.update(factors.keys())

    return all_factors


def create_dynamic_table(conn: sqlite3.Connection, factors: Set[str]):
    """
    Create media_factors_structured table with dynamic columns.
    """
    cursor = conn.cursor()

    # Drop existing table
    cursor.execute("DROP TABLE IF EXISTS media_factors_structured")

    # Build column definitions
    columns = ['case_id TEXT PRIMARY KEY']
    for factor in sorted(factors):
        # Sanitize column name
        col_name = re.sub(r'[^\w]', '_', factor)
        columns.append(f"{col_name} TEXT")

    # Create table
    create_sql = f"CREATE TABLE media_factors_structured ({', '.join(columns)})"
    cursor.execute(create_sql)
    conn.commit()

    return sorted(factors)


def populate_table(conn: sqlite3.Connection, factor_columns: list):
    """
    Phase 2: Extract factors for each case and populate table.
    """
    cursor = conn.cursor()

    # Get all records
    cursor.execute("SELECT case_id, media_raw_text FROM media_protocols WHERE LENGTH(media_raw_text) > 100")
    records = cursor.fetchall()

    inserted = 0
    for case_id, text in records:
        factors = extract_factors_from_text(text)

        # Build INSERT statement
        columns = ['case_id']
        values = [case_id]
        placeholders = ['?']

        for col in factor_columns:
            columns.append(col)
            placeholders.append('?')
            values.append(factors.get(col))  # None if not found

        sql = f"INSERT OR REPLACE INTO media_factors_structured ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        cursor.execute(sql, values)
        inserted += 1

    conn.commit()
    return inserted


def main():
    print("=" * 60)
    print("MEDIA FACTORS STRUCTURED EXTRACTION")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # Phase 1: Discovery
    print("\n[Phase 1] Discovering all unique factors...")
    all_factors = discover_all_factors(conn)
    print(f"   Found {len(all_factors)} unique factors:")
    for f in sorted(all_factors):
        print(f"     - {f}")

    # Phase 2: Create table
    print(f"\n[Phase 2] Creating dynamic table...")
    factor_columns = create_dynamic_table(conn, all_factors)
    print(f"   Created table with {len(factor_columns) + 1} columns")

    # Phase 3: Populate
    print(f"\n[Phase 3] Extracting and populating...")
    count = populate_table(conn, factor_columns)
    print(f"   Populated {count} records")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    cursor = conn.cursor()

    # Show table structure
    cursor.execute("PRAGMA table_info(media_factors_structured)")
    cols = cursor.fetchall()
    print(f"\nTable columns ({len(cols)}):")
    for c in cols[:15]:
        print(f"   {c[1]}: {c[2]}")
    if len(cols) > 15:
        print(f"   ... and {len(cols) - 15} more")

    # Show sample data
    print("\nSample records (3 cases):")
    cursor.execute("SELECT * FROM media_factors_structured LIMIT 3")
    rows = cursor.fetchall()
    col_names = [c[1] for c in cols]

    for row in rows:
        print(f"\n  Case: {row[0]}")
        for i, val in enumerate(row[1:], 1):
            if val:
                print(f"    {col_names[i]}: {val}")

    # Stats
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    for col in factor_columns[:10]:
        cursor.execute(f"SELECT COUNT(*) FROM media_factors_structured WHERE {col} IS NOT NULL")
        cnt = cursor.fetchone()[0]
        print(f"   {col}: {cnt} records")

    conn.close()
    print("\n[DONE] Table media_factors_structured created successfully!")


if __name__ == "__main__":
    main()
