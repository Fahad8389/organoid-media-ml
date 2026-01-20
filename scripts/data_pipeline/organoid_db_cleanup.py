#!/usr/bin/env python3
"""
Organoid Database Cleanup & Master Dataset Creation
Fixes data quality issues and creates unified master_dataset_v2
"""

import sqlite3
import csv
import re
from pathlib import Path

# Configuration - use centralized paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH, TSV_PATH

# Top 50 genes by case coverage (from mutations_with_vaf analysis)
TOP_50_GENES = [
    'TP53', 'TTN', 'MUC16', 'KRAS', 'APC', 'SYNE1', 'KCNQ1', 'RYR2', 'OBSCN', 'FLG',
    'CSMD3', 'PCLO', 'LRP1B', 'ZFHX4', 'CSMD1', 'FAT4', 'FAT3', 'DNAH5', 'FSIP2', 'HYDIN',
    'RYR1', 'HMCN1', 'KMT2D', 'USH2A', 'RYR3', 'APOB', 'AHNAK2', 'CCDC168', 'ADGRV1', 'XIRP2',
    'DNAH11', 'PLEC', 'NEB', 'ARID1A', 'CSMD2', 'RP1', 'SPTA1', 'MUC12', 'DCHS2', 'EYS',
    'PIK3CA', 'DNAH3', 'DNAH9', 'MACF1', 'TNXB', 'COL6A5', 'DNAH7', 'LRP2', 'PIEZO2', 'PKHD1L1'
]

# ATCC Formulation Components (known concentrations)
# Only include components where we have actual concentration data
ATCC_FORMULATIONS = {
    1: {  # Colorectal/Intestinal
        'egf': '50 ng/mL',
        'n_acetyl_cysteine': '1 mM',
        'a83_01': '500 nM',
        'sb202190': '10 uM',
        'y27632': '10 uM',  # first 2-3 days
        'noggin': None,  # Present but no concentration
        'gastrin': None,
        'nicotinamide': None,
        'b27': None,
        'n2': None,
    },
    3: {  # Pancreatic/Gastric - similar to #1
        'egf': '50 ng/mL',
        'n_acetyl_cysteine': '1 mM',
        'a83_01': '500 nM',
        'y27632': '10 uM',
        'noggin': None,
        'gastrin': None,
        'nicotinamide': None,
        'b27': None,
        'n2': None,
        'fgf10': None,
    },
    4: {  # Brain
        'egf': '20 ng/mL',
        'fgf2': '20 ng/mL',
        'b27': None,
        'n2': None,
        'heparin': None,
    },
    5: {  # Lung/Esophageal
        'egf': '50 ng/mL',
        'n_acetyl_cysteine': '1 mM',
        'a83_01': '500 nM',
        'sb202190': '10 uM',
        'y27632': '10 uM',
        'noggin': None,
        'rspondin': None,
        'fgf7': None,
        'fgf10': None,
    },
}

# TSV Column Mapping (original name -> SQL name)
TSV_COLUMN_MAPPING = [
    ('Name', 'model_name'),
    ('Primary Site', 'primary_site'),
    ('Clinical Tumor Diagnosis', 'clinical_tumor_diagnosis'),
    ('Histological Subtype', 'histological_subtype'),
    ('Tissue Status', 'tissue_status'),
    ('Acquisition Site', 'acquisition_site'),
    ('Gender', 'gender'),
    ('Race', 'race'),
    ('Age At Acquisition (Years)', 'age_at_acquisition_years'),
    ('Age At Diagnosis (Years)', 'age_at_diagnosis_years'),
    ('Disease Status', 'disease_status'),
    ('Vital Status', 'vital_status'),
    ('TNM Stage', 'tnm_stage'),
    ('Clinical Stage Grouping', 'clinical_stage_grouping'),
    ('Histological Grade', 'histological_grade'),
    ('Has Multiple Models', 'has_multiple_models'),
    ('Neoadjuvant Therapy', 'neoadjuvant_therapy'),
    ('Chemotherapeutic Drug List Available', 'chemo_drug_list_available'),
    ('Therapy', 'therapy'),
    ('Available Molecular Characterizations', 'molecular_characterizations'),
    ('Link To Distributor', 'distributor'),
    ('Model Type', 'model_type'),
    ('Licensing Required For Commercial Use', 'licensing_required'),
    ('Expansion Status', 'expansion_status'),
    ('# Mutated Genes', 'mutated_genes_1'),  # First occurrence
    ('# Mutated Genes', 'mutated_genes_2'),  # Second occurrence (duplicate column)
    ('# Research Somatic Variants', 'research_somatic_variants'),
    ('# Clinical Variants', 'clinical_variants'),
    ('# Histo-pathological Biomarkers', 'histopath_biomarkers'),
    ('Split Ratio', 'split_ratio'),
    ('Doubling Time', 'doubling_time'),
    ('Time to Split', 'time_to_split'),
    ('Date Created', 'date_created'),
    ('Date Of Availability', 'date_of_availability'),
    ('Link to Model Details', 'link_model_details'),
    ('Link To Sequencing Data', 'link_sequencing_data'),
    ('Link To Masked Somatic MAF', 'link_maf'),
    ('Link To Proteomics Data', 'link_proteomics'),
]


def extract_case_id(model_name: str) -> str:
    """Extract case_id from model name (e.g., HCM-CSHL-0056-C18 -> HCM-CSHL-0056)"""
    parts = model_name.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return model_name


def sanitize_value(value: str) -> str | None:
    """Convert --, empty strings to NULL"""
    if value is None:
        return None
    value = value.strip()
    if value in ('--', '', '-'):
        return None
    return value


def step1_import_clinical_data(conn: sqlite3.Connection):
    """Step 1: Import all 38 columns from TSV into clinical_data_full table

    IMPORTANT: model_name is the PRIMARY KEY (unique identifier for each row).
    case_id is a DERIVED column used for joining with genomic/media data.
    Multiple models can share the same case_id (same patient, different samples).
    """
    print("=" * 60)
    print("STEP 1: Importing clinical data from TSV")
    print("=" * 60)

    cursor = conn.cursor()

    # Drop existing table if exists
    cursor.execute("DROP TABLE IF EXISTS clinical_data_full")

    # Create schema - model_name is PK, case_id is derived for joins
    cursor.execute("""
        CREATE TABLE clinical_data_full (
            model_name TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            primary_site TEXT,
            clinical_tumor_diagnosis TEXT,
            histological_subtype TEXT,
            tissue_status TEXT,
            acquisition_site TEXT,
            gender TEXT,
            race TEXT,
            age_at_acquisition_years INTEGER,
            age_at_diagnosis_years INTEGER,
            disease_status TEXT,
            vital_status TEXT,
            tnm_stage TEXT,
            clinical_stage_grouping TEXT,
            histological_grade TEXT,
            has_multiple_models TEXT,
            neoadjuvant_therapy TEXT,
            chemo_drug_list_available TEXT,
            therapy TEXT,
            molecular_characterizations TEXT,
            distributor TEXT,
            model_type TEXT,
            licensing_required TEXT,
            expansion_status TEXT,
            mutated_genes_1 INTEGER,
            mutated_genes_2 INTEGER,
            research_somatic_variants INTEGER,
            clinical_variants INTEGER,
            histopath_biomarkers INTEGER,
            split_ratio TEXT,
            doubling_time TEXT,
            time_to_split TEXT,
            date_created TEXT,
            date_of_availability TEXT,
            link_model_details TEXT,
            link_sequencing_data TEXT,
            link_maf TEXT,
            link_proteomics TEXT
        )
    """)

    # Read TSV and insert data
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)

        # Map column positions (handle duplicate "# Mutated Genes")
        col_positions = []
        mutated_genes_count = 0
        for i, col in enumerate(header):
            if col == '# Mutated Genes':
                if mutated_genes_count == 0:
                    col_positions.append((i, 'mutated_genes_1'))
                else:
                    col_positions.append((i, 'mutated_genes_2'))
                mutated_genes_count += 1
            else:
                # Find matching SQL column name
                for tsv_col, sql_col in TSV_COLUMN_MAPPING:
                    if tsv_col == col:
                        col_positions.append((i, sql_col))
                        break

        rows_inserted = 0
        for row in reader:
            # Extract case_id from model_name (for genomic/media joins)
            model_name = row[0]
            case_id = extract_case_id(model_name)

            # Build values dict - model_name first (PK), then case_id (for joins)
            values = {'model_name': model_name, 'case_id': case_id}
            for pos, sql_col in col_positions:
                if pos < len(row):
                    # Skip model_name since we already added it
                    if sql_col == 'model_name':
                        continue
                    val = sanitize_value(row[pos])
                    # Convert integer columns
                    if sql_col in ('age_at_acquisition_years', 'age_at_diagnosis_years',
                                   'mutated_genes_1', 'mutated_genes_2',
                                   'research_somatic_variants', 'clinical_variants',
                                   'histopath_biomarkers'):
                        try:
                            val = int(val) if val else None
                        except (ValueError, TypeError):
                            val = None
                    values[sql_col] = val

            # Insert row - use INSERT (not REPLACE) to ensure all 660 rows are kept
            columns = list(values.keys())
            placeholders = ', '.join(['?' for _ in columns])
            sql = f"INSERT INTO clinical_data_full ({', '.join(columns)}) VALUES ({placeholders})"
            cursor.execute(sql, [values.get(c) for c in columns])
            rows_inserted += 1

    conn.commit()

    # Validation
    cursor.execute("SELECT COUNT(*) FROM clinical_data_full")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM clinical_data_full WHERE case_id IS NULL")
    null_count = cursor.fetchone()[0]

    print(f"  Rows inserted: {rows_inserted}")
    print(f"  Total rows in table: {count}")
    print(f"  Rows with NULL case_id: {null_count}")
    print(f"  VALIDATION: {'PASS' if count == 660 and null_count == 0 else 'FAIL'}")


def step2a_create_formulations_lookup(conn: sqlite3.Connection):
    """Step 2a: Create ATCC formulations lookup table"""
    print("\n" + "=" * 60)
    print("STEP 2a: Creating ATCC formulations lookup table")
    print("=" * 60)

    cursor = conn.cursor()

    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS atcc_formulations")
    cursor.execute("""
        CREATE TABLE atcc_formulations (
            formulation_number INTEGER,
            factor_name TEXT,
            concentration TEXT,
            PRIMARY KEY (formulation_number, factor_name)
        )
    """)

    # Insert formulation data
    rows_inserted = 0
    for form_num, factors in ATCC_FORMULATIONS.items():
        for factor_name, concentration in factors.items():
            cursor.execute("""
                INSERT INTO atcc_formulations (formulation_number, factor_name, concentration)
                VALUES (?, ?, ?)
            """, (form_num, factor_name, concentration))
            rows_inserted += 1

    conn.commit()
    print(f"  Rows inserted: {rows_inserted}")


def step2b_extract_media_factors(conn: sqlite3.Connection):
    """Step 2b: Re-extract media factors with improved regex patterns"""
    print("\n" + "=" * 60)
    print("STEP 2b: Extracting media factors with improved regex")
    print("=" * 60)

    cursor = conn.cursor()

    # Drop and create new structured table
    cursor.execute("DROP TABLE IF EXISTS media_factors_structured_v2")
    cursor.execute("""
        CREATE TABLE media_factors_structured_v2 (
            case_id TEXT PRIMARY KEY,
            formulation_number INTEGER,
            egf TEXT,
            noggin TEXT,
            gastrin TEXT,
            n_acetyl_cysteine TEXT,
            nicotinamide TEXT,
            a83_01 TEXT,
            sb202190 TEXT,
            b27 TEXT,
            n2 TEXT,
            y27632 TEXT,
            fgf2 TEXT,
            fgf7 TEXT,
            fgf10 TEXT,
            rspondin TEXT,
            wnt3a TEXT,
            heparin TEXT,
            cholera_toxin TEXT,
            hydrocortisone TEXT,
            insulin TEXT,
            prostaglandin_e2 TEXT,
            primocin TEXT,
            forskolin TEXT,
            heregulin TEXT,
            neuregulin TEXT
        )
    """)

    # Improved regex patterns (handle no-space variants like "10uM")
    patterns = {
        'egf': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\s*(?:of\s+)?(?:recombinant\s+)?(?:human\s+)?EGF',
            r'EGF\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\)?',
        ],
        'y27632': [
            r'(\d+(?:\.\d+)?)\s*(uM|µM|um|μM)\s*(?:ROCK\s+)?(?:Inhibitor\s+)?Y-?27632',
            r'Y-?27632\s*\(?(\d+(?:\.\d+)?)\s*(uM|µM|um|μM)\)?',
            r'Include\s+(\d+(?:\.\d+)?)\s*(uM|µM|um|μM)\s+ROCK\s+Inhibitor\s+Y-?27632',
        ],
        'a83_01': [
            r'(\d+(?:\.\d+)?)\s*(nM|uM|µM)\s*A[-\s]?83[-\s]?01',
            r'A[-\s]?83[-\s]?01\s*\(?(\d+(?:\.\d+)?)\s*(nM|uM|µM)\)?',
        ],
        'sb202190': [
            r'(\d+(?:\.\d+)?)\s*(uM|µM|um|μM)\s*SB[-\s]?202190',
            r'SB[-\s]?202190\s*\(?(\d+(?:\.\d+)?)\s*(uM|µM|um|μM)\)?',
        ],
        'n_acetyl_cysteine': [
            r'(\d+(?:\.\d+)?)\s*(mM)\s*N-?[Aa]cetyl-?[Cc]ysteine',
            r'N-?[Aa]cetyl-?[Cc]ysteine\s*\(?(\d+(?:\.\d+)?)\s*(mM)\)?',
        ],
        'cholera_toxin': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\s*(?:of\s+)?[Cc]holera\s*[Tt]oxin',
            r'[Cc]holera\s*[Tt]oxin\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\)?',
        ],
        'fgf2': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\s*(?:of\s+)?(?:recombinant\s+)?(?:human\s+)?(?:bFGF|FGF-?2)',
            r'(?:bFGF|FGF-?2)\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\)?',
        ],
        'fgf7': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\s*(?:of\s+)?(?:recombinant\s+)?(?:human\s+)?FGF-?7',
            r'FGF-?7\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\)?',
        ],
        'fgf10': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\s*(?:of\s+)?(?:recombinant\s+)?(?:human\s+)?FGF-?10',
            r'FGF-?10\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml)\)?',
        ],
        'heregulin': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|nM)\s*(?:of\s+)?[Hh]eregulin',
            r'[Hh]eregulin\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|nM)\)?',
        ],
        'neuregulin': [
            r'(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|nM)\s*(?:of\s+)?[Nn]euregulin',
            r'[Nn]euregulin\s*\(?(\d+(?:\.\d+)?)\s*(ng/mL|ng/ml|nM)\)?',
        ],
        'hydrocortisone': [
            r'(\d+(?:\.\d+)?)\s*(ug/mL|µg/mL|ng/mL|uM)\s*[Hh]ydrocortisone',
            r'[Hh]ydrocortisone\s*\(?(\d+(?:\.\d+)?)\s*(ug/mL|µg/mL|ng/mL|uM)\)?',
        ],
        'insulin': [
            r'(\d+(?:\.\d+)?)\s*(ug/mL|µg/mL|ng/mL)\s*[Ii]nsulin',
            r'[Ii]nsulin\s*\(?(\d+(?:\.\d+)?)\s*(ug/mL|µg/mL|ng/mL)\)?',
        ],
        'forskolin': [
            r'(\d+(?:\.\d+)?)\s*(uM|µM|um)\s*[Ff]orskolin',
            r'[Ff]orskolin\s*\(?(\d+(?:\.\d+)?)\s*(uM|µM|um)\)?',
        ],
    }

    # Formulation detection pattern
    formulation_pattern = r'Organoid\s+(?:Media\s+)?Formulation\s+#(\d+)'

    # Get all media protocols
    cursor.execute("SELECT case_id, media_raw_text FROM media_protocols WHERE media_raw_text IS NOT NULL")
    records = cursor.fetchall()

    rows_inserted = 0
    y27632_with_conc = 0

    for case_id, raw_text in records:
        if not raw_text:
            continue

        factors = {
            'case_id': case_id,
            'formulation_number': None,
            'egf': None,
            'noggin': None,
            'gastrin': None,
            'n_acetyl_cysteine': None,
            'nicotinamide': None,
            'a83_01': None,
            'sb202190': None,
            'b27': None,
            'n2': None,
            'y27632': None,
            'fgf2': None,
            'fgf7': None,
            'fgf10': None,
            'rspondin': None,
            'wnt3a': None,
            'heparin': None,
            'cholera_toxin': None,
            'hydrocortisone': None,
            'insulin': None,
            'prostaglandin_e2': None,
            'primocin': None,
            'forskolin': None,
            'heregulin': None,
            'neuregulin': None,
        }

        # Detect formulation number
        form_match = re.search(formulation_pattern, raw_text, re.IGNORECASE)
        if form_match:
            factors['formulation_number'] = int(form_match.group(1))

        # Extract concentrations using regex patterns
        for factor_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    unit = match.group(2)
                    factors[factor_name] = f"{value} {unit}"
                    break

        # If formulation detected, inject known concentrations (only if regex didn't find)
        if factors['formulation_number'] in ATCC_FORMULATIONS:
            form_data = ATCC_FORMULATIONS[factors['formulation_number']]
            for factor_name, concentration in form_data.items():
                if factors.get(factor_name) is None and concentration is not None:
                    factors[factor_name] = concentration

        # Track Y27632 extraction
        if factors['y27632'] and 'uM' in factors['y27632'].lower():
            y27632_with_conc += 1

        # Insert row
        columns = list(factors.keys())
        placeholders = ', '.join(['?' for _ in columns])
        sql = f"INSERT OR REPLACE INTO media_factors_structured_v2 ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(sql, [factors.get(c) for c in columns])
        rows_inserted += 1

    conn.commit()

    # Validation
    cursor.execute("SELECT COUNT(*) FROM media_factors_structured_v2")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM media_factors_structured_v2 WHERE y27632 LIKE '%uM%' OR y27632 LIKE '%µM%'")
    y27632_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM media_factors_structured_v2 WHERE formulation_number IS NOT NULL")
    form_count = cursor.fetchone()[0]

    print(f"  Rows inserted: {rows_inserted}")
    print(f"  Records with formulation number: {form_count}")
    print(f"  Records with Y-27632 concentration: {y27632_count}")
    print(f"  VALIDATION: Y-27632 extraction {'IMPROVED' if y27632_count > 0 else 'NEEDS REVIEW'}")


def step3_create_mutations_pivot(conn: sqlite3.Connection):
    """Step 3: Create wide-format mutations pivot table with Top 50 genes

    This table is keyed by case_id (patient level, not model level).
    When joined to clinical_data_full, multiple models sharing the same
    case_id will all receive the same genomic data.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating mutations pivot table (Top 50 genes)")
    print("=" * 60)

    cursor = conn.cursor()

    # Get UNIQUE case_ids from clinical_data_full (632 patients, not 660 models)
    cursor.execute("SELECT DISTINCT case_id FROM clinical_data_full")
    all_cases = set(row[0] for row in cursor.fetchall())

    # Get cases that have mutation data
    cursor.execute("SELECT DISTINCT case_id FROM mutations_with_vaf")
    cases_with_mutations = set(row[0] for row in cursor.fetchall())

    print(f"  Total clinical cases: {len(all_cases)}")
    print(f"  Cases with mutation data: {len(cases_with_mutations)}")

    # Build dynamic column definitions
    gene_columns = ', '.join([f"{gene}_vaf REAL" for gene in TOP_50_GENES])

    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS mutations_top50_pivot")
    cursor.execute(f"""
        CREATE TABLE mutations_top50_pivot (
            case_id TEXT PRIMARY KEY,
            has_sequencing_data INTEGER,
            {gene_columns}
        )
    """)

    # Get max VAF for each gene/case combination
    cursor.execute("""
        SELECT case_id, hugo_symbol, MAX(vaf) as max_vaf
        FROM mutations_with_vaf
        WHERE hugo_symbol IN ({})
        GROUP BY case_id, hugo_symbol
    """.format(','.join(['?' for _ in TOP_50_GENES])), TOP_50_GENES)

    # Build mutation map: case_id -> {gene: vaf}
    mutation_map = {}
    for case_id, gene, vaf in cursor.fetchall():
        if case_id not in mutation_map:
            mutation_map[case_id] = {}
        mutation_map[case_id][gene] = vaf

    # Insert rows for all cases
    rows_inserted = 0
    for case_id in all_cases:
        has_sequencing = 1 if case_id in cases_with_mutations else 0

        # Build gene values
        gene_values = []
        for gene in TOP_50_GENES:
            if case_id not in cases_with_mutations:
                # No sequencing data = NULL
                gene_values.append(None)
            elif gene in mutation_map.get(case_id, {}):
                # Has mutation = VAF value
                gene_values.append(mutation_map[case_id][gene])
            else:
                # Has sequencing but no mutation = 0 (wild-type)
                gene_values.append(0)

        # Insert row
        placeholders = ', '.join(['?' for _ in range(len(TOP_50_GENES) + 2)])
        gene_cols = ', '.join([f"{gene}_vaf" for gene in TOP_50_GENES])
        sql = f"INSERT INTO mutations_top50_pivot (case_id, has_sequencing_data, {gene_cols}) VALUES ({placeholders})"
        cursor.execute(sql, [case_id, has_sequencing] + gene_values)
        rows_inserted += 1

    conn.commit()

    # Validation
    cursor.execute("SELECT COUNT(*) FROM mutations_top50_pivot")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM mutations_top50_pivot WHERE has_sequencing_data = 1")
    with_seq = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM mutations_top50_pivot WHERE TP53_vaf IS NOT NULL AND TP53_vaf > 0")
    tp53_mutated = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM mutations_top50_pivot WHERE TP53_vaf = 0")
    tp53_wildtype = cursor.fetchone()[0]

    print(f"  Rows inserted: {rows_inserted}")
    print(f"  Unique case_ids (patients): {total}")
    print(f"  Cases with sequencing: {with_seq}")
    print(f"  TP53 mutated cases: {tp53_mutated}")
    print(f"  TP53 wild-type cases: {tp53_wildtype}")
    print(f"  NOTE: This table has {total} unique patients, maps to 660 models via JOIN")


def step4_create_master_dataset(conn: sqlite3.Connection):
    """Step 4: Create master_dataset_v2 by joining all tables

    Join strategy:
    - clinical_data_full: 660 rows (model_name is PK, case_id for joins)
    - media_factors_structured_v2: keyed by case_id (patient level)
    - mutations_top50_pivot: keyed by case_id (patient level)

    Result: 660 rows - all models, with shared genomic/media data for
    models from the same patient (same case_id).
    """
    print("\n" + "=" * 60)
    print("STEP 4: Creating master_dataset_v2")
    print("=" * 60)

    cursor = conn.cursor()

    # Build gene column selections
    gene_cols = ', '.join([f"m.{gene}_vaf" for gene in TOP_50_GENES])

    # Drop and create master table
    cursor.execute("DROP TABLE IF EXISTS master_dataset_v2")

    # Build the CREATE TABLE with all columns
    # model_name is PK, case_id is for reference/joins
    cursor.execute(f"""
        CREATE TABLE master_dataset_v2 AS
        SELECT
            c.model_name,
            c.case_id,
            c.primary_site,
            c.clinical_tumor_diagnosis,
            c.histological_subtype,
            c.tissue_status,
            c.acquisition_site,
            c.gender,
            c.race,
            c.age_at_acquisition_years,
            c.age_at_diagnosis_years,
            c.disease_status,
            c.vital_status,
            c.tnm_stage,
            c.clinical_stage_grouping,
            c.histological_grade,
            c.has_multiple_models,
            c.neoadjuvant_therapy,
            c.chemo_drug_list_available,
            c.therapy,
            c.molecular_characterizations,
            c.distributor,
            c.model_type,
            c.licensing_required,
            c.expansion_status,
            c.mutated_genes_1,
            c.mutated_genes_2,
            c.research_somatic_variants,
            c.clinical_variants,
            c.histopath_biomarkers,
            c.split_ratio,
            c.doubling_time,
            c.time_to_split,
            c.date_created,
            c.date_of_availability,
            c.link_model_details,
            c.link_sequencing_data,
            c.link_maf,
            c.link_proteomics,
            -- Media factors
            med.formulation_number,
            med.egf,
            med.noggin,
            med.gastrin,
            med.n_acetyl_cysteine,
            med.nicotinamide,
            med.a83_01,
            med.sb202190,
            med.b27,
            med.n2,
            med.y27632,
            med.fgf2,
            med.fgf7,
            med.fgf10,
            med.rspondin,
            med.wnt3a,
            med.heparin,
            med.cholera_toxin,
            med.hydrocortisone,
            med.insulin,
            med.prostaglandin_e2,
            med.primocin,
            med.forskolin,
            med.heregulin,
            med.neuregulin,
            -- Genomic data
            m.has_sequencing_data,
            {gene_cols},
            -- Computed fields
            CASE WHEN m.TP53_vaf IS NOT NULL AND m.TP53_vaf > 0 THEN 1 ELSE 0 END as has_tp53_mutation,
            CASE WHEN m.KRAS_vaf IS NOT NULL AND m.KRAS_vaf > 0 THEN 1 ELSE 0 END as has_kras_mutation,
            CASE WHEN m.APC_vaf IS NOT NULL AND m.APC_vaf > 0 THEN 1 ELSE 0 END as has_apc_mutation,
            -- Mutation count (top 10 genes with mutations)
            (
                CASE WHEN m.TP53_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.TTN_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.MUC16_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.KRAS_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.APC_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.SYNE1_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.KCNQ1_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.RYR2_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.OBSCN_vaf > 0 THEN 1 ELSE 0 END +
                CASE WHEN m.FLG_vaf > 0 THEN 1 ELSE 0 END
            ) as mutation_count_top10,
            -- Data completeness score
            ROUND(
                (
                    CASE WHEN c.age_at_acquisition_years IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN c.gender IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN c.race IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN c.tnm_stage IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN c.histological_grade IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN med.formulation_number IS NOT NULL THEN 1.0 ELSE 0.0 END +
                    CASE WHEN m.has_sequencing_data = 1 THEN 1.0 ELSE 0.0 END
                ) / 7.0,
                2
            ) as data_completeness_score
        FROM clinical_data_full c
        LEFT JOIN media_factors_structured_v2 med ON c.case_id = med.case_id
        LEFT JOIN mutations_top50_pivot m ON c.case_id = m.case_id
    """)

    conn.commit()

    # Validation
    cursor.execute("SELECT COUNT(*) FROM master_dataset_v2")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(CASE WHEN formulation_number IS NOT NULL THEN 1 ELSE 0 END) FROM master_dataset_v2")
    has_media = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(CASE WHEN TP53_vaf IS NOT NULL THEN 1 ELSE 0 END) FROM master_dataset_v2")
    has_tp53 = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(data_completeness_score) FROM master_dataset_v2")
    avg_completeness = cursor.fetchone()[0]

    print(f"  Total rows: {total}")
    print(f"  Rows with media data: {has_media}")
    print(f"  Rows with TP53 data: {has_tp53}")
    print(f"  Average data completeness: {avg_completeness:.2%}")
    print(f"  VALIDATION: {'PASS' if total == 660 else 'FAIL'} (expected 660 rows)")


def run_verification_queries(conn: sqlite3.Connection):
    """Run verification queries to validate the data"""
    print("\n" + "=" * 60)
    print("VERIFICATION QUERIES")
    print("=" * 60)

    cursor = conn.cursor()

    # Step 1 validation
    cursor.execute("SELECT COUNT(*) FROM clinical_data_full")
    count = cursor.fetchone()[0]
    print(f"\n1. clinical_data_full row count: {count} (expected: 660)")

    # Step 2 validation
    cursor.execute("SELECT COUNT(*) FROM media_factors_structured_v2 WHERE y27632 LIKE '%uM%' OR y27632 LIKE '%µM%'")
    y27632_count = cursor.fetchone()[0]
    print(f"2. Y-27632 with concentration: {y27632_count}")

    # Formulation distribution
    cursor.execute("""
        SELECT formulation_number, COUNT(*) as cnt
        FROM media_factors_structured_v2
        WHERE formulation_number IS NOT NULL
        GROUP BY formulation_number
    """)
    print("3. Formulation distribution:")
    for row in cursor.fetchall():
        print(f"   Formulation #{row[0]}: {row[1]} records")

    # Step 3 validation
    cursor.execute("SELECT COUNT(*) FROM mutations_top50_pivot")
    pivot_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT case_id) FROM clinical_data_full")
    unique_cases = cursor.fetchone()[0]
    print(f"4. mutations_top50_pivot row count: {pivot_count} (expected: {unique_cases} unique patients)")

    # Step 4 validation
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN formulation_number IS NOT NULL THEN 1 ELSE 0 END) as has_media,
            SUM(CASE WHEN TP53_vaf IS NOT NULL THEN 1 ELSE 0 END) as has_tp53,
            SUM(CASE WHEN has_tp53_mutation = 1 THEN 1 ELSE 0 END) as tp53_mutated,
            SUM(CASE WHEN has_kras_mutation = 1 THEN 1 ELSE 0 END) as kras_mutated,
            AVG(data_completeness_score) as avg_completeness
        FROM master_dataset_v2
    """)
    row = cursor.fetchone()
    print(f"5. master_dataset_v2 summary:")
    print(f"   Total rows: {row[0]}")
    print(f"   Has media data: {row[1]}")
    print(f"   Has TP53 data: {row[2]}")
    print(f"   TP53 mutated: {row[3]}")
    print(f"   KRAS mutated: {row[4]}")
    print(f"   Avg completeness: {row[5]:.2%}")

    # Sample data
    print("\n6. Sample master_dataset_v2 rows:")
    cursor.execute("""
        SELECT model_name, case_id, primary_site, formulation_number, has_sequencing_data,
               TP53_vaf, KRAS_vaf, has_tp53_mutation, data_completeness_score
        FROM master_dataset_v2
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"   {row}")

    # Verify multi-model patients get same genomic data
    print("\n7. Multi-model patient verification (same case_id, different models):")
    cursor.execute("""
        SELECT model_name, case_id, TP53_vaf, KRAS_vaf
        FROM master_dataset_v2
        WHERE case_id IN (
            SELECT case_id FROM master_dataset_v2 GROUP BY case_id HAVING COUNT(*) > 1
        )
        ORDER BY case_id, model_name
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"   {row}")


def main():
    print("=" * 60)
    print("ORGANOID DATABASE CLEANUP & MASTER DATASET CREATION")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"TSV Source: {TSV_PATH}")
    print()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    try:
        # Execute all steps
        step1_import_clinical_data(conn)
        step2a_create_formulations_lookup(conn)
        step2b_extract_media_factors(conn)
        step3_create_mutations_pivot(conn)
        step4_create_master_dataset(conn)
        run_verification_queries(conn)

        print("\n" + "=" * 60)
        print("CLEANUP COMPLETE")
        print("=" * 60)
        print("New tables created:")
        print("  - clinical_data_full (660 rows, 39 columns)")
        print("  - atcc_formulations (lookup table)")
        print("  - media_factors_structured_v2 (structured media data)")
        print("  - mutations_top50_pivot (660 rows, 52 columns)")
        print("  - master_dataset_v2 (660 rows, unified dataset)")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
