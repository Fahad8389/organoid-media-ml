#!/usr/bin/env python3
"""Verify the link between patients_metadata and mutations tables."""

import sqlite3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH

def verify_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # JOIN query to find matching records
    cursor.execute("""
        SELECT p.case_id, p.distributor, p.model_name, m.hugo_symbol
        FROM patients_metadata p
        JOIN mutations m ON p.case_id = m.case_id
        LIMIT 5
    """)

    results = cursor.fetchall()

    if results:
        print("SUCCESS: Tables are linked!")
        print(f"\nFound {len(results)} sample records:\n")
        for row in results:
            case_id, distributor, model_name, hugo_symbol = row
            print(f"  case_id: {case_id}")
            print(f"  distributor: {distributor}")
            print(f"  model_name: {model_name}")
            print(f"  hugo_symbol: {hugo_symbol}")
            print()
    else:
        print("No matching records found between tables.")

    # Also show counts
    cursor.execute("SELECT COUNT(DISTINCT case_id) FROM patients_metadata")
    metadata_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT case_id) FROM mutations")
    mutation_cases = cursor.fetchone()[0]

    print(f"Summary:")
    print(f"  Unique case_ids in patients_metadata: {metadata_cases}")
    print(f"  Unique case_ids in mutations: {mutation_cases}")

    conn.close()

if __name__ == "__main__":
    verify_tables()
