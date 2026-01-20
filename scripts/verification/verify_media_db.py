#!/usr/bin/env python3
"""Verify database integrity for media_protocols table."""

import sqlite3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH

def verify():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 60)
    print("DATABASE INTEGRITY CHECK")
    print("=" * 60)

    # 1. Join Check
    cursor.execute("""
        SELECT COUNT(*)
        FROM patients_metadata p
        JOIN media_protocols m ON p.case_id = m.case_id
    """)
    join_count = cursor.fetchone()[0]
    print(f"\n1. JOIN CHECK")
    print(f"   Matching records (patients_metadata + media_protocols): {join_count}")

    # 2. Integrity Check - orphaned case_ids
    cursor.execute("""
        SELECT m.case_id
        FROM media_protocols m
        LEFT JOIN patients_metadata p ON m.case_id = p.case_id
        WHERE p.case_id IS NULL
    """)
    orphans = cursor.fetchall()
    print(f"\n2. INTEGRITY CHECK")
    if orphans:
        print(f"   WARNING: {len(orphans)} orphaned case_ids in media_protocols:")
        for o in orphans[:5]:
            print(f"     - {o[0]}")
    else:
        print("   All case_ids in media_protocols exist in patients_metadata")

    # 3. Visual Verification - 3 samples
    cursor.execute("""
        SELECT p.model_name, m.source_url, SUBSTR(m.media_raw_text, 1, 200)
        FROM patients_metadata p
        JOIN media_protocols m ON p.case_id = m.case_id
        WHERE m.media_raw_text IS NOT NULL AND m.media_raw_text != ''
        LIMIT 3
    """)
    samples = cursor.fetchall()
    print(f"\n3. VISUAL VERIFICATION (3 samples)")
    for i, (model, url, text) in enumerate(samples, 1):
        print(f"\n   Sample {i}:")
        print(f"   Model: {model}")
        print(f"   URL: {url}")
        print(f"   Text: {text}...")

    # 4. Duplicate Check
    cursor.execute("""
        SELECT case_id, COUNT(*) as cnt
        FROM media_protocols
        GROUP BY case_id
        HAVING cnt > 1
    """)
    duplicates = cursor.fetchall()
    print(f"\n4. DUPLICATE CHECK")
    if duplicates:
        print(f"   WARNING: {len(duplicates)} duplicate case_ids found:")
        for d in duplicates[:5]:
            print(f"     - {d[0]}: {d[1]} occurrences")
    else:
        print("   No duplicates - case_id is unique in media_protocols")

    # Summary stats
    cursor.execute("SELECT COUNT(*) FROM media_protocols WHERE media_raw_text != ''")
    with_text = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM media_protocols WHERE media_raw_text = ''")
    empty = cursor.fetchone()[0]

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Records with content: {with_text}")
    print(f"Records empty/failed: {empty}")
    print(f"Total in media_protocols: {with_text + empty}")

    conn.close()

if __name__ == "__main__":
    verify()
