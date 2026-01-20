#!/usr/bin/env python3
"""
ATCC Media Formulation Scraper
==============================

Scrapes culture method, media formulation, and storage information
from ATCC product pages for organoid models.

Usage:
    python atcc_media_scraper.py

Output:
    - Populates media_protocols table in organoid_data.db
    - Creates scraper_log_YYYYMMDD_HHMMSS.log
"""

import sqlite3
import requests
import time
import random
import re
import logging
from datetime import datetime
from typing import Optional, Tuple, Set
from bs4 import BeautifulSoup

# =============================================================================
# CONFIGURATION
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DB_PATH, LOGS_DIR
LOG_FILE = str(LOGS_DIR / f"scraper_log_{datetime.now():%Y%m%d_%H%M%S}.log")

# ATCC URL pattern
ATCC_BASE_URL = "https://www.atcc.org/products/"

# Request settings
REQUEST_TIMEOUT = 30
MIN_DELAY = 1.5
MAX_DELAY = 3.0

# Realistic browser headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Boilerplate text patterns to remove (for ML cleaning)
BOILERPLATE_PATTERNS = [
    r'Copyright.*?\d{4}',
    r'All rights reserved\.?',
    r'Terms of [Uu]se',
    r'Privacy [Pp]olicy',
    r'Cookie [Pp]olicy',
    r'Contact [Uu]s',
    r'Sign up for.*?newsletter',
    r'Subscribe.*?updates',
    r'Follow us on',
    r'Share this',
    r'Print this page',
    r'Add to cart',
    r'Add to wishlist',
    r'Customer [Ss]ervice',
    r'\$[\d,]+\.\d{2}',  # Prices
    r'SKU:?\s*\w+',
    r'Item #:?\s*\w+',
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 60)
    logging.info("ATCC Media Scraper Started")
    logging.info(f"Log file: {LOG_FILE}")
    logging.info("=" * 60)


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def create_media_protocols_table(conn: sqlite3.Connection):
    """Create media_protocols table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media_protocols (
            case_id TEXT UNIQUE,
            media_raw_text TEXT,
            source_url TEXT
        )
    """)
    conn.commit()
    logging.info("media_protocols table ready")


def get_existing_case_ids(conn: sqlite3.Connection) -> Set[str]:
    """Get case_ids that have already been scraped."""
    cursor = conn.cursor()
    cursor.execute("SELECT case_id FROM media_protocols WHERE media_raw_text IS NOT NULL")
    return {row[0] for row in cursor.fetchall()}


def get_cases_to_scrape(conn: sqlite3.Connection) -> list:
    """
    Get unique (case_id, distributor) pairs to scrape.
    Returns list of tuples: (case_id, pdm_code)
    """
    cursor = conn.cursor()
    # Get unique distributor codes with their case_ids
    cursor.execute("""
        SELECT DISTINCT case_id, distributor
        FROM patients_metadata
        WHERE distributor IS NOT NULL
        AND distributor != ''
        AND distributor != '--'
    """)
    return cursor.fetchall()


def save_media_protocol(conn: sqlite3.Connection, case_id: str,
                        media_text: str, source_url: str):
    """Save scraped media protocol to database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO media_protocols (case_id, media_raw_text, source_url)
        VALUES (?, ?, ?)
    """, (case_id, media_text, source_url))
    conn.commit()


# =============================================================================
# TEXT CLEANING (ML-OPTIMIZED)
# =============================================================================

def clean_text_for_ml(raw_text: str) -> str:
    """
    Clean extracted text for ML training.

    - Removes HTML artifacts
    - Removes boilerplate (copyright, terms, etc.)
    - Normalizes whitespace
    - Keeps scientific content intact
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove common boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Normalize unicode characters
    text = text.replace('\u2019', "'")  # Smart quote
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u00b5', 'u')  # Micro sign to u (for µL -> uL)
    text = text.replace('\u00b0', ' degrees ')  # Degree sign
    text = text.replace('\u2122', '')  # Trademark
    text = text.replace('\u00ae', '')  # Registered

    # Normalize whitespace
    text = re.sub(r'\t+', ' ', text)  # Tabs to space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n +', '\n', text)  # Space after newline
    text = re.sub(r' +\n', '\n', text)  # Space before newline

    # Remove empty parentheses/brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    # Clean up punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])\s*([.,;:!?])', r'\1', text)  # Remove duplicate punctuation

    # Final strip
    text = text.strip()

    return text


# =============================================================================
# WEB SCRAPING
# =============================================================================

def construct_atcc_url(pdm_code: str) -> str:
    """Construct ATCC product URL from PDM code."""
    # Normalize the code (lowercase, handle variations)
    code = pdm_code.strip().lower()
    return f"{ATCC_BASE_URL}{code}"


def extract_handling_info(soup: BeautifulSoup) -> str:
    """
    Extract handling information, culture method, and media formulation.

    Targets specifically:
    - Handling information section
    - Complete Medium / Media Formulation
    - Handling procedure (seeding, splitting, etc.)
    - Storage Instructions
    """
    extracted_parts = []
    full_text = soup.get_text(separator='\n', strip=True)

    # Strategy 1: Extract the "Handling information" section using text markers
    # Look for text between "Handling information" and next major section
    handling_markers = [
        (r'Handling information\s*\n', r'\n(?:Quality control|References|Citations|Related products|You may also)'),
        (r'Unpacking and storage', r'\n(?:Quality control|References|Citations|Related products)'),
        (r'Complete medium', r'\n(?:Quality control|References|Citations|Cryopreservation medium|Related)'),
        (r'Handling procedure', r'\n(?:Cryopreservation|Quality control|References|Related)'),
    ]

    for start_pattern, end_pattern in handling_markers:
        match = re.search(
            f'{start_pattern}(.*?){end_pattern}',
            full_text,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            section = match.group(0)
            if len(section) > 100 and section not in extracted_parts:
                extracted_parts.append(section)

    # Strategy 2: Extract key protocol lines directly
    protocol_patterns = [
        r'Seeding density:.*?(?=\n[A-Z]|\n\n|$)',
        r'Split ratio:.*?(?=\n[A-Z]|\n\n|$)',
        r'Media renewal:.*?(?=\n[A-Z]|\n\n|$)',
        r'Temperature:?\s*\d+.*?(?=\n|$)',
        r'Atmosphere:?\s*\d+.*?(?=\n|$)',
        r'ECM:.*?(?=\n[A-Z]|\n\n|$)',
        r'(?:Complete medium|Growth medium).*?(?=Temperature|Atmosphere|\n\n)',
        r'ROCK Inhibitor.*?(?=\n[A-Z]|\n\n|$)',
        r'Organoid (?:Growth Kit|Media Formulation).*?(?=\n[A-Z]|\n\n|$)',
    ]

    for pattern in protocol_patterns:
        matches = re.findall(pattern, full_text, re.DOTALL | re.IGNORECASE)
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 20 and clean_m not in extracted_parts:
                extracted_parts.append(clean_m)

    # Strategy 3: If we found handling section, use that primarily
    if extracted_parts:
        # Combine and deduplicate
        combined = '\n\n'.join(extracted_parts)
    else:
        # Fallback: look for any section with culture keywords
        culture_keywords = ['seeding', 'split ratio', 'medium', 'passage', 'subculture']
        relevant_lines = []
        for line in full_text.split('\n'):
            if any(kw in line.lower() for kw in culture_keywords):
                if len(line) > 20:
                    relevant_lines.append(line)
        combined = '\n'.join(relevant_lines[:50])  # Limit to 50 lines

    return combined


def scrape_atcc_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scrape ATCC product page for media/culture information.

    Returns:
        Tuple of (cleaned_text, error_message)
        If successful: (text, None)
        If failed: (None, error_message)
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if response.status_code == 404:
            return None, "Page not found (404)"

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract relevant sections
        raw_text = extract_handling_info(soup)

        if not raw_text or len(raw_text) < 100:
            # Fallback: get all text from body
            body = soup.find('body')
            if body:
                raw_text = body.get_text(separator='\n', strip=True)

        # Clean for ML
        cleaned_text = clean_text_for_ml(raw_text)

        if len(cleaned_text) < 50:
            return None, "Insufficient content extracted"

        return cleaned_text, None

    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Parsing error: {str(e)}"


# =============================================================================
# MAIN SCRAPER
# =============================================================================

def run_scraper():
    """Main scraper execution."""
    setup_logging()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Create table if needed
    create_media_protocols_table(conn)

    # Get already scraped case_ids (for resume)
    existing_ids = get_existing_case_ids(conn)
    logging.info(f"Already scraped: {len(existing_ids)} case_ids")

    # Get cases to scrape
    all_cases = get_cases_to_scrape(conn)
    logging.info(f"Total cases in database: {len(all_cases)}")

    # Filter out already scraped
    to_scrape = [(cid, pdm) for cid, pdm in all_cases if cid not in existing_ids]
    logging.info(f"Cases to scrape: {len(to_scrape)}")

    if not to_scrape:
        logging.info("Nothing to scrape. All cases already processed.")
        conn.close()
        return

    # Track statistics
    success_count = 0
    error_count = 0
    errors = {}

    # Scrape each case
    logging.info("=" * 60)
    logging.info("Starting scrape...")
    logging.info("=" * 60)

    for idx, (case_id, pdm_code) in enumerate(to_scrape):
        url = construct_atcc_url(pdm_code)

        logging.info(f"[{idx + 1}/{len(to_scrape)}] {case_id} -> {url}")

        # Scrape the page
        text, error = scrape_atcc_page(url)

        if text:
            # Save to database
            save_media_protocol(conn, case_id, text, url)
            success_count += 1
            logging.info(f"  SUCCESS: {len(text)} chars extracted")
        else:
            # Log error but continue
            error_count += 1
            errors[case_id] = error
            # Save with empty text to mark as attempted
            save_media_protocol(conn, case_id, "", url)
            logging.warning(f"  FAILED: {error}")

        # Polite delay between requests
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)

    conn.close()

    # Final summary
    logging.info("=" * 60)
    logging.info("SCRAPE COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {error_count}")

    if errors:
        logging.info("Failed cases:")
        for case_id, err in list(errors.items())[:10]:  # Show first 10
            logging.info(f"  {case_id}: {err}")
        if len(errors) > 10:
            logging.info(f"  ... and {len(errors) - 10} more")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_scraper()
