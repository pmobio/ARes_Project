#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Data Integration
=========================
Parses LLM-extracted Markdown tables, validates format, filters invalid rows,
and merges all results into a unified CSV dataset.

In addition to the LLM-extracted fields, this step also attaches an
authoritative provenance identifier for every record:
  - PMC_ID: derived deterministically from the response filename
            (response_PMC6051542.csv -> PMC6051542), NOT from the LLM.
  - Pubmed: the LLM often fails to find/guess the PubMed ID from the article
            body (returns "-"). If Step 1 (01_paper_retrieval.py) already
            downloaded eupmc_result.json for this paper, its "pmid" field is
            the ground truth and is used to fill/override the Pubmed column.

Usage:
    python 04_data_integration.py \
        --input ../data_demo/3_response_pmc \
        --output ../data_demo/4_integrated/LLM_resistance_merged.csv \
        --download-dir ../data_demo/1_download_pmc

Input:  response_*.csv files (Markdown table format from LLM output)
Output: Merged CSV with validated records (includes PMC_ID / Pubmed columns)
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Expected columns from LLM extraction
EXPECTED_COLUMNS = [
    "Source", "Pubmed", "Organism", "Strain", "Genes", "Mutation",
    "Antibiotic", "Class", "Mechanism", "standard_relation",
    "standard_type", "standard_value", "standard_units",
    "Susceptibility", "SMILES", "Notes"
]

# Boilerplate patterns indicating LLM template responses (no actual data)
BOILERPLATE_PATTERNS = [
    r"no.*relevant.*data",
    r"no.*antibiotic.*resistance",
    r"does not contain",
    r"not.*applicable",
    r"the article does not",
    r"no.*information.*extracted",
]


def parse_markdown_table(content: str) -> Optional[pd.DataFrame]:
    """Parse a Markdown table from LLM response into a DataFrame."""
    lines = content.strip().split("\n")
    table_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip separator lines (e.g., |---|---|)
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        # Keep lines that look like table rows
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_lines.append(cells)

    if len(table_lines) < 2:
        return None

    # First row as header
    header = table_lines[0]
    data = table_lines[1:]

    # Filter rows with wrong column count
    n_cols = len(header)
    valid_data = [row for row in data if len(row) == n_cols]

    if not valid_data:
        return None

    df = pd.DataFrame(valid_data, columns=header)
    return df


def is_boilerplate(content: str) -> bool:
    """Check if the response is a boilerplate/no-data template."""
    lower = content.lower()[:500]
    return any(re.search(p, lower) for p in BOILERPLATE_PATTERNS)


def extract_pmc_id_from_filename(filepath: str) -> Optional[str]:
    """Recover the authoritative PMC ID from a `response_<PMCID>.csv` filename.

    This is deterministic (set by Step 3 based on the input .md filename) and
    should always be trusted over anything the LLM may have written into the
    'Source'/'Pubmed' columns of the extracted table.
    """
    stem = Path(filepath).stem  # e.g. "response_PMC6051542"
    m = re.match(r"^response_(PMC\d+)$", stem)
    return m.group(1) if m else None


def load_pmid_lookup(download_dir: Optional[str]) -> dict:
    """Build a {PMC_ID: PMID} lookup from Step 1's eupmc_result.json files.

    Step 1 (01_paper_retrieval.py) saves the ground-truth Europe PMC metadata
    for every downloaded paper at `<download_dir>/<PMC_ID>/eupmc_result.json`,
    which contains a reliable "pmid" field. This is far more trustworthy than
    asking the LLM to spot a PubMed ID inside the article body (it usually
    isn't printed there at all, hence the LLM outputs "-").
    """
    lookup = {}
    if not download_dir or not os.path.isdir(download_dir):
        return lookup

    for entry in sorted(os.listdir(download_dir)):
        if not entry.startswith("PMC"):
            continue
        json_path = os.path.join(download_dir, entry, "eupmc_result.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  Failed to parse {json_path}: {e}")
            continue

        pmid = meta.get("pmid")
        if pmid:
            lookup[entry] = str(pmid).strip()

    logger.info(f"Loaded {len(lookup)} PMC_ID -> PubMed ID mappings from {download_dir}")
    return lookup


def process_response_file(filepath: str, pmid_lookup: dict) -> Optional[pd.DataFrame]:
    """Process a single response file and return parsed DataFrame."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return None

    if is_boilerplate(content):
        logger.debug(f"  Boilerplate skipped: {filepath}")
        return None

    df = parse_markdown_table(content)
    if df is not None and len(df) > 0:
        # Add source file info
        df["_source_file"] = os.path.basename(filepath)

        # Attach the authoritative PMC ID (from filename, not from the LLM)
        pmc_id = extract_pmc_id_from_filename(filepath)
        df["PMC_ID"] = pmc_id or ""

        # Fill/override Pubmed with the ground-truth PMID from Step 1's
        # metadata, when available. Only fall back to whatever the LLM
        # produced if no authoritative PMID could be found.
        if pmc_id and pmc_id in pmid_lookup and "Pubmed" in df.columns:
            df["Pubmed"] = pmid_lookup[pmc_id]

        return df

    return None


def main():
    parser = argparse.ArgumentParser(description="Integrate LLM extraction results")
    parser.add_argument("--input", required=True, help="Directory containing response_*.csv files")
    parser.add_argument("--output", required=True, help="Output path for merged CSV")
    parser.add_argument(
        "--download-dir", default=None,
        help="Step 1 download directory (e.g. ../data_demo/1_download_pmc) containing "
             "<PMC_ID>/eupmc_result.json, used to look up ground-truth PubMed IDs. "
             "If omitted, inferred as '<input>/../1_download_pmc' when it exists.",
    )
    args = parser.parse_args()

    # Resolve the Step-1 download dir used for authoritative PMID lookup.
    download_dir = args.download_dir
    if not download_dir:
        guess = os.path.join(os.path.dirname(os.path.normpath(args.input)), "1_download_pmc")
        if os.path.isdir(guess):
            download_dir = guess
    pmid_lookup = load_pmid_lookup(download_dir)

    # Find all response files (may be in subdirectories).
    # Skip hidden directories (e.g. Jupyter's .ipynb_checkpoints/), which
    # otherwise silently duplicate every record in the merged output.
    response_files = []
    for root, dirs, files in os.walk(args.input):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("response_") and f.endswith(".csv"):
                response_files.append(os.path.join(root, f))

    logger.info(f"Found {len(response_files)} response files")

    # Process all files
    all_dfs = []
    success, skipped = 0, 0

    for filepath in sorted(response_files):
        df = process_response_file(filepath, pmid_lookup)
        if df is not None:
            all_dfs.append(df)
            success += 1
        else:
            skipped += 1

    logger.info(f"Parsed: {success} | Skipped: {skipped}")

    if not all_dfs:
        logger.warning("No valid data extracted. Check input files.")
        return

    # Merge
    merged = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total records: {len(merged)}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)
    logger.info(f"Saved merged dataset: {args.output}")


if __name__ == "__main__":
    main()
