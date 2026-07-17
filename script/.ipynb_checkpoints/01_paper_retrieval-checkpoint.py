#!/usr/bin/env python3
"""
Step 1: Paper Retrieval from Europe PMC & arXiv
================================================

This script automates the full literature retrieval pipeline:
  1. Keyword-based search via pygetpapers (Europe PMC / arXiv)
  2. PMC ID extraction from search results
  3. Bulk PDF + supplementary download by PMC ID list

It also supports importing PMC IDs from external databases (ChEMBL, BV-BRC).

Usage:
------
  # Mode 1: Keyword search → extract IDs → download (full pipeline)
  python 01_paper_retrieval.py --mode search \
      --query "(METHODS:'MIC') AND (TITLE:'antibiotic resistance')" \
      --max-results 100 --output ./data/1_download_pmc

  # Mode 2: Download from an existing PMC ID list
  python 01_paper_retrieval.py --mode download \
      --id-file pmc_ids.txt --output ./data/1_download_pmc

  # Mode 3: Import PMC IDs from ChEMBL/BV-BRC CSV and download
  python 01_paper_retrieval.py --mode import \
      --import-file chembl_pub_info.csv --id-column pmc_id \
      --output ./data/1_download_pmc

Author: Ziyi Yang (chriszyyang@tencent.com)
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import requests
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_QUERY = (
    "(METHODS:'MIC' OR METHODS:'minimum inhibitory concentration' "
    "OR METHODS:'EC50' OR METHODS:'IC50' OR METHODS:'Ki' OR METHODS:'Kd') "
    "AND (TITLE: resistance OR TITLE: antibiotic OR 'multidrug resistance' "
    "OR 'antibiotic resistance' OR 'antimicrobial resistance') "
    "AND (LICENSE:'cc by' OR LICENSE:'cc-by')"
)

DEFAULT_OUTPUT = PROJECT_DIR / "data_demo" / "1_download_pmc"
DEFAULT_RAW_DIR = PROJECT_DIR / "data_demo" / "0_raw_data"
DEFAULT_LOG_DIR = PROJECT_DIR / "logs"

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Core Functions
# ==============================================================================


def check_pygetpapers():
    """Check if pygetpapers is installed."""
    if shutil.which("pygetpapers") is None:
        logger.error("pygetpapers not found. Install with: pip install pygetpapers")
        sys.exit(1)


def search_europepmc(query: str, max_results: int, output_dir: Path) -> List[str]:
    """
    Step 1: Run keyword search on Europe PMC and return list of PMC IDs.

    Args:
        query: Europe PMC query string
        max_results: Maximum number of results to retrieve
        output_dir: Directory to store raw metadata (eupmc_results.json)

    Returns:
        List of PMC IDs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching Europe PMC with query (max {max_results} results)...")
    logger.info(f"Query: {query[:100]}...")

    # Use pygetpapers to search (metadata only, no PDF download)
    cmd = [
        "pygetpapers",
        "--query", query,
        "--output", str(output_dir),
        "--onlyquery",
        "-k", str(max_results),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"pygetpapers search failed: {result.stderr}")
        sys.exit(1)

    # Parse the results JSON to extract PMC IDs
    json_file = output_dir / "eupmc_results.json"
    if not json_file.exists():
        logger.error(f"Search results not found: {json_file}")
        sys.exit(1)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", {})
    pmc_ids = [pmc_id for pmc_id in papers.keys() if pmc_id.startswith("PMC")]

    logger.info(f"Found {len(pmc_ids)} papers from Europe PMC keyword search.")
    return pmc_ids


def search_arxiv(query: str, max_results: int, output_dir: Path) -> List[str]:
    """
    Search arXiv via pygetpapers and download PDFs directly.

    Args:
        query: arXiv query string
        max_results: Maximum number of results
        output_dir: Download directory

    Returns:
        List of downloaded paper folder names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching arXiv with query (max {max_results} results)...")

    cmd = [
        "pygetpapers",
        "--query", query,
        "--output", str(output_dir),
        "-k", str(max_results),
        "--pdf",
        "--api", "arxiv",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"arXiv search failed: {result.stderr}")
        return []

    # List downloaded folders
    downloaded = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    logger.info(f"Downloaded {len(downloaded)} papers from arXiv.")
    return downloaded


def download_by_pmc_ids(
    pmc_ids: List[str],
    output_dir: Path,
    log_dir: Path,
    sleep_interval: float = 1.0,
) -> dict:
    """
    Step 3: Bulk download PDFs and supplementary files by PMC ID list.

    Args:
        pmc_ids: List of PMC IDs (e.g., ['PMC12189289', ...])
        output_dir: Directory to save downloaded papers
        log_dir: Directory for log files
        sleep_interval: Seconds to wait between downloads (avoid rate limiting)

    Returns:
        Dict with counts: {'total', 'success', 'skipped', 'failed'}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    failed_log = log_dir / "failed_downloads.txt"

    stats = {"total": len(pmc_ids), "success": 0, "skipped": 0, "failed": 0}

    logger.info(f"Starting bulk download of {len(pmc_ids)} papers...")

    for pmc_id in tqdm(pmc_ids, desc="Downloading"):
        pmc_id = str(pmc_id).strip()
        if not pmc_id:
            continue

        paper_dir = output_dir / pmc_id

        # Skip if already downloaded
        if paper_dir.exists():
            stats["skipped"] += 1
            continue

        # Download PDF and supplementary files
        cmd = [
            "pygetpapers",
            "-q", pmc_id,
            "--output", str(output_dir),
            "--pdf", "--supp",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            with open(failed_log, "a") as f:
                f.write(f"{pmc_id}\n")

        time.sleep(sleep_interval)

    logger.info(
        f"Download complete: {stats['success']} success, "
        f"{stats['skipped']} skipped, {stats['failed']} failed "
        f"(out of {stats['total']} total)"
    )
    return stats


def cleanup_empty_folders(download_dir: Path) -> int:
    """
    Post-processing: remove folders that only contain eupmc_result.json
    (i.e., no actual PDF/XML content was downloaded).

    Returns:
        Number of removed folders
    """
    removed = 0
    for folder in os.listdir(download_dir):
        folder_path = download_dir / folder
        if folder_path.is_dir():
            files = os.listdir(folder_path)
            if files == ["eupmc_result.json"]:
                shutil.rmtree(folder_path)
                removed += 1
                logger.debug(f"Removed empty folder: {folder}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} empty folders (no PDF content).")
    return removed


def load_pmc_ids_from_file(filepath: Path) -> List[str]:
    """Load PMC IDs from a text file (one ID per line)."""
    if not filepath.exists():
        logger.error(f"ID file not found: {filepath}")
        sys.exit(1)

    pmc_ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pmc_ids.append(line)

    logger.info(f"Loaded {len(pmc_ids)} PMC IDs from {filepath}")
    return pmc_ids


def import_pmc_ids_from_csv(
    filepath: Path,
    id_column: str = "pmc_id",
    pmid_column: Optional[str] = None,
    separator: str = ",",
) -> List[str]:
    """
    Import PMC IDs from a CSV/TSV file (e.g., ChEMBL or BV-BRC exports).

    If the file has a PMCID column, use it directly.
    If only PMIDs are available, convert them to PMCIDs via Europe PMC API.

    Args:
        filepath: Path to CSV/TSV file
        id_column: Column name containing PMC IDs
        pmid_column: Column name containing PMIDs (for conversion)
        separator: File separator (',' for CSV, '\\t' for TSV)

    Returns:
        List of PMC IDs
    """
    if not filepath.exists():
        logger.error(f"Import file not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath, sep=separator, low_memory=False)

    # If PMC ID column exists, use directly
    if id_column in df.columns:
        pmc_ids = df[id_column].dropna().unique().tolist()
        logger.info(f"Imported {len(pmc_ids)} PMC IDs from column '{id_column}'")
        return [str(x).strip() for x in pmc_ids]

    # If only PMID column is available, convert to PMCID
    if pmid_column and pmid_column in df.columns:
        logger.info(f"Converting PMIDs to PMCIDs via Europe PMC API...")
        pmid_list = df[pmid_column].astype(str).dropna().unique().tolist()
        pmc_ids = _convert_pmids_to_pmcids(pmid_list)
        logger.info(f"Converted {len(pmc_ids)} PMIDs → PMCIDs")
        return pmc_ids

    logger.error(
        f"Column '{id_column}' not found in {filepath}. "
        f"Available columns: {list(df.columns)}"
    )
    sys.exit(1)


def _convert_pmids_to_pmcids(
    pmid_list: List[str], batch_size: int = 50, delay: float = 1.0
) -> List[str]:
    """
    Convert PMIDs to PMCIDs using Europe PMC REST API.

    Args:
        pmid_list: List of PubMed IDs
        batch_size: Number of PMIDs per API call
        delay: Sleep time between requests

    Returns:
        List of PMC IDs (only those with available PMCID)
    """
    pmc_ids = []

    for i in tqdm(range(0, len(pmid_list), batch_size), desc="Converting PMIDs"):
        batch = pmid_list[i : i + batch_size]
        query = " OR ".join([f"EXT_ID:{pmid}" for pmid in batch])
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": batch_size,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                results = response.json().get("resultList", {}).get("result", [])
                for entry in results:
                    pmcid = entry.get("pmcid")
                    if pmcid:
                        pmc_ids.append(pmcid)
        except Exception as e:
            logger.warning(f"Error fetching batch {i // batch_size + 1}: {e}")

        time.sleep(delay)

    return list(set(pmc_ids))


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper Retrieval Pipeline for AResKG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: keyword search → download
  python 01_paper_retrieval.py --mode search \\
      --query "(METHODS:'MIC') AND (TITLE:'antibiotic resistance')" \\
      --max-results 500 --output ./data/1_download_pmc

  # Download from PMC ID list
  python 01_paper_retrieval.py --mode download \\
      --id-file pmc_ids.txt --output ./data/1_download_pmc

  # Import from ChEMBL CSV
  python 01_paper_retrieval.py --mode import \\
      --import-file chembl_pub_info.csv --id-column pmc_id

  # Import from BV-BRC TSV (PMID → PMCID conversion)
  python 01_paper_retrieval.py --mode import \\
      --import-file bvbrc_amr.tsv --separator '\\t' \\
      --pmid-column pmid
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["search", "download", "import"],
        required=True,
        help="Pipeline mode: 'search' (keyword→download), 'download' (from ID list), 'import' (from CSV/TSV)",
    )

    # Search mode options
    parser.add_argument(
        "--query", type=str, default=DEFAULT_QUERY,
        help="Europe PMC query string (for search mode)",
    )
    parser.add_argument(
        "--max-results", type=int, default=100,
        help="Maximum number of search results (default: 100)",
    )
    parser.add_argument(
        "--api", choices=["europepmc", "arxiv"], default="europepmc",
        help="Search API to use (default: europepmc)",
    )

    # Download mode options
    parser.add_argument(
        "--id-file", type=str,
        help="Text file with PMC IDs (one per line, for download mode)",
    )

    # Import mode options
    parser.add_argument(
        "--import-file", type=str,
        help="CSV/TSV file to import PMC IDs from (for import mode)",
    )
    parser.add_argument(
        "--id-column", type=str, default="pmc_id",
        help="Column name containing PMC IDs (default: pmc_id)",
    )
    parser.add_argument(
        "--pmid-column", type=str, default=None,
        help="Column name containing PMIDs (triggers PMID→PMCID conversion)",
    )
    parser.add_argument(
        "--separator", type=str, default=",",
        help="File separator for CSV/TSV (default: ',')",
    )

    # Common options
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output directory for downloads (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=str(DEFAULT_RAW_DIR),
        help=f"Directory for raw search metadata (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(DEFAULT_LOG_DIR),
        help=f"Log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Sleep interval between downloads in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Skip post-download cleanup of empty folders",
    )
    parser.add_argument(
        "--save-ids", type=str, default=None,
        help="Save extracted PMC IDs to a text file (optional)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    raw_dir = Path(args.raw_dir)
    log_dir = Path(args.log_dir)

    # Check dependencies
    check_pygetpapers()

    pmc_ids = []

    # === Mode: Keyword Search ===
    if args.mode == "search":
        logger.info("=" * 60)
        logger.info(" Mode: Keyword Search → Extract IDs → Download")
        logger.info("=" * 60)

        if args.api == "europepmc":
            # Step 1: Search Europe PMC
            pmc_ids = search_europepmc(args.query, args.max_results, raw_dir)
        elif args.api == "arxiv":
            # arXiv mode: search and download directly
            arxiv_output = output_dir.parent / "1_download_arxiv"
            search_arxiv(args.query, args.max_results, arxiv_output)
            logger.info("arXiv papers downloaded directly (no PMC ID extraction needed).")
            return

    # === Mode: Download from ID list ===
    elif args.mode == "download":
        logger.info("=" * 60)
        logger.info(" Mode: Download from PMC ID List")
        logger.info("=" * 60)

        if not args.id_file:
            logger.error("--id-file is required for download mode")
            sys.exit(1)
        pmc_ids = load_pmc_ids_from_file(Path(args.id_file))

    # === Mode: Import from CSV/TSV ===
    elif args.mode == "import":
        logger.info("=" * 60)
        logger.info(" Mode: Import PMC IDs from External Database")
        logger.info("=" * 60)

        if not args.import_file:
            logger.error("--import-file is required for import mode")
            sys.exit(1)

        pmc_ids = import_pmc_ids_from_csv(
            Path(args.import_file),
            id_column=args.id_column,
            pmid_column=args.pmid_column,
            separator=args.separator,
        )

    # === Save PMC IDs if requested ===
    if args.save_ids and pmc_ids:
        save_path = Path(args.save_ids)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for pmc_id in pmc_ids:
                f.write(f"{pmc_id}\n")
        logger.info(f"Saved {len(pmc_ids)} PMC IDs to {save_path}")

    # === Download ===
    if pmc_ids:
        stats = download_by_pmc_ids(pmc_ids, output_dir, log_dir, args.sleep)

        # Post-processing: cleanup empty folders
        if not args.no_cleanup:
            cleanup_empty_folders(output_dir)

        # Final summary
        remaining = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("PMC")
        ]
        logger.info(f"Final count: {len(remaining)} valid PMC paper folders.")
    else:
        logger.warning("No PMC IDs to download.")


if __name__ == "__main__":
    main()
