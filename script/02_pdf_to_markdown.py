#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: PDF-to-Markdown Conversion (Multi-process)
===================================================
Converts downloaded PDFs and supplementary materials into unified Markdown files
for downstream LLM-based extraction.

Supports: PDF, DOCX, CSV, TSV, XLSX, TXT

Usage:
    python 02_pdf_to_markdown.py --input <pdf_folder> --output <md_folder> [--workers N]

Example:
    python 02_pdf_to_markdown.py \
        --input ../data_demo/1_download_pmc \
        --output ../data_demo/2_convert_md_pmc \
        --workers 8
"""

import os
import argparse
import logging
from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from pymupdf4llm import to_markdown

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_md_from_pdf(pdf_path: str) -> Optional[str]:
    """Convert PDF to Markdown using PyMuPDF."""
    try:
        return to_markdown(
            pdf_path,
            pages=None,
            write_images=False,
            embed_images=False,
            dpi=150,
            image_format="png",
            table_strategy="lines_strict",
            show_progress=False,
        )
    except Exception as e:
        logger.error(f"PDF→MD failed: {pdf_path} | {e}")
        return None


def read_file_with_fallback(filepath: str, sep: str = None) -> pd.DataFrame:
    """Read CSV/TSV with multiple encoding fallbacks."""
    encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, sep=sep or ",", encoding=enc)
        except Exception:
            continue
    raise ValueError(f"All encoding attempts failed: {filepath}")


def convert_supplementary_file(filepath: str) -> Optional[str]:
    """Convert supplementary file to Markdown. Supports CSV/TSV/XLSX/TXT/DOCX."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = read_file_with_fallback(filepath, sep=sep)
            return df.to_markdown(index=False)
        elif ext == ".xlsx":
            return pd.read_excel(filepath).to_markdown(index=False)
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f"```\n{f.read()}\n```"
        elif ext == ".docx":
            doc = Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.warning(f"Failed to convert {filepath}: {e}")
    return None


def process_single_paper(args: tuple) -> str:
    """Process one PMC folder → single merged Markdown file."""
    pmc_folder, output_dir = args
    pmc_id = os.path.basename(pmc_folder)
    md_output_path = os.path.join(output_dir, f"{pmc_id}.md")
    os.makedirs(output_dir, exist_ok=True)

    combined_md = f"# {pmc_id}\n\n"

    # Main fulltext
    fulltext_pdf = os.path.join(pmc_folder, "fulltext.pdf")
    if os.path.isfile(fulltext_pdf):
        md = extract_md_from_pdf(fulltext_pdf)
        if md:
            combined_md += f"## Fulltext\n\n{md}\n\n"

    # Supplementary files
    supp_dir = os.path.join(pmc_folder, "supplementaryfiles")
    if os.path.isdir(supp_dir):
        for fname in sorted(os.listdir(supp_dir)):
            file_path = os.path.join(supp_dir, fname)
            if fname.lower().endswith(".pdf"):
                md = extract_md_from_pdf(file_path)
                if md:
                    combined_md += f"## Supplementary: {fname}\n\n{md}\n\n"
            elif fname.lower().endswith((".csv", ".tsv", ".xlsx", ".txt", ".docx")):
                md = convert_supplementary_file(file_path)
                if md:
                    combined_md += f"## Supplementary: {fname}\n\n{md}\n\n"

    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(combined_md)

    return pmc_id


def main():
    parser = argparse.ArgumentParser(description="Convert PMC PDFs to Markdown")
    parser.add_argument("--input", required=True, help="Root directory containing PMC folders")
    parser.add_argument("--output", required=True, help="Output directory for .md files")
    parser.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1),
                        help="Number of parallel workers (default: CPU_COUNT - 1)")
    args = parser.parse_args()

    # Get already-processed papers
    done_papers = set()
    if os.path.exists(args.output):
        done_papers = {f[:-3] for f in os.listdir(args.output) if f.endswith(".md")}
    logger.info(f"Already processed: {len(done_papers)} papers")

    # Get papers to process
    pmc_folders = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if f.startswith("PMC") and f not in done_papers and os.path.isdir(os.path.join(args.input, f))
    ]
    logger.info(f"Papers to process: {len(pmc_folders)}")

    if not pmc_folders:
        logger.info("Nothing to process. Done.")
        return

    # Multi-process conversion
    task_args = [(folder, args.output) for folder in pmc_folders]
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_paper, task_args),
            total=len(task_args),
            desc="Converting PDFs",
        ))

    logger.info(f"Conversion complete. Processed {len(results)} papers.")


if __name__ == "__main__":
    main()
