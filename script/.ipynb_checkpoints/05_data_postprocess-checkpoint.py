#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Data Post-processing & Standardization
================================================
Performs comprehensive cleaning and normalization of the merged dataset:
- Gene name normalization
- Mutation format standardization (3-letter → 1-letter amino acid codes)
- Antibiotic name standardization (with ChEMBL lookup)
- Organism/strain name cleaning
- Unit normalization (→ μg/mL or μM)
- Value formatting and validation

Usage:
    python 05_data_postprocess.py \
        --input ../data_demo/4_integrated/LLM_resistance_merged.csv \
        --output ../data_demo/5_standardized/antibiotic_resistance_database.csv

Note: This script uses the ChEMBL public API (no key required) for 
      antibiotic SMILES lookup. Internet connection is needed.
"""

import os
import re
import argparse
import logging

import pandas as pd

# Import utility functions
from utils.data_postprocess import (
    clean_dataframe,
    normalize_gene_names,
    standardize_mutations,
    standardize_antibiotic_names,
    clean_organism_names,
    normalize_units,
    format_values,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Post-process and standardize extracted data")
    parser.add_argument("--input", required=True, help="Input CSV (merged extraction results)")
    parser.add_argument("--output", required=True, help="Output CSV (standardized database)")
    parser.add_argument("--skip-chembl", action="store_true",
                        help="Skip ChEMBL API lookup (offline mode)")
    args = parser.parse_args()

    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Input records: {len(df)}")

    # Step-by-step processing
    logger.info("Step 1/6: Basic cleaning...")
    df = clean_dataframe(df)

    logger.info("Step 2/6: Gene name normalization...")
    df = normalize_gene_names(df)

    logger.info("Step 3/6: Mutation standardization...")
    df = standardize_mutations(df)

    logger.info("Step 4/6: Antibiotic name standardization...")
    df = standardize_antibiotic_names(df, use_chembl=not args.skip_chembl)

    logger.info("Step 5/6: Organism name cleaning...")
    df = clean_organism_names(df)

    logger.info("Step 6/6: Unit & value normalization...")
    df = normalize_units(df)
    df = format_values(df)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Output records: {len(df)}")
    logger.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
