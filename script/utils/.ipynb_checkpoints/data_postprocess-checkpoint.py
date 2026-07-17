#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data post-processing utility functions for antibiotic resistance data normalization.
"""

import re
import pandas as pd
import numpy as np

# Three-letter to one-letter amino acid mapping
AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic DataFrame cleaning: strip whitespace, remove empty rows."""
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "None": "", "N/A": "", "-": ""})

    # Remove rows where all key fields are empty
    key_cols = ["Organism", "Genes", "Antibiotic"]
    existing_keys = [c for c in key_cols if c in df.columns]
    if existing_keys:
        mask = df[existing_keys].replace("", np.nan).notna().any(axis=1)
        df = df[mask].reset_index(drop=True)

    return df


def normalize_gene_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize gene names: lowercase, remove extra whitespace."""
    if "Genes" not in df.columns:
        return df

    def _normalize(name):
        if not name or name == "":
            return ""
        # Remove extra spaces
        name = re.sub(r"\s+", " ", name).strip()
        # Common corrections
        name = name.replace("'", "'")
        return name

    df["Genes"] = df["Genes"].apply(_normalize)
    return df


def standardize_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 3-letter amino acid codes to 1-letter in mutations."""
    if "Mutation" not in df.columns:
        return df

    def _convert(mutation):
        if not mutation or mutation == "":
            return ""
        # Pattern: Ala123Val → A123V
        pattern = r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"
        def replacer(m):
            aa1 = AA_3TO1.get(m.group(1), m.group(1))
            pos = m.group(2)
            aa2 = AA_3TO1.get(m.group(3), m.group(3))
            return f"{aa1}{pos}{aa2}"
        return re.sub(pattern, replacer, mutation)

    df["Mutation"] = df["Mutation"].apply(_convert)
    return df


def standardize_antibiotic_names(df: pd.DataFrame, use_chembl: bool = True) -> pd.DataFrame:
    """Standardize antibiotic names and optionally lookup SMILES via ChEMBL."""
    if "Antibiotic" not in df.columns:
        return df

    def _clean_name(name):
        if not name or name == "":
            return ""
        # Title case, strip
        name = name.strip().lower()
        # Common fixes
        name = name.replace("amoxicillin/clavulanate", "amoxicillin-clavulanate")
        return name

    df["Antibiotic"] = df["Antibiotic"].apply(_clean_name)

    # Optional: ChEMBL SMILES lookup
    if use_chembl and "SMILES" in df.columns:
        try:
            from chembl_webresource_client.new_client import new_client
            molecule = new_client.molecule

            # Get unique antibiotics missing SMILES
            missing_smiles = df[df["SMILES"] == ""]["Antibiotic"].unique()
            smiles_map = {}

            for ab in missing_smiles:
                if not ab:
                    continue
                try:
                    results = molecule.filter(pref_name__iexact=ab)
                    if results:
                        smiles_map[ab] = results[0].get("molecule_structures", {}).get("canonical_smiles", "")
                except Exception:
                    continue

            if smiles_map:
                df.loc[df["SMILES"] == "", "SMILES"] = df.loc[df["SMILES"] == "", "Antibiotic"].map(smiles_map)

        except ImportError:
            pass  # chembl_webresource_client not installed

    return df


def clean_organism_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize organism names."""
    if "Organism" not in df.columns:
        return df

    def _clean(name):
        if not name or name == "":
            return ""
        name = re.sub(r"\s+", " ", name).strip()
        # Remove strain info if it's in a separate column
        # Keep genus + species (italicized format)
        return name

    df["Organism"] = df["Organism"].apply(_clean)
    return df


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize measurement units to standard forms."""
    if "standard_units" not in df.columns:
        return df

    unit_map = {
        "ug/ml": "μg/mL", "ug/mL": "μg/mL", "µg/ml": "μg/mL",
        "mg/l": "μg/mL", "mg/L": "μg/mL",
        "um": "μM", "uM": "μM", "µM": "μM",
        "nm": "nM", "nM": "nM",
        "mm": "mM", "mM": "mM",
    }

    df["standard_units"] = df["standard_units"].map(
        lambda x: unit_map.get(x, x) if x else ""
    )
    return df


def format_values(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric values in standard_value column."""
    if "standard_value" not in df.columns:
        return df

    def _format(val):
        if not val or val == "":
            return ""
        try:
            num = float(val)
            if num == int(num) and abs(num) < 1e6:
                return str(int(num))
            return f"{num:.4g}"
        except (ValueError, TypeError):
            return str(val)

    df["standard_value"] = df["standard_value"].apply(_format)
    return df
