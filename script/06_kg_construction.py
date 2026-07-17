#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: Knowledge Graph Construction
======================================
Converts the standardized antibiotic resistance database into a knowledge graph
represented as subject-predicate-object triples (TSV format).

Entity Types (14):
    Organism, Strain, Gene, Mutation, Antibiotic, Antibiotic_Class,
    Resistance_Mechanism, GO_MF, GO_BP, GO_CC, Measurement_ID,
    Standard_Type, Standard_Value, Standard_Unit

Relation Types (27):
    has_gene, has_mutation, resistant_to, sensitive_to, confers_resistance_to,
    targeted_by, upregulated_by_antibiotic, activates, represses, is_part_of,
    has, is_involved_in, has_mechanism, has_structure, belongs_to_class, ...

Usage:
    python 06_kg_construction.py \
        --input ../data_demo/5_standardized/antibiotic_resistance_database.csv \
        --output ../data_demo/6_knowledge_graph/antibiotic_resistance_kg.tsv \
        [--gene-interaction ../data_demo/0_raw_data/gene_gene_interaction.csv] \
        [--external-kg path/to/external_kg1.tsv path/to/external_kg2.csv ...]

Extension point:
    --external-kg accepts one or more third-party KG triple files (e.g.
    DrugBank/CARD exports, or a KG produced by another pipeline) and merges
    them into the output after deduplication. Column names are auto-detected
    via common aliases (Subject/Predicate/Object, head/relation/tail,
    source/relationship/target); header-less 3-column files are also
    accepted positionally. See `load_external_triples()`.

Output: TSV file with columns: Subject, Predicate, Object
"""

import os
import re
import hashlib
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _split_multi_values(raw: str) -> list:
    """Split a possibly multi-valued cell (e.g. "gyrA, gyrB (dual plasmids)")
    into individual entity names. Trailing parenthetical annotations such as
    "(dual plasmids)" are stripped since they describe the relation, not the
    entity identity."""
    if not raw or raw == "nan":
        return []
    # Drop trailing parenthetical notes, then split on comma/semicolon.
    cleaned = re.sub(r"\([^)]*\)", "", raw)
    parts = [p.strip() for p in re.split(r"[,;]", cleaned)]
    return [p for p in parts if p and p != "nan"]


def _stable_measurement_id(*parts: str) -> str:
    """Deterministic measurement ID, independent of PYTHONHASHSEED /
    process, so re-running the pipeline on unchanged data yields identical
    node IDs (needed for diffing / incremental updates)."""
    key = "|".join(parts).encode("utf-8")
    return f"Measurement_{hashlib.md5(key).hexdigest()[:12]}"


# --- External KG merge interface -------------------------------------------
# Column-name aliases accepted when auto-detecting subject/predicate/object
# columns in an externally supplied triple file (case-insensitive).
_SUBJ_ALIASES = {"subject", "head", "source", "subj", "s"}
_PRED_ALIASES = {"predicate", "relation", "relationship", "rel", "p"}
_OBJ_ALIASES = {"object", "tail", "target", "obj", "o"}


def load_external_triples(filepath: str) -> list:
    """Load a third-party / external knowledge graph as (subject, predicate,
    object) triples so it can be merged into this pipeline's output.

    This is the extension point for plugging in outside KGs (e.g. DrugBank,
    CARD, KEGG exports, or a KG produced by a different pipeline) without
    having to touch ``build_triples_from_row``.

    Supported input:
        - TSV or CSV file (separator inferred from extension; falls back to
          auto-sniffing for anything else).
        - Header row using any common alias for subject/predicate/object,
          e.g. ``Subject/Predicate/Object``, ``head/relation/tail``,
          ``source/relationship/target`` (case-insensitive).
        - Header-less files with exactly 3 columns are also accepted; the
          columns are then used positionally as (subject, predicate, object).

    Returns:
        A list of ``(subject, predicate, object)`` string tuples. Rows with
        a missing subject/object are dropped. Returns an empty list (with a
        logged warning) if the file cannot be parsed into a triple schema.
    """
    ext = os.path.splitext(filepath)[1].lower()
    sep = "\t" if ext == ".tsv" else ("," if ext == ".csv" else None)

    try:
        df = pd.read_csv(filepath, sep=sep, engine="python" if sep is None else "c")
    except Exception as exc:
        logger.warning(f"Failed to read external KG file '{filepath}': {exc}")
        return []

    # Try to resolve subject/predicate/object columns from the header via
    # known aliases.
    col_lookup = {str(c).strip().lower(): c for c in df.columns}
    subj_col = next((col_lookup[a] for a in _SUBJ_ALIASES if a in col_lookup), None)
    pred_col = next((col_lookup[a] for a in _PRED_ALIASES if a in col_lookup), None)
    obj_col = next((col_lookup[a] for a in _OBJ_ALIASES if a in col_lookup), None)

    if subj_col is None or pred_col is None or obj_col is None:
        # No recognizable header. If the file has exactly 3 columns, assume
        # it is a header-less triple file and re-read positionally.
        if df.shape[1] == 3:
            df = pd.read_csv(filepath, sep=sep, engine="python" if sep is None else "c", header=None)
            subj_col, pred_col, obj_col = df.columns[:3]
        else:
            logger.warning(
                f"Could not identify subject/predicate/object columns in external "
                f"KG file '{filepath}' (columns found: {list(df.columns)}); skipping."
            )
            return []

    triples = []
    for _, row in df.iterrows():
        s = str(row.get(subj_col, "")).strip()
        p = str(row.get(pred_col, "")).strip() or "related_to"
        o = str(row.get(obj_col, "")).strip()
        if s and o and s != "nan" and o != "nan":
            triples.append((s, p, o))
    return triples


def build_triples_from_row(row: pd.Series) -> list:
    """Generate knowledge graph triples from a single database row."""
    triples = []

    organism = str(row.get("Organism", "")).strip()
    strain = str(row.get("Strain", "")).strip()
    genes = _split_multi_values(str(row.get("Genes", "")).strip())
    mutation = str(row.get("Mutation", "")).strip()
    antibiotic = str(row.get("Antibiotic", "")).strip()
    ab_class = str(row.get("Class of Antibiotic", "")).strip()
    mechanism = str(row.get("Mechanism of resistance", "")).strip()
    susceptibility = str(row.get("Susceptibility", "")).strip().lower()
    smiles = str(row.get("SMILES", "")).strip()
    std_type = str(row.get("standard_type", "")).strip()
    std_value = str(row.get("standard_value", "")).strip()
    std_units = str(row.get("standard_units", "")).strip()

    # Skip rows with no meaningful data, or rows where a source CSV header
    # leaked into the data (e.g. a stray "Organism" cell containing the
    # literal string "Organism").
    if not organism or organism in ["nan", "", "Organism"]:
        return triples

    # Organism → Gene
    for gene in genes:
        triples.append((organism, "has_gene", gene))

    # Gene → Mutation
    if genes and mutation and mutation != "nan":
        for gene in genes:
            triples.append((gene, "has_mutation", mutation))

    # Resistance/Susceptibility relations
    if antibiotic and antibiotic != "nan":
        if "resistant" in susceptibility or "resistance" in susceptibility:
            triples.append((organism, "resistant_to", antibiotic))
            for gene in genes:
                triples.append((gene, "confers_resistance_to", antibiotic))
        elif "sensitive" in susceptibility or "susceptible" in susceptibility:
            triples.append((organism, "sensitive_to", antibiotic))

        # Antibiotic → Class
        if ab_class and ab_class != "nan":
            triples.append((antibiotic, "belongs_to_class", ab_class))

        # Antibiotic → SMILES
        if smiles and smiles != "nan" and len(smiles) > 3:
            triples.append((antibiotic, "has_structure", smiles))

        # Antibiotic targets Gene
        for gene in genes:
            triples.append((antibiotic, "targeted_by", gene))

    # Gene → Mechanism
    if genes and mechanism and mechanism != "nan":
        for gene in genes:
            triples.append((gene, "has_mechanism", mechanism))

    # Strain relations
    if strain and strain != "nan":
        triples.append((organism, "has_strain", strain))

    # Quantitative measurements
    if std_type and std_value and std_type != "nan" and std_value != "nan":
        measurement_id = _stable_measurement_id(
            organism, ",".join(genes), antibiotic, std_type, std_value
        )
        triples.append((organism, "has_measurement", measurement_id))
        triples.append((measurement_id, "has_type", std_type))
        triples.append((measurement_id, "has_value", std_value))
        if std_units and std_units != "nan":
            triples.append((measurement_id, "has_unit", std_units))

    return triples


def main():
    parser = argparse.ArgumentParser(description="Build antibiotic resistance knowledge graph")
    parser.add_argument("--input", required=True, help="Standardized database CSV")
    parser.add_argument("--output", required=True, help="Output KG triples (TSV)")
    parser.add_argument("--gene-interaction", default=None,
                        help="Optional: gene-gene interaction file for regulatory triples")
    parser.add_argument("--external-kg", nargs="+", default=None,
                        help="Optional: one or more external KG triple files (TSV/CSV) to merge "
                             "in, e.g. DrugBank/CARD exports or a KG from another pipeline. "
                             "Columns are auto-detected via common aliases "
                             "(Subject/Predicate/Object, head/relation/tail, source/relationship/target); "
                             "header-less 3-column files are also accepted.")
    args = parser.parse_args()

    logger.info(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Records: {len(df)}")

    # Build triples from main database
    all_triples = []
    for _, row in df.iterrows():
        triples = build_triples_from_row(row)
        all_triples.extend(triples)

    logger.info(f"Triples from main database: {len(all_triples)}")

    # Add gene-gene interactions if provided
    if args.gene_interaction and os.path.exists(args.gene_interaction):
        logger.info(f"Loading gene interactions: {args.gene_interaction}")
        ggi = pd.read_csv(args.gene_interaction)
        n_before = len(all_triples)
        # Column names vary across sources; try the known aliases in order.
        subj_col = next((c for c in ("Subject", "TF", "source") if c in ggi.columns), None)
        obj_col = next((c for c in ("Object", "TG", "target") if c in ggi.columns), None)
        pred_col = next((c for c in ("Predicate", "regulation", "type") if c in ggi.columns), None)
        if not subj_col or not obj_col:
            logger.warning(
                f"Gene-interaction file missing recognizable subject/object columns "
                f"(found: {list(ggi.columns)}); skipping this source."
            )
        else:
            for _, row in ggi.iterrows():
                tf = str(row.get(subj_col, "")).strip()
                tg = str(row.get(obj_col, "")).strip()
                relation = str(row.get(pred_col, "regulates")).strip() if pred_col else "regulates"
                if tf and tg and tf != "nan" and tg != "nan":
                    all_triples.append((tf, relation, tg))
            logger.info(f"Gene-interaction triples added: {len(all_triples) - n_before}")
        logger.info(f"Total triples (with interactions): {len(all_triples)}")

    # Merge in external KG(s), if provided
    if args.external_kg:
        for ext_path in args.external_kg:
            if not os.path.exists(ext_path):
                logger.warning(f"External KG file not found, skipping: {ext_path}")
                continue
            logger.info(f"Loading external KG: {ext_path}")
            n_before = len(all_triples)
            ext_triples = load_external_triples(ext_path)
            all_triples.extend(ext_triples)
            logger.info(f"External triples added from '{ext_path}': {len(all_triples) - n_before}")
        logger.info(f"Total triples (with external KG): {len(all_triples)}")

    # Deduplicate
    unique_triples = list(set(all_triples))
    logger.info(f"Unique triples: {len(unique_triples)}")

    # Save
    kg_df = pd.DataFrame(unique_triples, columns=["Subject", "Predicate", "Object"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    kg_df.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Saved KG: {args.output}")

    # Print statistics
    logger.info(f"\n{'='*50}")
    logger.info(f"Knowledge Graph Statistics:")
    logger.info(f"  Triples: {len(kg_df):,}")
    logger.info(f"  Unique subjects: {kg_df['Subject'].nunique():,}")
    logger.info(f"  Unique predicates: {kg_df['Predicate'].nunique():,}")
    logger.info(f"  Unique objects: {kg_df['Object'].nunique():,}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
