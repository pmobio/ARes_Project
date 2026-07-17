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
        [--gene-interaction ../data_demo/0_raw_data/gene_gene_interaction.csv]

Output: TSV file with columns: Subject, Predicate, Object
"""

import os
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_triples_from_row(row: pd.Series) -> list:
    """Generate knowledge graph triples from a single database row."""
    triples = []

    organism = str(row.get("Organism", "")).strip()
    strain = str(row.get("Strain", "")).strip()
    gene = str(row.get("Genes", "")).strip()
    mutation = str(row.get("Mutation", "")).strip()
    antibiotic = str(row.get("Antibiotic", "")).strip()
    ab_class = str(row.get("Class", "")).strip()
    mechanism = str(row.get("Mechanism", "")).strip()
    susceptibility = str(row.get("Susceptibility", "")).strip().lower()
    smiles = str(row.get("SMILES", "")).strip()
    std_type = str(row.get("standard_type", "")).strip()
    std_value = str(row.get("standard_value", "")).strip()
    std_units = str(row.get("standard_units", "")).strip()

    # Skip rows with no meaningful data
    if not organism or organism in ["nan", ""]:
        return triples

    # Organism → Gene
    if gene and gene != "nan":
        triples.append((organism, "has_gene", gene))

    # Gene → Mutation
    if gene and mutation and mutation != "nan":
        triples.append((gene, "has_mutation", mutation))

    # Resistance/Susceptibility relations
    if antibiotic and antibiotic != "nan":
        if "resistant" in susceptibility or "resistance" in susceptibility:
            triples.append((organism, "resistant_to", antibiotic))
            if gene and gene != "nan":
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
        if gene and gene != "nan":
            triples.append((antibiotic, "targeted_by", gene))

    # Gene → Mechanism
    if gene and mechanism and mechanism != "nan":
        triples.append((gene, "has_mechanism", mechanism))

    # Strain relations
    if strain and strain != "nan":
        triples.append((organism, "has_strain", strain))

    # Quantitative measurements
    if std_type and std_value and std_type != "nan" and std_value != "nan":
        measurement_id = f"Measurement_{hash((organism, gene, antibiotic, std_type)) % 10**8}"
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
        for _, row in ggi.iterrows():
            tf = str(row.get("TF", row.get("source", ""))).strip()
            tg = str(row.get("TG", row.get("target", ""))).strip()
            relation = str(row.get("regulation", row.get("type", "regulates"))).strip()
            if tf and tg and tf != "nan" and tg != "nan":
                all_triples.append((tf, relation, tg))
        logger.info(f"Total triples (with interactions): {len(all_triples)}")

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
