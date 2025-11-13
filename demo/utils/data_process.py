import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pathlib import Path


def parse_gcn_result_txt(
    txt_path,
    method_name="gcn",
    source_name="AResKG",
    keep_metrics=("precision", "recall", "f1", "map")
):
    """
    Parse block-structured result files such as 'res_gcn.txt'.

    Each block (separated by blank lines) corresponds to one fold.
    Returns a long-format DataFrame with columns:
        [Method, Fold, Metric, Score, Source]
    """
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")

    # Split the file by blank lines; each block represents one fold.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", txt) if b.strip()]
    rows = []

    for i, block in enumerate(blocks):
        fold = f"fold_{i}"
        # Extract key:value pairs such as "precision: 0.83"
        for line in block.splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9.eE+-]+)", line.strip())
            if not m:
                continue
            key, val = m.group(1).lower(), float(m.group(2))

            # Normalize metric names (e.g., pr_auc → auc_pr)
            if key == "pr_auc":
                metric = "auc_pr"
            else:
                metric = key

            # Keep only the specified metrics
            if metric in keep_metrics:
                rows.append({
                    "Method": method_name,
                    "Fold": fold,
                    "Metric": metric,
                    "Score": val,
                    "Source": source_name
                })

    df = pd.DataFrame(rows, columns=["Method", "Fold", "Metric", "Score", "Source"])
    return df


def extract_predict_results(data_path, strain, methods, folds, metrics_of_interest):
    """
    Extract model prediction results across folds and methods.
    Each 'results.txt' file is searched for metrics of interest.

    Parameters
    ----------
    data_path : str or Path
        Base directory containing model outputs.
    strain : str
        Strain name (used as 'Source' in the output table).
    methods : list of str
        Model names (e.g., ['pra', 'gcn', 'gat']).
    folds : list of str
        Fold identifiers (e.g., ['fold_0', 'fold_1']).
    metrics_of_interest : list of str
        Metric names to extract (e.g., ['precision','recall','f1','auc_pr']).
    """
    data = []

    for method in methods:
        for fold in folds:
            # PRA has a slightly different directory structure
            if method == 'pra':
                result_file = os.path.join(
                    data_path, method, "model_instance", fold, "instance", "test", "results", "results.txt"
                )
            else:
                result_file = os.path.join(
                    data_path, method, "model_instance", fold, "test", "results", "results.txt"
                )

            if not os.path.exists(result_file):
                print(f"Missing: {result_file}")
                continue

            with open(result_file, "r") as f:
                content = f.read()

            results = {}
            for metric in metrics_of_interest:
                match = re.search(fr"{metric}:\s*([0-9.]+)", content)
                if match:
                    results[metric] = float(match.group(1))
                else:
                    results[metric] = None

            for metric, value in results.items():
                data.append({
                    "Method": method,
                    "Fold": fold,
                    "Metric": metric,
                    "Score": value,
                    "Source": strain
                })
    return pd.DataFrame(data)


def significance_stars(p):
    """
    Return conventional significance-star notation for a p-value.
    """
    if p <= 1e-4:
        return "****"
    elif p <= 1e-3:
        return "***"
    elif p <= 1e-2:
        return "**"
    elif p <= 0.05:
        return "*"
    return None
