"""Cluster Interpretation Tables — Task 5 of the CENF clustering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _label_name(label: int, dead_label: int, k: int) -> str:
    if label == dead_label:
        return "dead"
    if label == k:
        return "rare"
    return f"wl_cat_{label}"


def interpret_clusters(
    labels: np.ndarray,
    spider_ids: list[int],
    outcome_matrix: np.ndarray,
    evo_df: pd.DataFrame,
    circuit_name: str,
    dead_label: int,
    k: int,
) -> dict:
    """Build interpretation tables for one circuit's clustering result.

    Parameters
    ----------
    labels         : (N,) cluster labels — 0-based for informative spiders,
                     -1 (always dead) or -2 (always alive) for excluded ones
    spider_ids     : ordered list of spider_ids matching rows of outcome_matrix
    outcome_matrix : (N, R) integer outcome matrix
    evo_df         : evolutionary features DataFrame (filtered internally)
    circuit_name   : used for filtering evo_df
    dead_label     : integer label for dead outcome (K+1)
    k              : WL vocabulary size

    Returns
    -------
    dict with keys:
        outcome_profile       : DataFrame — per-cluster outcome distribution
        input_crosstab_type   : DataFrame — spider_type counts per cluster
        input_crosstab_phase  : DataFrame — initial_phase counts per cluster
        summary               : DataFrame — one row per cluster
    """
    N, R = outcome_matrix.shape
    n_labels = dead_label + 1
    label_col_names = [_label_name(i, dead_label, k) for i in range(n_labels)]
    unique_clusters = sorted(set(labels.tolist()))

    # Outcome profile
    profile_rows = []
    for c in unique_clusters:
        mask = labels == c
        outcomes = outcome_matrix[mask].flatten()
        counts = np.bincount(outcomes, minlength=n_labels)
        total = counts.sum()
        row = {"cluster": c, "n_spiders": int(mask.sum())}
        for lbl, col in enumerate(label_col_names):
            row[col] = round(counts[lbl] / total, 4)
        profile_rows.append(row)
    outcome_profile = pd.DataFrame(profile_rows).set_index("cluster")

    # Input cross-tabs
    circ_evo = evo_df[evo_df["circuit_name"] == circuit_name].set_index("spider_id")
    label_series = pd.Series(labels, index=spider_ids, name="cluster")
    merged = circ_evo.join(label_series, how="left")

    type_crosstab = merged.groupby(["cluster", "spider_type"]).size().unstack(fill_value=0)
    phase_crosstab = merged.groupby(["cluster", "initial_phase"]).size().unstack(fill_value=0)

    # Summary
    summary_rows = []
    for c in unique_clusters:
        mask = labels == c
        circ_cluster = merged[merged["cluster"] == c]
        all_outcomes = outcome_matrix[mask].flatten()
        counts = np.bincount(all_outcomes, minlength=n_labels)
        dominant_outcome = _label_name(int(np.argmax(counts)), dead_label, k)
        dominant_type = circ_cluster["spider_type"].value_counts().idxmax() if len(circ_cluster) > 0 else "?"
        summary_rows.append({
            "circuit_name": circuit_name,
            "cluster": c,
            "n_spiders": int(mask.sum()),
            "dominant_outcome": dominant_outcome,
            "dominant_spider_type": dominant_type,
            "mean_survival_rate": round(float(circ_cluster["survival_rate"].mean()), 4),
            "mean_bernoulli_ambiguity": round(float(circ_cluster["bernoulli_ambiguity"].mean()), 4),
        })

    return {
        "outcome_profile": outcome_profile,
        "input_crosstab_type": type_crosstab,
        "input_crosstab_phase": phase_crosstab,
        "summary": pd.DataFrame(summary_rows).set_index("cluster"),
    }
