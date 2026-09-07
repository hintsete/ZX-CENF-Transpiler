"""Agglomerative Clustering — Task 4 of the CENF clustering pipeline."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def _spider_entropy(row: np.ndarray) -> float:
    counts = np.bincount(row)
    total = counts.sum()
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def cluster_spiders(
    nmi_matrix: np.ndarray,
    outcome_matrix: np.ndarray,
    dead_label: int,
    k: int | None = None,
    max_k: int = 6,
    entropy_threshold: float = 1e-9,
) -> tuple[np.ndarray, int, float]:
    """Cluster informative spiders on the 1-NMI distance matrix.

    Zero-entropy spiders (always dead or always the same outcome) are excluded
    from clustering and assigned fixed labels:
        -1 : always dead
        -2 : always alive in the same WL category

    The remaining informative spiders are clustered with agglomerative
    clustering (average linkage). If k is None, the best k in [2, max_k]
    is selected by silhouette score on the 1-NMI distance matrix.

    Parameters
    ----------
    nmi_matrix      : (N, N) symmetric NMI matrix
    outcome_matrix  : (N, R) integer outcome matrix
    dead_label      : integer label for dead runs (K+1)
    k               : fixed number of clusters, or None for auto-selection
    max_k           : upper bound for auto k search
    entropy_threshold : entropy below this is treated as zero

    Returns
    -------
    labels     : np.ndarray (N,) — 0-based cluster for informative spiders,
                 -1/-2 for excluded spiders
    best_k     : int
    best_score : float silhouette score (-1 if not enough informative spiders)
    """
    N = nmi_matrix.shape[0]

    entropies = np.array([_spider_entropy(outcome_matrix[i]) for i in range(N)])
    informative = np.where(entropies > entropy_threshold)[0]
    zero_entropy = np.where(entropies <= entropy_threshold)[0]

    final_labels = np.full(N, -1, dtype=int)
    for i in zero_entropy:
        final_labels[i] = -1 if np.all(outcome_matrix[i] == dead_label) else -2

    n_info = len(informative)
    if n_info < 2:
        print(f"  Warning: only {n_info} informative spiders — skipping clustering.")
        return final_labels, 1, -1.0

    sub_nmi = nmi_matrix[np.ix_(informative, informative)]
    distance_matrix = np.clip(1.0 - sub_nmi, 0.0, 1.0)
    Z = linkage(squareform(distance_matrix, checks=False), method="average")

    if n_info < 3:
        for idx, spider_idx in enumerate(informative):
            final_labels[spider_idx] = 0
        return final_labels, 1, -1.0

    def _apply(candidate_k: int) -> tuple[np.ndarray, float]:
        raw = fcluster(Z, t=candidate_k, criterion="maxclust")
        sub_labels = (raw - 1).astype(int)
        if len(set(sub_labels)) < 2:
            return sub_labels, -1.0
        return sub_labels, float(silhouette_score(distance_matrix, sub_labels, metric="precomputed"))

    if k is not None:
        sub_labels, score = _apply(k)
        for idx, spider_idx in enumerate(informative):
            final_labels[spider_idx] = sub_labels[idx]
        return final_labels, k, score

    best_k, best_score, best_sub_labels = 2, -1.0, np.zeros(n_info, dtype=int)
    for candidate_k in range(2, min(max_k + 1, n_info)):
        sub_labels, score = _apply(candidate_k)
        if score > best_score:
            best_score, best_k, best_sub_labels = score, candidate_k, sub_labels

    for idx, spider_idx in enumerate(informative):
        final_labels[spider_idx] = best_sub_labels[idx]

    return final_labels, best_k, best_score
