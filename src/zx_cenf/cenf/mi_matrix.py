"""Pairwise NMI Matrix — Task 3 of the CENF clustering pipeline."""

from __future__ import annotations

import numpy as np


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _mi_pair(col_u: np.ndarray, col_v: np.ndarray, n_labels: int) -> float:
    R = len(col_u)
    joint = np.zeros((n_labels, n_labels), dtype=np.int64)
    for r in range(R):
        joint[col_u[r], col_v[r]] += 1

    p_joint = joint / R
    p_u = joint.sum(axis=1) / R
    p_v = joint.sum(axis=0) / R

    mi = 0.0
    for a in range(n_labels):
        for b in range(n_labels):
            pab = p_joint[a, b]
            if pab > 0 and p_u[a] > 0 and p_v[b] > 0:
                mi += pab * np.log(pab / (p_u[a] * p_v[b]))
    return float(mi)


def compute_nmi_matrix(outcome_matrix: np.ndarray) -> np.ndarray:
    """Compute the N x N NMI matrix for a circuit.

    For each pair (u, v), mutual information is estimated from the 30
    paired outcome samples and normalised:
        NMI(u, v) = MI(u, v) / sqrt(H(u) * H(v))

    Parameters
    ----------
    outcome_matrix : np.ndarray shape (N, R)

    Returns
    -------
    nmi : np.ndarray shape (N, N), float64, symmetric, diagonal = 1.0
    """
    N, R = outcome_matrix.shape
    n_labels = int(outcome_matrix.max()) + 1

    entropies = np.zeros(N)
    for i in range(N):
        counts = np.bincount(outcome_matrix[i], minlength=n_labels)
        entropies[i] = _entropy(counts)

    nmi = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(nmi, 1.0)

    for i in range(N):
        for j in range(i + 1, N):
            denom = np.sqrt(entropies[i] * entropies[j])
            if denom < 1e-12:
                nmi[i, j] = nmi[j, i] = 0.0
            else:
                mi = _mi_pair(outcome_matrix[i], outcome_matrix[j], n_labels)
                nmi[i, j] = nmi[j, i] = min(mi / denom, 1.0)

    return nmi
