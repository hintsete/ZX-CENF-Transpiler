"""Outcome Matrix — Task 2 of the CENF clustering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_outcome_matrix(
    evo_df: pd.DataFrame,
    encoded_wl: dict[tuple[int, int], int],
    dead_label: int,
    circuit_name: str,
) -> tuple[np.ndarray, list[int]]:
    """Build the (N x R) joint outcome matrix for one circuit.

    Each cell [spider_idx, run] holds an integer label:
        0..K-1  : survived in a top-K WL neighbourhood category
        K       : survived but WL hash was rare (outside top-K)
        K+1     : did not survive (dead)

    Parameters
    ----------
    evo_df:
        DataFrame from spider_evolutionary_features.csv.
        Required columns: circuit_name, spider_id, survival_vector.
    encoded_wl:
        Dict[(spider_id, run_index)] -> int label from encode_wl().
    dead_label:
        Integer label for dead runs (= K+1).
    circuit_name:
        The circuit to build the matrix for.

    Returns
    -------
    matrix     : np.ndarray shape (N, R), dtype int32
    spider_ids : list[int] of length N — row index to spider_id mapping
    """
    circuit_df = evo_df[evo_df["circuit_name"] == circuit_name].copy()
    spider_ids: list[int] = sorted(circuit_df["spider_id"].tolist())
    spider_idx: dict[int, int] = {sid: i for i, sid in enumerate(spider_ids)}

    first_vec = circuit_df.iloc[0]["survival_vector"]
    R = len(first_vec.split(","))
    N = len(spider_ids)

    matrix = np.full((N, R), fill_value=dead_label, dtype=np.int32)

    for _, row in circuit_df.iterrows():
        sid = int(row["spider_id"])
        idx = spider_idx[sid]
        bits = [int(b) for b in row["survival_vector"].split(",")]
        for r, bit in enumerate(bits):
            if bit == 1:
                label = encoded_wl.get((sid, r), dead_label - 1)
                matrix[idx, r] = label

    return matrix, spider_ids
