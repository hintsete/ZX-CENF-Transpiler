"""WL Category Encoding — Task 1 of the CENF clustering pipeline."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def encode_wl(
    neighborhoods_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, tuple[dict[tuple[int, int], int], int]]:
    """Build a WL hash vocabulary per circuit and encode every (spider, run) pair.

    Parameters
    ----------
    neighborhoods_df:
        DataFrame from spider_run_neighborhoods.csv.
        Required columns: circuit_name, spider_id, run_index, wl_signature_k2.
    k:
        Number of most-frequent WL hashes to keep as named categories.

    Returns
    -------
    dict mapping circuit_name to (encoded, dead_label) where:
        encoded    : dict[(spider_id, run_index)] -> int label
        dead_label : int = k+1, reserved for runs where the spider was dead
    """
    result: dict[str, tuple[dict[tuple[int, int], int], int]] = {}

    for circuit_name, group in neighborhoods_df.groupby("circuit_name"):
        freq = Counter(group["wl_signature_k2"].tolist())
        top_k = [h for h, _ in freq.most_common(k)]
        vocab: dict[str, int] = {h: i for i, h in enumerate(top_k)}
        rare_label = k
        dead_label = k + 1

        encoded: dict[tuple[int, int], int] = {}
        for _, row in group.iterrows():
            label = vocab.get(row["wl_signature_k2"], rare_label)
            encoded[(int(row["spider_id"]), int(row["run_index"]))] = label

        result[str(circuit_name)] = (encoded, dead_label)

    return result
