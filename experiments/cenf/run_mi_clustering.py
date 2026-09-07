"""Track 2 entry point — Spider MI Clustering.

Reads the two Track 1 output CSVs, runs the full MI clustering pipeline
for every circuit, and writes results to data/track2_results/.

Usage (from project root):
    uv run experiments/cenf/run_mi_clustering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zx_cenf.cenf.wl_encoder import encode_wl
from zx_cenf.cenf.outcome_matrix import build_outcome_matrix
from zx_cenf.cenf.mi_matrix import compute_nmi_matrix
from zx_cenf.cenf.cluster import cluster_spiders
from zx_cenf.cenf.interpret import interpret_clusters

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
EVO_CSV   = _ROOT / "data" / "track1_results" / "spider_evolutionary_features.csv"
NBHD_CSV  = _ROOT / "data" / "track1_results" / "spider_run_neighborhoods.csv"
OUT_DIR   = _ROOT / "data" / "track2_results"
WL_TOP_K  = 10   # vocabulary size per circuit


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Track 1 outputs...")
    evo_df  = pd.read_csv(EVO_CSV)
    nbhd_df = pd.read_csv(NBHD_CSV)

    # ── Task 1 — WL encoding ───────────────────────────────────────────
    print(f"Encoding WL hashes (top-{WL_TOP_K} per circuit)...")
    wl_encoded = encode_wl(nbhd_df, k=WL_TOP_K)

    circuits = evo_df["circuit_name"].unique().tolist()
    print(f"Circuits: {circuits}\n")

    all_labels: list[dict] = []
    all_summaries: list[pd.DataFrame] = []
    nmi_archive: dict[str, np.ndarray] = {}

    for circuit in circuits:
        print(f"-- {circuit} " + "-" * 30)

        encoded, dead_label = wl_encoded.get(circuit, ({}, WL_TOP_K + 1))

        # ── Task 2 — Outcome matrix ────────────────────────────────────
        matrix, spider_ids = build_outcome_matrix(
            evo_df, encoded, dead_label, circuit
        )
        N, R = matrix.shape
        print(f"  Spiders: {N}  Runs: {R}")

        # ── Task 3 — NMI matrix ────────────────────────────────────────
        print("  Computing NMI matrix...")
        nmi = compute_nmi_matrix(matrix)
        nmi_archive[circuit] = nmi

        # ── Task 4 — Clustering ────────────────────────────────────────
        labels, best_k, score = cluster_spiders(nmi, outcome_matrix=matrix, dead_label=dead_label, k=None, max_k=6)
        print(f"  Best k={best_k}  silhouette={score:.4f}")

        # ── Task 5 — Interpretation ────────────────────────────────────
        tables = interpret_clusters(
            labels, spider_ids, matrix, evo_df, circuit, dead_label, WL_TOP_K
        )
        all_summaries.append(tables["summary"].reset_index())

        print("  Outcome profile:")
        print(tables["outcome_profile"].to_string())
        print("\n  Spider-type cross-tab:")
        print(tables["input_crosstab_type"].to_string())
        print("\n  Phase cross-tab:")
        print(tables["input_crosstab_phase"].to_string())
        print()

        # Collect labels
        for spider_id, lbl in zip(spider_ids, labels.tolist()):
            all_labels.append({
                "circuit_name": circuit,
                "spider_id": spider_id,
                "cluster_label": lbl,
            })

    # ── Write outputs ──────────────────────────────────────────────────
    labels_path = OUT_DIR / "spider_cluster_labels.csv"
    pd.DataFrame(all_labels).to_csv(labels_path, index=False)
    print(f"Wrote cluster labels -> {labels_path}")

    summary_path = OUT_DIR / "cluster_interpretation.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Wrote interpretation  -> {summary_path}")

    nmi_path = OUT_DIR / "nmi_matrices.npz"
    np.savez(nmi_path, **{k.replace("-", "_"): v for k, v in nmi_archive.items()})
    print(f"Wrote NMI matrices   -> {nmi_path}")


if __name__ == "__main__":
    main()
