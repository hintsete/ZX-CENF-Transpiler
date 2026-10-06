"""Run the CENF clustering pipeline on the 60-circuit natural population.


"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from zx_cenf.cenf.cluster import cluster_spiders
from zx_cenf.cenf.interpret import interpret_clusters
from zx_cenf.cenf.mi_matrix import compute_nmi_matrix
from zx_cenf.cenf.outcome_matrix import build_outcome_matrix
from zx_cenf.cenf.wl_encoder import encode_wl


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "track1_results" / "natural"
OUTPUT_DIR = ROOT / "data" / "track2_results" / "natural"
EVO_CSV = INPUT_DIR / "spider_evolutionary_features.csv"
NBHD_CSV = INPUT_DIR / "spider_run_neighborhoods.csv"
WL_TOP_K = 10
MAX_CLUSTERS = 6


def _circuit_fields(circuit_name: str) -> tuple[str, str]:
    split, remainder = circuit_name.split("_", maxsplit=1)
    family, _index = remainder.rsplit("_", maxsplit=1)
    return split, family


def _off_diagonal_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) < 2:
        return np.array([], dtype=float)
    submatrix = matrix[np.ix_(indices, indices)]
    return submatrix[np.triu_indices(len(indices), k=1)]


def main() -> None:
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading natural-population Track 1 outputs...")
    evo_df = pd.read_csv(EVO_CSV)
    nbhd_df = pd.read_csv(NBHD_CSV)
    wl_encoded = encode_wl(nbhd_df, k=WL_TOP_K)
    circuits = sorted(evo_df["circuit_name"].unique().tolist())

    all_labels: list[dict] = []
    all_interpretations: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    circuit_summaries: list[dict] = []
    nmi_archive: dict[str, np.ndarray] = {}

    for position, circuit in enumerate(circuits, start=1):
        circuit_started = perf_counter()
        encoded, dead_label = wl_encoded.get(circuit, ({}, WL_TOP_K + 1))
        outcome_matrix, spider_ids = build_outcome_matrix(
            evo_df, encoded, dead_label, circuit
        )
        nmi = compute_nmi_matrix(outcome_matrix)
        labels, selected_k, silhouette = cluster_spiders(
            nmi,
            outcome_matrix=outcome_matrix,
            dead_label=dead_label,
            k=None,
            max_k=MAX_CLUSTERS,
        )
        tables = interpret_clusters(
            labels,
            spider_ids,
            outcome_matrix,
            evo_df,
            circuit,
            dead_label,
            WL_TOP_K,
        )

        interpretation = tables["summary"].reset_index()
        all_interpretations.append(interpretation)
        profile = tables["outcome_profile"].reset_index()
        profile.insert(0, "circuit_name", circuit)
        all_profiles.append(profile)
        nmi_archive[circuit] = nmi

        for spider_id, label in zip(spider_ids, labels.tolist()):
            all_labels.append(
                {
                    "circuit_name": circuit,
                    "spider_id": spider_id,
                    "cluster_label": label,
                }
            )

        informative = np.flatnonzero(labels >= 0)
        pair_nmi = _off_diagonal_values(nmi, informative)
        split, family = _circuit_fields(circuit)
        effective_clusters = len(set(labels[informative].tolist()))
        circuit_summaries.append(
            {
                "circuit_name": circuit,
                "split": split,
                "family": family,
                "n_vertices": int(outcome_matrix.shape[0]),
                "n_runs": int(outcome_matrix.shape[1]),
                "n_informative": int(len(informative)),
                "n_always_dead": int(np.sum(labels == -1)),
                "n_constant_alive": int(np.sum(labels == -2)),
                "selected_k": int(selected_k),
                "effective_clusters": int(effective_clusters),
                "silhouette": float(silhouette),
                "mean_informative_pair_nmi": (
                    float(pair_nmi.mean()) if pair_nmi.size else np.nan
                ),
                "median_informative_pair_nmi": (
                    float(np.median(pair_nmi)) if pair_nmi.size else np.nan
                ),
                "max_informative_pair_nmi": (
                    float(pair_nmi.max()) if pair_nmi.size else np.nan
                ),
                "elapsed_seconds": perf_counter() - circuit_started,
            }
        )
        print(
            f"[{position:02d}/{len(circuits)}] {circuit}: "
            f"N={outcome_matrix.shape[0]}, informative={len(informative)}, "
            f"k={effective_clusters}, silhouette={silhouette:.4f}"
        )

    labels_df = pd.DataFrame(all_labels)
    labels_df.to_csv(OUTPUT_DIR / "spider_cluster_labels.csv", index=False)
    pd.concat(all_interpretations, ignore_index=True).to_csv(
        OUTPUT_DIR / "cluster_interpretation.csv", index=False
    )
    pd.concat(all_profiles, ignore_index=True).to_csv(
        OUTPUT_DIR / "cluster_outcome_profiles.csv", index=False
    )
    summary_df = pd.DataFrame(circuit_summaries)
    summary_df.to_csv(OUTPUT_DIR / "circuit_summary.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "nmi_matrices.npz",
        **{name.replace("-", "_"): value for name, value in nmi_archive.items()},
    )

    metadata = {
        "dataset": "natural",
        "input_evolutionary_features": str(EVO_CSV.relative_to(ROOT)),
        "input_neighborhoods": str(NBHD_CSV.relative_to(ROOT)),
        "n_circuits": len(circuits),
        "n_vertices": int(summary_df["n_vertices"].sum()),
        "n_runs_per_circuit": sorted(summary_df["n_runs"].unique().tolist()),
        "n_pairwise_nmi_calculations": int(
            sum(n * (n - 1) // 2 for n in summary_df["n_vertices"])
        ),
        "wl_top_k": WL_TOP_K,
        "dead_label": WL_TOP_K + 1,
        "max_clusters": MAX_CLUSTERS,
        "total_elapsed_seconds": perf_counter() - started,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote natural-population CENF results to {OUTPUT_DIR}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
