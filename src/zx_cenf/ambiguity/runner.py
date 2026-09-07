"""Orchestrates PyZX ambiguity sweep + confluent control,
written to a single CSV"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyzx as zx
from tqdm import tqdm

from zx_cenf.ambiguity.rules_py_scoring import score_control
from zx_cenf.ambiguity.scoring import build_ambiguity_result, _extract_and_measure
from zx_cenf.ambiguity.tagging import build_spider_tags
from zx_cenf.ambiguity.evolutionary import EvolutionaryTracker
from zx_cenf.ambiguity.pyzx_ordering import run_randomized_pass_order, run_full_reduce_baseline
from zx_cenf.ingestion.loader import load_qasm_to_multigraph


def run_track1(
    qasm_paths,
    output_csv: Path,
    spider_tags_csv: Path | None = None,
    n_orderings: int = 30,
    evolutionary_features_csv: Path | None = None,
    run_neighborhoods_csv: Path | None = None,
) -> None:

    base_seed = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if spider_tags_csv is not None:
        spider_tags_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_spider_tags = []
    all_evo_features: list = []
    all_nbhd_records: list = []

    for path in tqdm(qasm_paths, desc="Track 1: ambiguity scoring"):
        diagram_id = path.stem

        circuit = zx.Circuit.load(str(path))
        g = circuit.to_graph()

        # ── Single shared rewrite loop ───────────────────────────────────────
        run_data: list[tuple[int, Any, dict]] = []
        for i in range(n_orderings):
            seed = base_seed + i
            g_copy = g.copy()
            pass_counts = run_randomized_pass_order(g_copy, seed=seed)
            run_data.append((seed, g_copy, pass_counts))

        # Baseline (full_reduce)
        g_baseline = g.copy()
        run_full_reduce_baseline(g_baseline)
        baseline_result = _extract_and_measure(g_baseline, -1, {})

        # ── Feed consumers with shared G_r copies ───────────────────────────
        result = build_ambiguity_result(g, diagram_id, run_data, baseline_result)
        spider_tags = build_spider_tags(g, diagram_id, [(s, gr) for s, gr, _ in run_data])
        all_spider_tags.extend(spider_tags)

        # ── Evolutionary tracking (only when requested) ──────────────────────
        if evolutionary_features_csv is not None or run_neighborhoods_csv is not None:
            tracker = EvolutionaryTracker(g, diagram_id, n_orderings)
            for i, (seed, g_r, _pass_counts) in enumerate(run_data):
                tracker.observe_run(i, g_r)
            features, nbhd_records = tracker.aggregate()
            all_evo_features.extend(features)
            all_nbhd_records.extend(nbhd_records)

        nx_graph = load_qasm_to_multigraph(str(path))
        control = score_control(nx_graph, diagram_id, n_orderings=n_orderings)

        n_ambiguous = sum(1 for t in spider_tags if t.is_ambiguous)

        row = {
            "diagram_id": diagram_id,
            "input_spiders": g.num_vertices(),
            "input_edges": g.num_edges(),
            "n_orderings": n_orderings,
            "n_successful_extractions": len(result.successful_runs),
            "twoqubitcount_ambiguity_score": result.ambiguity_score("twoqubitcount"),
            "twoqubitcount_is_ambiguous": result.is_ambiguous("twoqubitcount"),
            "tcount_ambiguity_score": result.ambiguity_score("tcount"),
            "depth_ambiguity_score": result.ambiguity_score("depth"),
            "full_reduce_twoqubitcount": result.full_reduce_baseline.twoqubitcount,
            "full_reduce_tcount": result.full_reduce_baseline.tcount,
            "full_reduce_depth": result.full_reduce_baseline.depth,
            "headroom_vs_full_reduce_twoqubitcount": result.headroom_vs_full_reduce("twoqubitcount"),
            "headroom_vs_full_reduce_tcount": result.headroom_vs_full_reduce("tcount"),
            "n_ambiguous_spiders": n_ambiguous,
            "pct_ambiguous_spiders": round(100 * n_ambiguous / g.num_vertices(), 1) if g.num_vertices() else 0.0,
        }
        row.update(control)
        rows.append(row)

    with output_csv.open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if spider_tags_csv is not None and all_spider_tags:
        fieldnames = list(vars(all_spider_tags[0]).keys())
        with spider_tags_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tag in all_spider_tags:
                writer.writerow(vars(tag))

    if evolutionary_features_csv is not None:
        evolutionary_features_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["circuit_name", "spider_id", "spider_type", "initial_phase",
                      "survival_vector", "survival_rate", "bernoulli_ambiguity",
                      "final_degree_variance", "phase_delta_variance"]
        with evolutionary_features_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for feat in all_evo_features:
                writer.writerow(vars(feat))

    if run_neighborhoods_csv is not None:
        run_neighborhoods_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["circuit_name", "spider_id", "run_index", "final_phase",
                      "final_degree", "wl_signature_k2"]
        with run_neighborhoods_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  # always write header even if no records
            for rec in all_nbhd_records:
                writer.writerow(vars(rec))
