"""Orchestrates PyZX ambiguity sweep + confluent control,
written to a single CSV"""

from __future__ import annotations

import csv
from pathlib import Path

import pyzx as zx
from tqdm import tqdm

from zx_cenf.ambiguity.rules_py_scoring import score_control
from zx_cenf.ambiguity.scoring import score_diagram
from zx_cenf.ingestion.loader import load_qasm_to_multigraph
from zx_cenf.ambiguity.tagging import tag_diagram_spiders



def _flatten_summary(prefix: str, summary: dict | None) -> dict:
    """Flatten metric summaries into CSV columns."""
    keys = (
        "min",
        "max",
        "mean",
        "n",
        "ambiguity_pct",
        "label",
    )

    if summary is None:
        return {f"{prefix}_{k}": None for k in keys}

    return {f"{prefix}_{k}": summary[k] for k in keys}


def run_track1(
    qasm_paths,
    output_csv: Path,
    spider_tags_csv: Path | None = None,
    n_orderings: int = 30,
) -> None:
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if spider_tags_csv is not None:
        spider_tags_csv.parent.mkdir(parents=True, exist_ok=True)
 
    rows = []
    all_spider_tags = []

    for path in tqdm(qasm_paths, desc="Track 1: ambiguity scoring"):
        diagram_id = path.stem
 
        circuit = zx.Circuit.load(str(path))
        g = circuit.to_graph()
        result = score_diagram(g, diagram_id, n_orderings=n_orderings)
 
        nx_graph = load_qasm_to_multigraph(str(path))
        control = score_control(nx_graph, diagram_id, n_orderings=n_orderings)
 
        
        spider_tags = tag_diagram_spiders(g, diagram_id, n_orderings=n_orderings)
        all_spider_tags.extend(spider_tags)
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
    
 