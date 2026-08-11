"""Control group: confluence check on the baseline rule set
(self-loop, spider fusion, identity removal, parallel-edge wipe).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from zx_cenf.rewrite import rules

RULE_NAMES = ("self_loop", "spider_fusion", "identity_removal", "parallel_wipe")

_FIND_FNS = {
    "self_loop": rules.find_all_self_loops,
    "spider_fusion": rules.find_all_spider_fusions,
    "identity_removal": rules.find_all_identity_removals,
    "parallel_wipe": rules.find_all_parallel_wipes,
}
_APPLY_FNS = {
    "self_loop": rules.apply_self_loop_match,
    "spider_fusion": rules.apply_spider_fusion_match,
    "identity_removal": rules.apply_identity_removal_match,
    "parallel_wipe": rules.apply_parallel_wipe_match,
}


@dataclass
class ControlRunResult:
    seed: int
    spider_count: int
    edge_count: int


def run_to_fixed_point_random(graph: nx.MultiGraph, seed: int) -> None:
    rng = random.Random(seed)
    while True:
        order = list(RULE_NAMES)
        rng.shuffle(order)
        applied = False
        for name in order:
            matches = _FIND_FNS[name](graph)
            if matches:
                _APPLY_FNS[name](graph, rng.choice(matches))
                applied = True
                break
        if not applied:
            break


def score_control(graph: nx.MultiGraph, diagram_id: str, n_orderings: int = 30) -> dict:
    counts = []
    for seed in range(n_orderings):
        g = graph.copy()
        run_to_fixed_point_random(g, seed)
        counts.append(g.number_of_nodes())
    mean = sum(counts) / len(counts)
    return {
        "diagram_id": diagram_id,
        "control_min_spiders": min(counts),
        "control_max_spiders": max(counts),
        "control_ambiguity_score": 0.0 if mean == 0 else (max(counts) - min(counts)) / mean,
        "control_is_ambiguous": len(set(counts)) > 1,
    }