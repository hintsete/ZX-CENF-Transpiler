""" 
Tags each spider in a diagram by checking how often it survives into the final diagram after multiple randomized-order simplifications.
"""


from __future__ import annotations

from dataclasses import dataclass

from pyzx.utils import VertexType

from zx_cenf.ambiguity.pyzx_ordering import run_randomized_pass_order


@dataclass
class SpiderAmbiguityTag:
    diagram_id: str
    vertex_id: int
    input_type: str
    input_phase: str
    input_degree: int
    n_runs: int
    presence_count: int
    presence_rate: float
    ambiguity_score: float
    is_ambiguous: bool


def _vertex_type_name(g, v) -> str:
    t = g.type(v)
    if t == VertexType.Z:
        return "Z"
    if t == VertexType.X:
        return "X"
    return "B"


def tag_diagram_spiders(
    g_original,
    diagram_id: str,
    n_orderings: int = 30,
    base_seed: int = 0,
    ambiguity_threshold: float = 0.3,
) -> list[SpiderAmbiguityTag]:
    
    original_ids = set(g_original.vertices())
    presence_counts = {v: 0 for v in original_ids}

    for i in range(n_orderings):
        g_copy = g_original.copy()
        run_randomized_pass_order(g_copy, seed=base_seed + i)
        surviving = set(g_copy.vertices())
        for v in original_ids:
            if v in surviving:
                presence_counts[v] += 1

    tags = []
    for v in sorted(original_ids):
        p = presence_counts[v] / n_orderings
        ambiguity = 4 * p * (1 - p)
        tags.append(
            SpiderAmbiguityTag(
                diagram_id=diagram_id,
                vertex_id=v,
                input_type=_vertex_type_name(g_original, v),
                input_phase=str(g_original.phase(v)),
                input_degree=len(list(g_original.neighbors(v))),
                n_runs=n_orderings,
                presence_count=presence_counts[v],
                presence_rate=round(p, 4),
                ambiguity_score=round(ambiguity, 4),
                is_ambiguous=ambiguity >= ambiguity_threshold,
            )
        )
    return tags