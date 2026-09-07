""" 
Tags each spider in a diagram by checking how often it survives into the final diagram after multiple randomized-order simplifications.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def build_spider_tags(
    g_original,
    diagram_id: str,
    run_data: list[tuple[int, Any]],
    ambiguity_threshold: float = 0.3,
) -> list[SpiderAmbiguityTag]:
    """Build spider ambiguity tags from pre-computed (seed, g_r) pairs.

    Eliminates the redundant second pass over R rewrite runs by accepting
    graph copies already produced by the caller instead of running
    ``run_randomized_pass_order`` internally.

    Parameters
    ----------
    g_original:
        The original ZX graph before any rewrite passes (G_0).
    diagram_id:
        Identifier used in every returned ``SpiderAmbiguityTag``.
    run_data:
        A list of ``(seed, g_r)`` pairs where *g_r* is the simplified graph
        produced by one invocation of ``run_randomized_pass_order``.
    ambiguity_threshold:
        Presence-based ambiguity score threshold; spiders whose score meets
        or exceeds this value are marked ``is_ambiguous=True``.

    Returns
    -------
    list[SpiderAmbiguityTag]
        One tag per spider in *g_original*, sorted by vertex ID.
    """
    n_orderings = len(run_data)
    original_ids = set(g_original.vertices())
    presence_counts = {v: 0 for v in original_ids}

    for _seed, g_r in run_data:
        surviving = set(g_r.vertices())
        for v in original_ids:
            if v in surviving:
                presence_counts[v] += 1

    tags = []
    for v in sorted(original_ids):
        p = presence_counts[v] / n_orderings if n_orderings > 0 else 0.0
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


def tag_diagram_spiders(
    g_original,
    diagram_id: str,
    n_orderings: int = 30,
    base_seed: int = 0,
    ambiguity_threshold: float = 0.3,
) -> list[SpiderAmbiguityTag]:
    """Tag each spider by presence frequency across randomized rewrite runs.

    This function is preserved for backward compatibility. Internally it runs
    the rewrite loop to build ``(seed, g_r)`` pairs and then delegates to
    :func:`build_spider_tags`.

    Parameters
    ----------
    g_original:
        The original ZX graph before any rewrite passes.
    diagram_id:
        Identifier used in every returned ``SpiderAmbiguityTag``.
    n_orderings:
        Number of randomized rewrite runs (R).
    base_seed:
        The first seed value; run *i* uses seed ``base_seed + i``.
    ambiguity_threshold:
        Presence-based ambiguity score threshold.

    Returns
    -------
    list[SpiderAmbiguityTag]
        One tag per spider in *g_original*, sorted by vertex ID.
    """
    run_data: list[tuple[int, Any]] = []
    for i in range(n_orderings):
        seed = base_seed + i
        g_copy = g_original.copy()
        run_randomized_pass_order(g_copy, seed=seed)
        run_data.append((seed, g_copy))

    return build_spider_tags(g_original, diagram_id, run_data, ambiguity_threshold)