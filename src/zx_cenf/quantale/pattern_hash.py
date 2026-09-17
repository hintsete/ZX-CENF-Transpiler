"""Shared local-pattern fingerprinting for Phase F."""

from __future__ import annotations

import logging
from fractions import Fraction

import networkx as nx
from pyzx.utils import EdgeType, VertexType


ERROR_PATTERN = "ERROR"

DEFAULT_RADIUS = 1
DEFAULT_ITERATIONS = 1
DEFAULT_INCLUDE_DEGREE = True
DEFAULT_INCLUDE_PHASE = True


def is_valid_pattern(pattern_hash: str) -> bool:
    return bool(pattern_hash) and pattern_hash != ERROR_PATTERN


def phase_class(phase) -> str:
    try:
        p = Fraction(phase) % 2
    except Exception:
        return "unknown"
    if p == 0:
        return "zero"
    if p == 1:
        return "pauli"
    if p in (Fraction(1, 2), Fraction(3, 2)):
        return "clifford"
    return "nonclifford"


def _vertex_type_label(g, v) -> str:
    t = g.type(v)
    if t == VertexType.Z:
        return "Z"
    if t == VertexType.X:
        return "X"
    return "B"


def _pyzx_graph_to_labelled_networkx(g) -> nx.Graph:
    G = nx.Graph()
    for v in g.vertices():
        G.add_node(v, vtype=_vertex_type_label(g, v), pclass=phase_class(g.phase(v)))
    for e in g.edges():
        s, t = g.edge_st(e)
        if s == t:
            continue
        etype = "H" if g.edge_type(e) == EdgeType.HADAMARD else "S"
        existing = G.get_edge_data(s, t)
        if existing is None:
            G.add_edge(s, t, etype=etype)
        elif existing.get("etype") != "H" and etype == "H":
            G[s][t]["etype"] = "H"
    return G


def _build_node_label(ego, node, center_set, include_degree, include_phase, original_graph) -> str:
    data = ego.nodes[node]
    marker = "C" if node in center_set else "n"
    vtype = data.get("vtype", "?")
    parts = [marker, vtype]
    if include_phase:
        parts.append(data.get("pclass", "?"))
    if include_degree:
        degree = original_graph.degree(node)
        parts.append(str(degree))
    return "|".join(parts)


def compute_local_pattern_hash(
    g,
    center_vertices,
    radius: int = DEFAULT_RADIUS,
    iterations: int = DEFAULT_ITERATIONS,
    include_degree: bool = DEFAULT_INCLUDE_DEGREE,
    include_phase: bool = DEFAULT_INCLUDE_PHASE,
) -> str:
    try:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if iterations < 0:
            raise ValueError("iterations must be non-negative")

        nx_g = _pyzx_graph_to_labelled_networkx(g)
        centers = [v for v in center_vertices if v in nx_g]
        if not centers:
            return ERROR_PATTERN

        node_set = set()
        for v in centers:
            node_set |= set(nx.ego_graph(nx_g, v, radius=radius).nodes())
        if not node_set:
            return ERROR_PATTERN

        ego = nx_g.subgraph(node_set).copy()
        center_set = set(centers)
        labels = {
            node: _build_node_label(ego, node, center_set, include_degree, include_phase, nx_g)
            for node in ego.nodes()
        }
        nx.set_node_attributes(ego, labels, name="label")
        return nx.weisfeiler_lehman_graph_hash(ego, node_attr="label", edge_attr="etype", iterations=iterations)
    except Exception:
        logging.getLogger(__name__).warning(
            "pattern hash computation failed for vertices=%r", center_vertices, exc_info=True
        )
        return ERROR_PATTERN