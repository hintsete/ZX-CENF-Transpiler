"""
Individual rewrite rules.

"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx


def erase_self_loop(graph: nx.MultiGraph) -> bool:
    """Remove one self-loop edge (u == v) if any exist.
    Args:
        graph: the working MultiGraph.

    Returns:
        True if a self-loop was found and removed, False otherwise.
    """
    for u, v, k in graph.edges(keys=True):
        if u == v:
            graph.remove_edge(u, v, key=k)
            return True
    return False


def apply_spider_fusion(graph: nx.MultiGraph) -> bool:
    for u, v, k, data in graph.edges(keys=True, data=True):
        if u == v:
            continue
        if graph.nodes[u].get("type") != "Z" or graph.nodes[v].get("type") != "Z":
            continue
        if data.get("type") != "Standard":
            continue

        # Remove the fusion edge
        graph.remove_edge(u, v, key=k)

        # Snapshot v's remaining edges and reattach them to u
        v_edges = [(nbr, ek, ed) for _, nbr, ek, ed in graph.edges(v, keys=True, data=True)]
        for nbr, ek, ed in v_edges:
            graph.add_edge(u, nbr, type=ed.get("type"))
            graph.remove_edge(v, nbr, key=ek)

        # Merge phases modulo 2 (2π ≡ 0)
        new_phase = (graph.nodes[u]["phase"] + graph.nodes[v]["phase"]) % 2

        graph.remove_node(v)
        graph.nodes[u]["phase"] = new_phase

        return True

    return False


def apply_identity_removal(graph: nx.MultiGraph) -> bool:
    _COMP = {
        ("Standard", "Standard"): "Standard",
        ("Standard", "Hadamard"): "Hadamard",
        ("Hadamard", "Standard"): "Hadamard",
        ("Hadamard", "Hadamard"): "Standard",
    }

    for node, data in graph.nodes(data=True):
        if data.get("type") != "Z" or data.get("phase") != Fraction(0):
            continue
        if graph.degree(node) != 2:
            continue

        # Exclude nodes with self-loops (those count toward degree)
        if any(u == v for u, v, _ in graph.edges(node, keys=True)):
            continue

        edges = list(graph.edges(node, keys=True, data=True))
        if len(edges) != 2:
            continue

        # Extract neighbors and edge types
        nbrs = []
        edge_types = []
        for u, v, _, d in edges:
            nbrs.append(v if u == node else u)
            edge_types.append(d.get("type"))

        stitched_type = _COMP[(edge_types[0], edge_types[1])]

        # Remove the identity node (also removes its two edges)
        graph.remove_node(node)

        if nbrs[0] == nbrs[1]:
            graph.add_edge(nbrs[0], nbrs[0], type=stitched_type)
        else:
            graph.add_edge(nbrs[0], nbrs[1], type=stitched_type)

        return True

    return False


def apply_parallel_edge_wiper(graph: nx.MultiGraph) -> bool:
    for u in graph.nodes():
        for v in graph.neighbors(u):
            if u >= v:
                continue
            if graph.nodes[u].get("type") != "Z" or graph.nodes[v].get("type") != "Z":
                continue

            edge_dict = graph.get_edge_data(u, v)
            if edge_dict is None:
                continue

            hadamard_keys = [k for k, d in edge_dict.items() if d.get("type") == "Hadamard"]
            if len(hadamard_keys) >= 2:
                graph.remove_edge(u, v, key=hadamard_keys[0])
                graph.remove_edge(u, v, key=hadamard_keys[1])
                return True

    return False
# match/apply split for order-randomized reduction


from dataclasses import dataclass


@dataclass(frozen=True)
class SelfLoopMatch:
    u: int
    v: int
    k: int


def find_all_self_loops(graph: nx.MultiGraph) -> list[SelfLoopMatch]:
    return [SelfLoopMatch(u, v, k) for u, v, k in graph.edges(keys=True) if u == v]


def apply_self_loop_match(graph: nx.MultiGraph, match: SelfLoopMatch) -> None:
    graph.remove_edge(match.u, match.v, key=match.k)


@dataclass(frozen=True)
class SpiderFusionMatch:
    u: int
    v: int
    k: int


def find_all_spider_fusions(graph: nx.MultiGraph) -> list[SpiderFusionMatch]:
    matches = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        if u == v:
            continue
        if graph.nodes[u].get("type") != "Z" or graph.nodes[v].get("type") != "Z":
            continue
        if data.get("type") != "Standard":
            continue
        matches.append(SpiderFusionMatch(u, v, k))
    return matches


def apply_spider_fusion_match(graph: nx.MultiGraph, match: SpiderFusionMatch) -> None:
    u, v, k = match.u, match.v, match.k
    graph.remove_edge(u, v, key=k)

    v_edges = [(nbr, ek, ed) for _, nbr, ek, ed in graph.edges(v, keys=True, data=True)]
    for nbr, ek, ed in v_edges:
        graph.add_edge(u, nbr, type=ed.get("type"))
        graph.remove_edge(v, nbr, key=ek)

    new_phase = (graph.nodes[u]["phase"] + graph.nodes[v]["phase"]) % 2
    graph.remove_node(v)
    graph.nodes[u]["phase"] = new_phase


@dataclass(frozen=True)
class IdentityRemovalMatch:
    node: int
    nbr0: int
    nbr1: int
    stitched_type: str


_COMP = {
    ("Standard", "Standard"): "Standard",
    ("Standard", "Hadamard"): "Hadamard",
    ("Hadamard", "Standard"): "Hadamard",
    ("Hadamard", "Hadamard"): "Standard",
}


def find_all_identity_removals(graph: nx.MultiGraph) -> list[IdentityRemovalMatch]:
    matches = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "Z" or data.get("phase") != Fraction(0):
            continue
        if graph.degree(node) != 2:
            continue
        if any(u == v for u, v, _ in graph.edges(node, keys=True)):
            continue

        edges = list(graph.edges(node, keys=True, data=True))
        if len(edges) != 2:
            continue

        nbrs, edge_types = [], []
        for u, v, _, d in edges:
            nbrs.append(v if u == node else u)
            edge_types.append(d.get("type"))

        matches.append(
            IdentityRemovalMatch(node, nbrs[0], nbrs[1], _COMP[(edge_types[0], edge_types[1])])
        )
    return matches


def apply_identity_removal_match(graph: nx.MultiGraph, match: IdentityRemovalMatch) -> None:
    graph.remove_node(match.node)
    if match.nbr0 == match.nbr1:
        graph.add_edge(match.nbr0, match.nbr0, type=match.stitched_type)
    else:
        graph.add_edge(match.nbr0, match.nbr1, type=match.stitched_type)


@dataclass(frozen=True)
class ParallelWipeMatch:
    u: int
    v: int
    key_a: int
    key_b: int


def find_all_parallel_wipes(graph: nx.MultiGraph) -> list[ParallelWipeMatch]:
    matches = []
    seen_pairs = set()
    for u in graph.nodes():
        for v in graph.neighbors(u):
            if u >= v or (u, v) in seen_pairs:
                continue
            seen_pairs.add((u, v))
            if graph.nodes[u].get("type") != "Z" or graph.nodes[v].get("type") != "Z":
                continue

            edge_dict = graph.get_edge_data(u, v)
            if edge_dict is None:
                continue

            hadamard_keys = [k for k, d in edge_dict.items() if d.get("type") == "Hadamard"]
            for i in range(len(hadamard_keys) - 1):
                for j in range(i + 1, len(hadamard_keys)):
                    matches.append(ParallelWipeMatch(u, v, hadamard_keys[i], hadamard_keys[j]))
    return matches


def apply_parallel_wipe_match(graph: nx.MultiGraph, match: ParallelWipeMatch) -> None:
    graph.remove_edge(match.u, match.v, key=match.key_a)
    graph.remove_edge(match.u, match.v, key=match.key_b)