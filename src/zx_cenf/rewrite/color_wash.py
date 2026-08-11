"""Color Wash.

Converts all 'X' spiders to 'Z' spiders. Every edge incident to an X-node
must have its type toggled ('Standard' <-> 'Hadamard') exactly once *per
X-endpoint it touches*.


"""

from __future__ import annotations

import networkx as nx


def color_wash(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Mutate `graph` in place: recolor every X-node to Z, per-endpoint edge flip.

    Args:
        graph: MultiGraph following the ZX-CENF node/edge schema. Node dicts
            must have a 'type' key in {'Z', 'X', 'B'}; edge dicts must have
            a 'type' key in {'Standard', 'Hadamard'}.

    Returns:
        The same graph object, mutated in place (returned for chaining).

    Rewrite rule:
        For each edge (u, v, k):
            flip_count = (1 if type(u) == 'X' else 0) + (1 if type(v) == 'X' else 0)
            if flip_count is odd: toggle edge type once
            if flip_count is even (0 or 2): leave edge type unchanged
        Then relabel every X-node's type to 'Z'.
    """
    x_nodes = [n for n, data in graph.nodes(data=True) if data.get("type") == "X"]
    x_node_set = set(x_nodes)

    edges_to_flip: list[tuple[int, int, int]] = []
    for u, v, k in graph.edges(keys=True):
        flip_count = int(u in x_node_set) + int(v in x_node_set)
        if flip_count % 2 == 1:
            edges_to_flip.append((u, v, k))

    for u, v, k in edges_to_flip:
        current_type = graph[u][v][k]["type"]
        graph[u][v][k]["type"] = "Hadamard" if current_type == "Standard" else "Standard"

    for n in x_nodes:
        graph.nodes[n]["type"] = "Z"

    return graph
