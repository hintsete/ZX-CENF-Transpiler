"""Ingestion. 
Load a QASM file and convert it into our MultiGraph schema.
"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import pyzx as zx
from pyzx.utils import EdgeType, VertexType

# Maps PyZX's internal vertex type enum to our schema's string labels.
# Anything not Z or X (i.e. boundary) is treated as 'B'.
_VERTEX_TYPE_MAP = {
    VertexType.Z: "Z",
    VertexType.X: "X",
}


def _pyzx_type_to_schema(vtype) -> str:
    return _VERTEX_TYPE_MAP.get(vtype, "B")


def _pyzx_edge_type_to_schema(etype) -> str:
    return "Hadamard" if etype == EdgeType.HADAMARD else "Standard"


def load_qasm_to_multigraph(path: str) -> nx.MultiGraph:
    """Load a QASM file and translate it into our custom MultiGraph schema.

    Args:
        path: filesystem path to a .qasm file.

    Returns:
        networkx.MultiGraph where:
            - each node has attrs: id (int), type ('Z'|'X'|'B'),
              phase (Fraction), cenf_cluster_id (None)
            - each edge has attrs: source (int), target (int),
              type ('Standard'|'Hadamard')

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if PyZX fails to parse the QASM file.
    """
    try:
        circuit = zx.Circuit.load(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pyzx raises plain Exception/ValueError on bad QASM
        raise ValueError(f"PyZX failed to parse QASM file '{path}': {exc}") from exc

    pyzx_graph = circuit.to_graph()

    G: nx.MultiGraph = nx.MultiGraph()

    for v in pyzx_graph.vertices():
        schema_type = _pyzx_type_to_schema(pyzx_graph.type(v))
        raw_phase = pyzx_graph.phase(v)
        phase = Fraction(raw_phase) if raw_phase else Fraction(0)

        G.add_node(
            v,
            id=v,
            type=schema_type,
            phase=phase,
            cenf_cluster_id=None,
        )

    for e in pyzx_graph.edges():
        s, t = pyzx_graph.edge_st(e)
        schema_edge_type = _pyzx_edge_type_to_schema(pyzx_graph.edge_type(e))

        G.add_edge(
            s,
            t,
            source=s,
            target=t,
            type=schema_edge_type,
        )

    return G
