"""Structural surrogate valuation """

from __future__ import annotations

from fractions import Fraction

from pyzx.utils import EdgeType, VertexType

from zx_cenf.quantale.algebra import QuantaleValue

SURROGATE_COORDINATE_NAMES = ("n_edges", "n_spiders", "n_hadamard_edges", "n_nonclifford")
SURROGATE_DIM = len(SURROGATE_COORDINATE_NAMES)
SURROGATE_PRIORITY = (0, 1, 2, 3)


def _is_nonclifford(phase) -> bool:
    try:
        p = Fraction(phase) % 2
    except Exception:
        return False
    return p not in (Fraction(0), Fraction(1), Fraction(1, 2), Fraction(3, 2))


def surrogate_value(g) -> QuantaleValue:
    n_edges = 0
    n_hadamard = 0
    for e in g.edges():
        s, t = g.edge_st(e)
        if s == t:
            continue
        n_edges += 1
        if g.edge_type(e) == EdgeType.HADAMARD:
            n_hadamard += 1

    n_spiders = 0
    n_nonclifford = 0
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            continue
        n_spiders += 1
        if _is_nonclifford(g.phase(v)):
            n_nonclifford += 1

    return QuantaleValue((float(n_edges), float(n_spiders), float(n_hadamard), float(n_nonclifford)))