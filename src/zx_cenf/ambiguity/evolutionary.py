"""
evolutionary.py — Data models and tracker for per-run spider evolutionary features.

This module defines the two CSV row dataclasses used by the Track 1 evolutionary
exporter pipeline:

- :class:`NeighborhoodRecord` — one row in ``spider_run_neighborhoods.csv``,
  representing a single surviving (spider, run) observation.
- :class:`EvolutionaryFeature` — one row in ``spider_evolutionary_features.csv``,
  representing the aggregate evolutionary summary for one spider across all R runs.

The :class:`EvolutionaryTracker` class accumulates per-run spider state via
:meth:`EvolutionaryTracker.observe_run` and produces both collections via
:meth:`EvolutionaryTracker.aggregate`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx


@dataclass
class NeighborhoodRecord:
    """One row in ``spider_run_neighborhoods.csv``.

    Represents a single surviving (spider, run) observation — i.e. a record is
    only written when spider ``v`` is present in ``G_r.vertices()`` for run ``r``
    (``s_v[r] = 1``).

    Fields are ordered to match the CSV column order defined in Requirement 5.2:
    ``circuit_name``, ``spider_id``, ``run_index``, ``final_phase``,
    ``final_degree``, ``wl_signature_k2``.
    """

    circuit_name: str
    """``path.stem`` of the source ``.qasm`` file that produced this diagram."""

    spider_id: int
    """Original vertex ID of the spider in G_0 (the pre-rewrite graph)."""

    run_index: int
    """0-based index ``r`` of the rewrite run (equals ``seed − base_seed``)."""

    final_phase: str
    """Phase of spider ``v`` in ``G_r``, formatted as ``str(G_r.phase(v))``.

    Consistent with Python :class:`fractions.Fraction.__str__` output,
    e.g. ``"1/2"`` or ``"0"``.
    """

    final_degree: int
    """Number of edges incident to spider ``v`` in ``G_r``."""

    wl_signature_k2: str
    """2-iteration Weisfeiler-Leman hash of the 2-hop ego graph of ``v`` in
    the NetworkX representation of ``G_r``.  Set to ``"ERROR"`` if the
    computation raises any exception (see Requirement 2.5).
    """


@dataclass
class EvolutionaryFeature:
    """One row in ``spider_evolutionary_features.csv``.

    Represents the aggregate evolutionary summary for a single spider ``v``
    across all R rewrite runs.  One row is written for **every** spider in
    G_0, including those that never survive any run (``survival_rate = 0``).

    Fields are ordered to match the CSV column order defined in Requirement 4.2:
    ``circuit_name``, ``spider_id``, ``spider_type``, ``initial_phase``,
    ``survival_vector``, ``survival_rate``, ``bernoulli_ambiguity``,
    ``final_degree_variance``, ``phase_delta_variance``.
    """

    circuit_name: str
    """``path.stem`` of the source ``.qasm`` file that produced this diagram."""

    spider_id: int
    """Original vertex ID of the spider in G_0 (the pre-rewrite graph)."""

    spider_type: str
    """Type of the spider: ``"Z"``, ``"X"``, or ``"B"`` (boundary)."""

    initial_phase: str
    """Phase of spider ``v`` in G_0, formatted as ``str(G_0.phase(v))``."""

    survival_vector: str
    """Comma-separated binary string of length R encoding per-run survival.

    Entry ``r`` is ``"1"`` if ``v`` survived run ``r`` (i.e. ``v`` is in
    ``G_r.vertices()``), otherwise ``"0"``.  Run indices are ordered 0..R-1
    matching ``base_seed + r`` ordering.  Example: ``"1,0,1,1,0"``.
    """

    survival_rate: float
    """Fraction of runs in which spider ``v`` survived: ``sum(s_v) / R``."""

    bernoulli_ambiguity: float
    """Bernoulli ambiguity score: ``4 * survival_rate * (1 - survival_rate)``.

    Equals 1.0 when ``survival_rate = 0.5`` (maximum ambiguity) and 0.0 when
    the spider always survives or never survives.
    """

    final_degree_variance: float
    """Population variance (ddof=0) of the final degree of ``v`` across the
    runs in which it survived.  Set to ``0.0`` when ``survival_rate == 0`` or
    only one run has ``s_v[r] = 1`` (Requirements 3.3, 3.4).
    """

    phase_delta_variance: float
    """Population variance (ddof=0) of the phase delta
    ``(float(phi_v_final[r]) - float(phi_v_initial)) % (2 * pi)`` across the
    runs in which ``v`` survived.  Set to ``0.0`` when ``survival_rate == 0``
    or only one run has ``s_v[r] = 1`` (Requirements 3.5, 3.6).
    """


class EvolutionaryTracker:
    """Accumulates per-run spider state across R rewrite runs and produces
    aggregate evolutionary features and per-run neighborhood records.

    Parameters
    ----------
    g_original : pyzx.Graph
        G_0 — the original graph before any rewrites.  Used to enumerate
        vertices at construction time and to read initial phases during
        :meth:`aggregate`.
    circuit_name : str
        Used as the ``circuit_name`` field in all output rows.
    n_runs : int
        Total number of rewrite runs R.  Used as the survival-rate
        denominator and to validate that :meth:`observe_run` is called
        exactly R times before :meth:`aggregate` is invoked.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, g_original, circuit_name: str, n_runs: int) -> None:
        self._g_original = g_original
        self._circuit_name = circuit_name
        self._n_runs = n_runs
        self._runs_observed: int = 0

        # Pre-allocate per-spider accumulator dicts indexed by vertex ID.
        # All lists have length n_runs so per-run writes are O(1) index
        # assignments rather than appends.
        vertices = list(g_original.vertices())
        self._survival_bits: dict[int, list[int]] = {
            v: [0] * n_runs for v in vertices
        }
        self._final_phases: dict[int, list[str | None]] = {
            v: [None] * n_runs for v in vertices
        }
        self._final_degrees: dict[int, list[int | None]] = {
            v: [None] * n_runs for v in vertices
        }
        self._wl_sigs: dict[int, list[str | None]] = {
            v: [None] * n_runs for v in vertices
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spider_type(g, v) -> str:
        """Return ``"Z"``, ``"X"``, or ``"B"`` for vertex *v* in graph *g*."""
        from pyzx.utils import VertexType

        t = g.type(v)
        if t == VertexType.Z:
            return "Z"
        if t == VertexType.X:
            return "X"
        return "B"

    @staticmethod
    def _compute_wl_signature(
        g_r,
        vertex_id: int,
        circuit_name: str = "",
        run_index: int = -1,
    ) -> str:
        """Compute a 2-iteration Weisfeiler-Leman hash for the 2-hop
        neighbourhood of *vertex_id* in the NetworkX conversion of *g_r*.

        Parameters
        ----------
        g_r : pyzx.Graph
            The simplified graph for this run.
        vertex_id : int
            The spider whose neighbourhood hash is required.
        circuit_name : str
            Used only in the warning message on failure.
        run_index : int
            Used only in the warning message on failure.

        Returns
        -------
        str
            The WL hash string, or ``"ERROR"`` if any exception is raised.
        """
        try:
            # 1. Build a NetworkX graph from g_r via direct adjacency
            #    iteration so that original integer vertex IDs are preserved.
            nx_g: nx.Graph = nx.Graph()
            for v in g_r.vertices():
                nx_g.add_node(v)
            for edge in g_r.edges():
                # pyzx edges are (s, t) or (s, t, edgetype) depending on
                # the graph type; unpack the first two elements only.
                s, t = edge[0], edge[1]
                nx_g.add_edge(s, t)

            # 2. Extract the 2-hop ego graph around vertex_id.
            ego: nx.Graph = nx.ego_graph(nx_g, vertex_id, radius=2)

            # 3. Assign node labels from degree so structurally different
            #    nodes receive different initial colours (Requirement 2.4).
            nx.set_node_attributes(
                ego,
                {n: str(d) for n, d in ego.degree()},
                name="label",
            )

            # 4. Compute the 2-iteration WL hash.
            return nx.weisfeiler_lehman_graph_hash(
                ego, node_attr="label", iterations=2
            )

        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "WL signature computation failed for circuit=%r spider=%r run=%r",
                circuit_name,
                vertex_id,
                run_index,
            )
            return "ERROR"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_run(self, run_index: int, g_r) -> None:
        """Record per-run state for all spiders in G_0 against this G_r.

        For each spider *v* in ``G_0.vertices()``:

        - If *v* is in ``G_r.vertices()``: record ``survival = 1``,
          ``final_phase``, ``final_degree``, and compute the WL signature.
        - Else: record ``survival = 0``; leave phase/degree/wl as ``None``.

        Parameters
        ----------
        run_index : int
            0-based index *r* of this run.
        g_r : pyzx.Graph
            The simplified graph produced by ``run_randomized_pass_order``
            for this run.
        """
        surviving = set(g_r.vertices())
        for v in self._g_original.vertices():
            if v in surviving:
                self._survival_bits[v][run_index] = 1
                self._final_phases[v][run_index] = str(g_r.phase(v))
                self._final_degrees[v][run_index] = len(list(g_r.neighbors(v)))
                self._wl_sigs[v][run_index] = self._compute_wl_signature(
                    g_r,
                    v,
                    circuit_name=self._circuit_name,
                    run_index=run_index,
                )
            # survival bit already initialised to 0; phase/degree/wl to None.

        self._runs_observed += 1

    def aggregate(
        self,
    ) -> tuple[list[EvolutionaryFeature], list[NeighborhoodRecord]]:
        """Build per-spider aggregate features from per-run observations.

        Must only be called after :meth:`observe_run` has been called
        *n_runs* times.

        Returns
        -------
        features : list[EvolutionaryFeature]
            One entry per spider in G_0.
        records : list[NeighborhoodRecord]
            One entry per (spider, run) pair where ``survival = 1``.

        Raises
        ------
        RuntimeError
            If fewer than *n_runs* calls to :meth:`observe_run` have
            been made before this method is invoked.
        """
        if self._runs_observed < self._n_runs:            raise RuntimeError(
                f"EvolutionaryTracker: aggregate() called after only "
                f"{self._runs_observed}/{self._n_runs} runs"
            )

        features: list[EvolutionaryFeature] = []
        records: list[NeighborhoodRecord] = []

        g0 = self._g_original
        R = self._n_runs
        circuit = self._circuit_name

        for v in sorted(g0.vertices()):
            bits = self._survival_bits[v]
            phases = self._final_phases[v]
            degrees = self._final_degrees[v]
            wl_sigs = self._wl_sigs[v]

            survival_rate = sum(bits) / R if R > 0 else 0.0
            bernoulli_ambiguity = 4.0 * survival_rate * (1.0 - survival_rate)

            # Surviving-run values only
            surviving_degrees: list[int] = [
                degrees[r] for r in range(R) if bits[r] == 1 and degrees[r] is not None
            ]
            surviving_phases: list[str] = [
                phases[r] for r in range(R) if bits[r] == 1 and phases[r] is not None
            ]

            # Population variance (ddof=0) helpers
            def _pvar(values: list[float]) -> float:
                n = len(values)
                if n < 2:
                    return 0.0
                mean = sum(values) / n
                return sum((x - mean) ** 2 for x in values) / n

            # Degree variance
            if survival_rate == 0.0 or len(surviving_degrees) < 1:
                final_degree_variance = 0.0
            else:
                final_degree_variance = _pvar([float(d) for d in surviving_degrees])

            # Phase-delta variance
            if survival_rate == 0.0 or len(surviving_phases) < 1:
                phase_delta_variance = 0.0
            else:
                phi0 = float(Fraction(g0.phase(v)))
                two_pi = 2.0 * math.pi
                deltas = [
                    (float(Fraction(phi_r_str)) - phi0) % two_pi
                    for phi_r_str in surviving_phases
                ]
                phase_delta_variance = _pvar(deltas)

            features.append(
                EvolutionaryFeature(
                    circuit_name=circuit,
                    spider_id=v,
                    spider_type=self._spider_type(g0, v),
                    initial_phase=str(g0.phase(v)),
                    survival_vector=",".join(str(b) for b in bits),
                    survival_rate=survival_rate,
                    bernoulli_ambiguity=bernoulli_ambiguity,
                    final_degree_variance=final_degree_variance,
                    phase_delta_variance=phase_delta_variance,
                )
            )

            # NeighborhoodRecord — one per surviving (v, r) pair
            for r in range(R):
                if bits[r] == 1:
                    records.append(
                        NeighborhoodRecord(
                            circuit_name=circuit,
                            spider_id=v,
                            run_index=r,
                            final_phase=phases[r] if phases[r] is not None else "",
                            final_degree=degrees[r] if degrees[r] is not None else 0,
                            wl_signature_k2=wl_sigs[r] if wl_sigs[r] is not None else "ERROR",
                        )
                    )

        return features, records
