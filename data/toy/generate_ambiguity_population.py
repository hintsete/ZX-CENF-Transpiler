"""Generates a small population of QASM circuits sized/dense enough to
exercise PyZX's non-confluent passes (pivot, lcomp, pivot_gadget, gadget) —
Track 1's confluent-rule control set won't show anything interesting on
these, by design, so this population is aimed at the PyZX side of the test.
"""

from pathlib import Path

import pyzx as zx

OUTPUT_DIR = Path(__file__).parent

CONFIGS = [
    dict(qubits=3, depth=25, p_t=0.25, p_s=0.1, p_hsh=0.2, p_cnot=0.3, seed=1, name="toy_small_dense"),
    dict(qubits=4, depth=40, p_t=0.25, p_s=0.1, p_hsh=0.2, p_cnot=0.3, seed=2, name="toy_medium_diverse"),
    dict(qubits=5, depth=60, p_t=0.2, p_s=0.15, p_hsh=0.15, p_cnot=0.35, seed=3, name="toy_larger_clifford_heavy"),
]

if __name__ == "__main__":
    for cfg in CONFIGS:
        g = zx.generate.cliffordT(
            cfg["qubits"], cfg["depth"],
            p_t=cfg["p_t"], p_s=cfg["p_s"], p_hsh=cfg["p_hsh"], p_cnot=cfg["p_cnot"],
            seed=cfg["seed"],
        )
        circ = zx.Circuit.from_graph(g)
        path = OUTPUT_DIR / f"{cfg['name']}.qasm"
        path.write_text(circ.to_qasm())
        print(f"{path} — {len(circ.gates)} gates")