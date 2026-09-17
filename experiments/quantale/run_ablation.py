from __future__ import annotations

import csv
from pathlib import Path

import pyzx as zx

from zx_cenf.quantale.instrumented_reduction import run_instrumented_reduction
from zx_cenf.quantale.pattern_table import build_pattern_rule_table
from zx_cenf.quantale.null_table import build_null_table
from zx_cenf.quantale.evaluation import evaluate_ablation

DATA_DIR = Path("data/toy")
OUT_DIR = Path("data/track3_results")
DIAGRAM_NAMES = ["toy_larger_clifford_heavy", "toy_medium_diverse", "toy_small_dense"]
N_TRAINING_SWEEPS_PER_DIAGRAM = 5
MIN_OCCURRENCES = 5
NULL_TABLE_SEED = 42  


def train_pattern_table(train_names: list[str]) -> tuple[dict, dict]:
    """Identical procedure to run_phase_f.py's train_pattern_table --
    not changed for the ablation, per instructions."""
    all_log_rows = []
    for name in train_names:
        circuit = zx.Circuit.load(str(DATA_DIR / f"{name}.qasm"))
        for sweep_seed in range(N_TRAINING_SWEEPS_PER_DIAGRAM):
            g = circuit.to_graph()
            log_rows, _reached = run_instrumented_reduction(g, seed=sweep_seed)
            all_log_rows.extend(log_rows)
    table, diagnostics = build_pattern_rule_table(all_log_rows, min_occurrences=MIN_OCCURRENCES)
    return table, diagnostics


def _fmt(mu) -> str:
    return "None" if mu is None else "(" + ", ".join(f"{x:g}" for x in mu) + ")"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    hit_rows = []

    for held_out in DIAGRAM_NAMES:
        train_names = [n for n in DIAGRAM_NAMES if n != held_out]
        print(f"\n{'='*72}")
        print(f"HELD OUT: {held_out}   (trained on: {train_names})")
        print(f"{'='*72}")

        learned_table, diagnostics = train_pattern_table(train_names)
        null_table = build_null_table(learned_table, diagnostics, seed=NULL_TABLE_SEED)
        print(f"  learned table: {len(learned_table)} entries | "
              f"null table: {len(null_table)} entries (same keys: "
              f"{set(learned_table.keys()) == set(null_table.keys())})")

        circuit = zx.Circuit.load(str(DATA_DIR / f"{held_out}.qasm"))
        g = circuit.to_graph()
        result = evaluate_ablation(g, held_out, learned_table, null_table, null_seed=NULL_TABLE_SEED)

        print(f"\n  oracle:       {_fmt(result.oracle_mu)}")
        print(f"  greedy:       {_fmt(result.greedy_mu)}  [{result.greedy_stop}]")
        print(f"  learned:      {_fmt(result.learned_mu)}  [{result.learned_stop}]  "
              f"hits={result.learned_hits} differs_from_greedy={result.learned_n_differs_from_greedy}")
        print(f"  null:         {_fmt(result.null_mu)}  [{result.null_stop}]  "
              f"hits={result.null_hits} differs_from_greedy={result.null_n_differs_from_greedy}")
        print(f"  full_reduce:  {_fmt(result.full_reduce_mu)}")

        # --- interpretation logic, per the ablation spec ---
        min_hits_for_comparison = 3
        if result.learned_hits < min_hits_for_comparison and result.null_hits < min_hits_for_comparison:
            verdict = "INCONCLUSIVE -- too few table hits for either table to compare meaningfully"
        elif result.learned_mu is None or result.null_mu is None:
            verdict = "INCONCLUSIVE -- one or both strategies failed to finalize to an extractable state"
        elif result.learned_mu == result.null_mu:
            verdict = "learned == null: improvement (if any) is NOT attributable to the specific learned preferences"
        else:
            from zx_cenf.quantale.evaluation import _mu_lexicographically_better
            cmp = _mu_lexicographically_better(result.learned_mu, result.null_mu)
            if cmp > 0:
                verdict = "learned > null: some evidence the specific learned preferences carry value"
            elif cmp < 0:
                verdict = "learned < null: possible NEGATIVE transfer from the learned preferences"
            else:
                verdict = "learned ~= null on priority coordinates: no clear causal signal"
        print(f"  VERDICT: {verdict}")

        summary_rows.append({
            "diagram_id": held_out,
            "oracle_mu": _fmt(result.oracle_mu),
            "greedy_mu": _fmt(result.greedy_mu),
            "greedy_stop": result.greedy_stop,
            "learned_mu": _fmt(result.learned_mu),
            "learned_stop": result.learned_stop,
            "learned_hits": result.learned_hits,
            "learned_misses": result.learned_misses,
            "learned_n_differs_from_greedy": result.learned_n_differs_from_greedy,
            "null_mu": _fmt(result.null_mu),
            "null_stop": result.null_stop,
            "null_hits": result.null_hits,
            "null_misses": result.null_misses,
            "null_n_differs_from_greedy": result.null_n_differs_from_greedy,
            "full_reduce_mu": _fmt(result.full_reduce_mu),
            "null_seed": result.null_seed,
            "verdict": verdict,
        })

        for table_type, trace in (("learned", result.learned_trace), ("null", result.null_trace)):
            for rec in trace:
                hit_rows.append({
                    "diagram_id": held_out,
                    "table_type": table_type,
                    "step": rec.step,
                    "pattern_hash": rec.pattern_hash,
                    "preferred_rule": rec.preferred_rule,
                    "competing_rules": "|".join(rec.competing_rules),
                    "rule_selected": rec.rule_selected,
                    "surrogate_before": "|".join(str(x) for x in rec.surrogate_before),
                    "surrogate_after": "|".join(str(x) for x in rec.surrogate_after) if rec.surrogate_after else "",
                    "differs_from_greedy_choice": rec.differs_from_greedy_choice,
                })

    summary_path = OUT_DIR / "ablation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    hits_path = OUT_DIR / "ablation_hits.csv"
    if hit_rows:
        with hits_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hit_rows[0].keys())
            writer.writeheader()
            writer.writerows(hit_rows)
        print(f"Wrote {hits_path} ({len(hit_rows)} hit records)")
    else:
        print(f"No table hits recorded (learned or null) -- {hits_path} not written")


if __name__ == "__main__":
    main()