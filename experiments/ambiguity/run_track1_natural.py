"""Run Track 1 on the reproducible 60-circuit non-planted population."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from zx_cenf.ambiguity.runner import run_track1
from zx_cenf.quantale.natural_population import (
    build_natural_circuit,
    make_population_specs,
)


ROOT = Path(__file__).resolve().parents[2]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-seed", type=int, default=17_000)
    parser.add_argument("--ordering-seed", type=int, default=0)
    parser.add_argument("--orderings", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "track1_results" / "natural",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = make_population_specs(base_seed=args.population_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zx-cenf-track1-") as temp_dir:
        qasm_paths = []
        for spec in specs:
            path = Path(temp_dir) / f"{spec.circuit_id}.qasm"
            path.write_text(build_natural_circuit(spec).to_qasm())
            qasm_paths.append(path)

        run_track1(
            sorted(qasm_paths),
            args.out_dir / "ambiguity_scores.csv",
            args.out_dir / "spider_ambiguity_tags.csv",
            n_orderings=args.orderings,
            evolutionary_features_csv=args.out_dir / "spider_evolutionary_features.csv",
            run_neighborhoods_csv=args.out_dir / "spider_run_neighborhoods.csv",
            pareto_summary_csv=args.out_dir / "pareto_summary.csv",
            run_results_csv=args.out_dir / "ordering_results.csv",
            base_seed=args.ordering_seed,
        )

    print(
        f"Wrote Track 1 results for {len(specs)} circuits x "
        f"{args.orderings} orderings to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
