from pathlib import Path

from zx_cenf.ambiguity.runner import run_track1

# Paths anchored to the project root (two levels up from this file),
# so the script works regardless of which directory you run it from.
_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _ROOT / "data" / "toy"
OUTPUT_CSV = _ROOT / "data" / "track1_results" / "ambiguity_scores.csv"
SPIDER_TAGS_CSV = _ROOT / "data" / "track1_results" / "spider_ambiguity_tags.csv"
EVO_FEATURES_CSV = _ROOT / "data" / "track1_results" / "spider_evolutionary_features.csv"
RUN_NBHD_CSV = _ROOT / "data" / "track1_results" / "spider_run_neighborhoods.csv"

if __name__ == "__main__":
    qasm_paths = sorted(DATA_DIR.glob("*.qasm"))
    n = len(qasm_paths)
    run_track1(
        qasm_paths,
        OUTPUT_CSV,
        SPIDER_TAGS_CSV,
        n_orderings=30,
        evolutionary_features_csv=EVO_FEATURES_CSV,
        run_neighborhoods_csv=RUN_NBHD_CSV,
    )
    print(f"Wrote diagram-level results for {n} diagrams to {OUTPUT_CSV}")
    print(f"Wrote spider-level ambiguity tags to {SPIDER_TAGS_CSV}")
    print(f"Wrote {n} circuits to {EVO_FEATURES_CSV}")
    print(f"Wrote {n} circuits to {RUN_NBHD_CSV}")
