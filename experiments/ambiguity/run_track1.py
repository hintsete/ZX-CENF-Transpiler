from pathlib import Path

from zx_cenf.ambiguity.runner import run_track1

DATA_DIR = Path("data/toy")
OUTPUT_CSV = Path("data/track1_results/ambiguity_scores.csv")
SPIDER_TAGS_CSV = Path("data/track1_results/spider_ambiguity_tags.csv")

if __name__ == "__main__":
    qasm_paths = sorted(DATA_DIR.glob("*.qasm"))
    run_track1(qasm_paths, OUTPUT_CSV, SPIDER_TAGS_CSV, n_orderings=30)
    print(f"Wrote diagram-level results for {len(qasm_paths)} diagrams to {OUTPUT_CSV}")
    print(f"Wrote spider-level ambiguity tags to {SPIDER_TAGS_CSV}")