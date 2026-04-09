"""Compare embedding models on a test set with known-answer queries."""
import shutil
import time
from pathlib import Path

from cidre.config import CidreConfig, save_config, EMBEDDING_MODELS
from cidre.db import init_db, get_index_stats
from cidre.indexer.scanner import scan_directory
from cidre.indexer.pipeline import IndexingPipeline
from cidre.providers.base import get_provider
from cidre.providers.ollama import OllamaLLM
from cidre.search.engine import SearchEngine

TEST_DIR = Path("/tmp/cidre-test-set")
RESULTS_DIR = Path("/tmp/cidre-model-comparison")

# Queries with expected top result filename
QUERIES = [
    ("quarterly budget planning meeting", "meeting_notes.md"),
    ("pasta cooking recipe italian food", "pasta_recipe.md"),
    ("Goa beach travel vacation", "travel_plans.md"),
    ("invoice payment billing", "invoice.pdf"),
]

MODELS_TO_TEST = [
    ("embeddinggemma", 768),
    ("qwen3-embedding:4b", 4096),
    ("qwen3-embedding:8b", 4096),
]


def run_test(model_name: str, dimensions: int) -> dict:
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name} ({dimensions}d)")
    print(f"{'='*60}")

    # Fresh DB for each model
    db_dir = RESULTS_DIR / model_name.replace(":", "_").replace("/", "_")
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True)
    db_path = db_dir / "cidre.db"

    conn = init_db(db_path, embedding_dimensions=dimensions)
    embedder = get_provider("ollama", model_name)
    llm = OllamaLLM(model="gemma4")
    pipeline = IndexingPipeline(conn=conn, llm=llm, embedder=embedder)

    # Index
    print("Indexing...")
    scanned = scan_directory(TEST_DIR, exclude_patterns=[])
    t0 = time.time()
    count = pipeline.index_batch([(s.path, s.file_type) for s in scanned])
    index_time = time.time() - t0
    print(f"  Indexed {count} files in {index_time:.1f}s")

    stats = get_index_stats(conn)
    print(f"  Stats: {stats}")

    # Search
    engine = SearchEngine(conn=conn, embedder=embedder)
    results_summary = {"model": model_name, "dimensions": dimensions, "index_time": index_time}
    hits = 0
    total = len(QUERIES)

    for query, expected_file in QUERIES:
        t0 = time.time()
        results = engine.search(query, k=5)
        search_time = time.time() - t0

        if results:
            top = Path(results[0].file_path).name
            top_score = results[0].score
            correct = top == expected_file
            if correct:
                hits += 1
            status = "HIT" if correct else "MISS"

            print(f"\n  Query: \"{query}\"")
            print(f"  Expected: {expected_file}")
            print(f"  Got:      {top} (score: {top_score:.4f}) [{status}]")
            print(f"  Time:     {search_time*1000:.0f}ms")

            # Show top 3
            for i, r in enumerate(results[:3], 1):
                name = Path(r.file_path).name
                print(f"    {i}. {name:40s} score={r.score:.4f}  [{', '.join(r.categories[:3])}]")
        else:
            print(f"\n  Query: \"{query}\" — NO RESULTS")

    accuracy = hits / total * 100
    results_summary["accuracy"] = accuracy
    results_summary["hits"] = hits
    results_summary["total"] = total

    print(f"\n  ACCURACY: {hits}/{total} ({accuracy:.0f}%)")

    conn.close()
    return results_summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_name, dims in MODELS_TO_TEST:
        try:
            result = run_test(model_name, dims)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR with {model_name}: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    # Summary
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30s} {'Accuracy':<12s} {'Index Time':<12s} {'Dims':<8s}")
    print("-" * 62)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<30s} ERROR: {r['error']}")
        else:
            print(f"{r['model']:<30s} {r['hits']}/{r['total']} ({r['accuracy']:.0f}%)    {r['index_time']:.1f}s         {r['dimensions']}")


if __name__ == "__main__":
    main()
