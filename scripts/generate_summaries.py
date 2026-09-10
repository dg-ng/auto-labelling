"""Generate summaries for rows in data/master_data.csv that are missing them.

Usage:
    uv run python scripts/generate_summaries.py [--extractive] [--sentences N]
    uv run python scripts/generate_summaries.py [--chunk-size N] [--batch-size N] [--model NAME]
    uv run python scripts/generate_summaries.py [--merge-only]

Modes
-----
--extractive      Fast, zero-RAM mode: extract the first N sentences of each
                  article as its summary. No model loaded. News articles are
                  written in inverted-pyramid style so the lead sentences
                  capture the key facts. Use this when neural models OOM.
--sentences N     Number of sentences to extract in --extractive mode (default: 3).

Neural mode flags (default when --extractive is not set):
    --model NAME     Summarization model HF hub name (default: config.SUMMARIZATION_MODEL_NAME).
    --chunk-size N   Rows processed per cache-save cycle (default: 50).
    --batch-size N   Rows fed to the model per forward pass (default: 4).

--merge-only     Skip generation, just merge the cache into a final CSV.

How it works
------------
1. Reads data/master_data.csv; identifies rows where 'summary' is null or empty.
2. Generates or extracts a summary per row, cached in data/summaries_cache.json
   keyed by exact text so resuming never re-does completed rows.
3. Merges the cache back into the full dataframe (existing 1,320 summaries kept
   as-is; newly generated summaries filled in).
4. Saves data/master_data_summarized.csv.
"""

import argparse
import sys
from pathlib import Path

# ── repo root on path ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from utils.summarization import summarize_texts_cached

MASTER_DATA_PATH = REPO_ROOT / "data" / "master_data.csv"
CACHE_PATH       = REPO_ROOT / "data" / "summaries_cache.json"
OUTPUT_PATH      = REPO_ROOT / "data" / "master_data_summarized.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extractive", action="store_true",
                   help="Use extractive (first-N-sentences) mode — no model, zero extra RAM")
    p.add_argument("--sentences", type=int, default=3,
                   help="Sentences to extract in --extractive mode (default: 3)")
    p.add_argument("--chunk-size", type=int, default=50,
                   help="Rows per cache-save cycle (default: 50)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Rows per model forward pass (default: 4)")
    p.add_argument("--model", type=str, default=None,
                   help="Override the summarization model (default: config.SUMMARIZATION_MODEL_NAME)")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip generation, just merge cache → final CSV")
    return p.parse_args()


def extractive_summarize(texts, n_sentences=3) -> list:
    """Extract the first `n_sentences` sentences from each text.

    Splits on '. ' / '? ' / '! ' boundaries (simple but effective for
    newswire-style articles). Falls back to the full text truncated to 500
    chars if the text is too short to yield any sentence boundary.
    """
    import re
    results = []
    sentence_end = re.compile(r'(?<=[.?!])\s+')
    for text in texts:
        text = text.strip()
        sentences = sentence_end.split(text)
        # Keep the first n_sentences; if fewer exist, keep all
        summary = " ".join(sentences[:n_sentences]).strip()
        if not summary:
            summary = text[:500]
        results.append(summary)
    return results


def extractive_summarize_cached(texts, cache_path, n_sentences=3, chunk_size=500) -> list:
    """Extractive summarization with the same durable JSON cache contract as
    summarize_texts_cached — safe to interrupt and resume, never re-processes
    already-cached texts."""
    import json

    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    texts = list(texts)
    missing = [t for t in texts if t not in cache]
    print(f"{len(texts) - len(missing)} reused from cache, {len(missing)} to generate (extractive)")

    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        summaries = extractive_summarize(chunk, n_sentences=n_sentences)
        for t, s in zip(chunk, summaries):
            cache[t] = s
        done = min(i + chunk_size, len(missing))
        print(f"  [{done}/{len(missing)}] chunk saved", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return [cache[t] for t in texts]


def load_data():
    df = pd.read_csv(MASTER_DATA_PATH)
    # Normalise: empty string → NaN so a single check covers both
    df["summary"] = df["summary"].replace("", None)
    return df


def rows_needing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of rows whose summary is null or blank."""
    missing_mask = df["summary"].isna()
    return df[missing_mask].copy()


def merge_cache_into_df(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Fill df['summary'] for any row whose text is in cache."""
    # Build a text→summary lookup from the cache (keyed by exact text value)
    result = df.copy()
    needs_fill = result["summary"].isna()
    result.loc[needs_fill, "summary"] = result.loc[needs_fill, "text"].map(cache)
    return result


def report_progress(df: pd.DataFrame):
    filled = df["summary"].notna().sum()
    total  = len(df)
    print(f"\nProgress: {filled}/{total} rows have summaries ({filled/total:.1%})")
    per_cat = df.groupby("category")["summary"].apply(lambda s: s.notna().sum())
    cat_total = df.groupby("category").size()
    summary_pct = (per_cat / cat_total * 100).round(1)
    print(summary_pct.to_string())
    print()


def main():
    args = parse_args()

    # ── load ────────────────────────────────────────────────────────────────
    print(f"Reading {MASTER_DATA_PATH} …")
    df = load_data()
    print(f"  {len(df)} total rows, "
          f"{df['summary'].notna().sum()} already have summaries, "
          f"{df['summary'].isna().sum()} need generating.")

    missing_df = rows_needing_summary(df)

    if missing_df.empty:
        print("All rows already have summaries — jumping straight to merge.")
    elif args.merge_only:
        print(f"--merge-only: skipping generation of {len(missing_df)} rows.")
    elif args.extractive:
        print(f"\nExtractive summarization for {len(missing_df)} rows …")
        print(f"  sentences={args.sentences}  chunk_size={args.chunk_size}")
        print(f"  Cache: {CACHE_PATH}")
        print("  No model loaded — runs entirely in RAM-free mode.\n")

        texts_to_generate = missing_df["text"].tolist()
        _ = extractive_summarize_cached(
            texts_to_generate,
            cache_path=CACHE_PATH,
            n_sentences=args.sentences,
            chunk_size=args.chunk_size,
        )
        print(f"\nExtractive summarization done.")
    else:
        from utils.config import SUMMARIZATION_MODEL_NAME
        model_name = args.model if args.model else SUMMARIZATION_MODEL_NAME
        print(f"\nGenerating summaries for {len(missing_df)} rows …")
        print(f"  model={model_name}")
        print(f"  chunk_size={args.chunk_size}  batch_size={args.batch_size}")
        print(f"  Cache: {CACHE_PATH}")
        print("  Progress is saved after every chunk — safe to Ctrl-C and resume.\n")

        texts_to_generate = missing_df["text"].tolist()
        _ = summarize_texts_cached(
            texts_to_generate,
            cache_path=CACHE_PATH,
            model_name=model_name,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        print(f"\nGeneration done (or cache already covered all rows).")

    # ── merge cache → full dataframe ────────────────────────────────────────
    import json
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    print(f"Cache contains {len(cache)} entries.")

    merged = merge_cache_into_df(df, cache)

    report_progress(merged)

    # ── save ────────────────────────────────────────────────────────────────
    still_missing = merged["summary"].isna().sum()
    if still_missing > 0:
        print(f"WARNING: {still_missing} rows still have no summary — run again without "
              f"--merge-only to continue generation.")
    else:
        print("OK: All rows have summaries.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"  Columns: {merged.columns.tolist()}")
    print(f"  Rows:    {len(merged)}")


if __name__ == "__main__":
    main()
