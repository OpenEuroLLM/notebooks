import argparse
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def is_categorical_string(series: pd.Series, max_unique_ratio: float = 0.05, max_unique_count: int = 50) -> bool:
    """Heuristic to detect if a string column is categorical."""
    if series.dtype != "object":
        return False

    # Check if values are strings (not lists/arrays)
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    if any(isinstance(v, (list, dict, np.ndarray)) for v in sample):
        return False

    n_unique = series.nunique()
    n_total = len(series)

    # Categorical if few unique values relative to total, or absolute count is small
    return (n_unique / n_total < max_unique_ratio) or (n_unique <= max_unique_count)


def is_list_column(series: pd.Series) -> bool:
    """Check if a column contains lists or arrays."""
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    return all(isinstance(v, (list, np.ndarray)) for v in sample)


def print_histogram(counts: Counter, title: str, max_bars: int = 30):
    """Print a text-based histogram."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    if not counts:
        print("  (no data)")
        return

    # Sort by count descending
    sorted_items = counts.most_common(max_bars)
    total = sum(counts.values())

    if len(counts) > max_bars:
        print(f"  (showing top {max_bars} of {len(counts)} unique values)")

    max_count = sorted_items[0][1] if sorted_items else 1
    bar_width = 40

    for value, count in sorted_items:
        pct = 100 * count / total
        bar_len = int(bar_width * count / max_count)
        bar = "█" * bar_len
        print(f"  {str(value)[:30]:<30} | {bar:<{bar_width}} | {count:>7} ({pct:5.1f}%)")

    print(f"\n  Total: {total}")


COLUMNS = [
    "id",
    "content_integrity",
    "content_ratio",
    "content_length",
    # "one_sentence_description",  # excluded - fills up memory
    "content_type",
    "business_sector",
    "technical_content",
    "information_density",
    "content_quality",
    "audience_level",
    "commercial_bias",
    "time_sensitivity",
    "content_safety",
    "educational_value",
    "reasoning_indicators",
    "pii_presence",
    "regional_relevance",
    "country_relevance",
]


def analyze_parquet(file_path: str, max_rows: int | None = None, batch_size: int = 1_000_000):
    """Analyze a parquet file and print histograms for categorical and list columns."""
    path = Path(file_path).expanduser()
    print(f"Loading: {path}")

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    rows_to_process = min(max_rows, total_rows) if max_rows else total_rows
    print(f"Total rows in file: {total_rows:,}")
    print(f"Rows to process: {rows_to_process:,}")
    print(f"Columns: {COLUMNS}")

    # First pass: detect column types from a small sample
    print("\nDetecting column types from sample...")
    sample_batch = next(pf.iter_batches(columns=COLUMNS, batch_size=1000)).to_pandas()

    categorical_cols = []
    list_cols = []

    print("\n" + "=" * 60)
    print("COLUMN TYPE DETECTION")
    print("=" * 60)

    for col in sample_batch.columns:
        if is_list_column(sample_batch[col]):
            list_cols.append(col)
            print(f"  {col}: LIST")
        elif is_categorical_string(sample_batch[col]):
            categorical_cols.append(col)
            print(f"  {col}: CATEGORICAL STRING")
        else:
            dtype = sample_batch[col].dtype
            print(f"  {col}: {dtype}")

    # Initialize counters
    categorical_counts = {col: Counter() for col in categorical_cols}
    list_value_counts = {col: Counter() for col in list_cols}
    list_length_counts = {col: Counter() for col in list_cols}

    # Process in batches
    print(f"\nProcessing data in batches of {batch_size:,}...")
    rows_read = 0

    with tqdm(total=rows_to_process, unit="rows") as pbar:
        for batch in pf.iter_batches(columns=COLUMNS, batch_size=batch_size):
            df = batch.to_pandas()

            # Respect max_rows limit
            if max_rows and rows_read + len(df) > max_rows:
                df = df.head(max_rows - rows_read)

            # Update categorical counters
            for col in categorical_cols:
                categorical_counts[col].update(df[col].dropna())

            # Update list counters
            for col in list_cols:
                for lst in df[col].dropna():
                    if isinstance(lst, (list, np.ndarray)):
                        list_value_counts[col].update(lst)
                        list_length_counts[col][len(lst)] += 1

            rows_read += len(df)
            pbar.update(len(df))

            if max_rows and rows_read >= max_rows:
                break

    print(f"\nProcessed {rows_read:,} rows")

    # Print histograms for categorical string columns
    for col in categorical_cols:
        print_histogram(categorical_counts[col], f"Column: {col} (categorical)")

    # Print histograms for list columns
    for col in list_cols:
        print(f"\n{'=' * 60}")
        print(f"Column: {col} (list)")
        print(f"{'=' * 60}")
        print_histogram(list_value_counts[col], f"  Value distribution in '{col}'")
        print_histogram(list_length_counts[col], f"  List length distribution in '{col}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parquet file distributions")
    parser.add_argument("--num_samples", type=int, default=100_000, help="Number of rows to load")
    args = parser.parse_args()

    # https://huggingface.co/datasets/openeurollm/propella-annotations/viewer/hplt-3/fin_Latn?p=1&row=100
    # hplt-3/fin_Latn.parquet
    analyze_parquet("data/hplt-3/fin_Latn.parquet", max_rows=args.num_samples)
