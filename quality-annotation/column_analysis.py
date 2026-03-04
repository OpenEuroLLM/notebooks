"""
Column analysis for propella-annotations datasets.

Loads datasets using HF `datasets` library and computes histograms.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset_builder
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Configuration
DATASET_ID = "openeurollm/propella-annotations"
DATA_ROOT = Path(__file__).parent / "data" / "analysis"
FIGURES_ROOT = Path(__file__).parent / "figures"

DATASETS = [
    ("hplt-3", "fin_Latn"),
    ("hplt-3", "deu_Latn"),
    ("nemotron-cc", "high_actual"),
    ("finepdfs", "fin_Latn"),
    ("finepdfs", "deu_Latn"),
    ("fineweb-2", "deu_Latn"),
    ("fineweb-2", "fin_Latn"),
]

# Columns to analyze (categorical columns with histograms)
CATEGORICAL_COLUMNS = [
    "content_quality",
    "information_density",
    "audience_level",
    "commercial_bias",
    "content_safety",
    "content_integrity",
    "content_ratio",
    "content_length",
    "time_sensitivity",
    "educational_value",
    "reasoning_indicators",
    "pii_presence",
]

# List columns (need special handling - count each value in the list)
LIST_COLUMNS = [
    "technical_content",
    "regional_relevance",
    "country_relevance",
    "content_type",
    "business_sector",
]

# Columns to exclude from loading
EXCLUDE_COLUMNS = ["one_sentence_description"]

# Ordered values for each column (best/most positive first for viridis gradient)
COLUMN_VALUE_ORDER = {
    "content_quality": ["excellent", "good", "adequate", "poor", "unacceptable"],
    "information_density": ["dense", "adequate", "moderate", "thin", "empty"],
    "audience_level": ["expert", "advanced", "general", "beginner", "youth", "children"],
    "commercial_bias": ["none", "minimal", "moderate", "heavy", "pure_marketing"],
    "content_safety": ["safe", "mild_concerns", "nsfw", "harmful", "illegal"],
    "content_integrity": ["complete", "mostly_complete", "fragment", "severely_degraded"],
    "content_ratio": ["complete_content", "mostly_content", "mixed_content", "mostly_navigation", "minimal_content"],
    "content_length": ["substantial", "moderate", "brief", "minimal"],
    "time_sensitivity": ["evergreen", "slowly_changing", "regularly_updating", "time_sensitive"],
    "educational_value": ["high", "moderate", "basic", "minimal", "none"],
    "reasoning_indicators": ["analytical", "explanatory", "basic_reasoning", "minimal", "none"],
    "pii_presence": ["no_pii", "contains_pii"],
}


def get_ordered_columns(pivot: pd.DataFrame, column: str) -> list[str]:
    """Get column order for a given column, handling unknown values."""
    if column in COLUMN_VALUE_ORDER:
        order = COLUMN_VALUE_ORDER[column]
        # Include any values not in the predefined order at the end
        existing = [v for v in order if v in pivot.columns]
        extra = [v for v in pivot.columns if v not in order]
        return existing + sorted(extra)
    return list(pivot.columns)


def compute_histograms(config: str, split: str, batch_size: int = 5_000_000) -> dict[str, Counter]:
    """
    Load dataset and compute histograms for all columns.

    Downloads parquet files via huggingface_hub and uses pyarrow value_counts.
    Processes in batches to avoid OOM.
    """
    print(f"  Loading {DATASET_ID} config={config} split={split}...")

    # Get total row count from dataset builder (without downloading)
    builder = load_dataset_builder(DATASET_ID, name=config)
    total_examples = builder.info.splits[split].num_examples
    print(f"  Total examples: {total_examples:,}")

    # Map split name to path (only high_actual -> high-actual)
    split_path = "high-actual" if split == "high_actual" else split
    prefix = f"data/propella-1-4b/{config}/{split_path}/"

    # List parquet files for this config/split
    all_files = list_repo_files(DATASET_ID, repo_type="dataset")
    parquet_files = sorted([f for f in all_files if f.startswith(prefix) and f.endswith(".parquet")])
    print(f"  Found {len(parquet_files)} parquet files")

    # Initialize counters for each column
    categorical_counts = {col: Counter() for col in CATEGORICAL_COLUMNS}
    list_counts = {col: Counter() for col in LIST_COLUMNS}

    # Columns to load (only the ones we need for histograms)
    columns_to_load = CATEGORICAL_COLUMNS + LIST_COLUMNS

    total_rows = 0
    with tqdm(total=total_examples, desc=f"  Processing {config}/{split}") as pbar:
        for parquet_file in parquet_files:
            # Download parquet file (cached locally)
            local_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=parquet_file,
                repo_type="dataset",
            )

            # Read parquet file in batches to avoid OOM
            pf = pq.ParquetFile(local_path)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns_to_load):
                # Use pyarrow value_counts for categorical columns (vectorized, fast)
                for col in CATEGORICAL_COLUMNS:
                    if col in batch.column_names:
                        vc = pc.value_counts(batch.column(col))
                        for item in vc.tolist():
                            if item["values"] is not None:
                                categorical_counts[col][item["values"]] += item["counts"]

                # For list columns: flatten lists, then value_counts
                for col in LIST_COLUMNS:
                    if col in batch.column_names:
                        flattened = pc.list_flatten(batch.column(col))
                        vc = pc.value_counts(flattened)
                        for item in vc.tolist():
                            if item["values"] is not None:
                                list_counts[col][item["values"]] += item["counts"]

                total_rows += len(batch)
                pbar.update(len(batch))

    print(f"  Processed {total_rows:,} rows")

    return {**categorical_counts, **list_counts}


def save_histograms_to_csv(config: str, split: str, histograms: dict[str, Counter]):
    """Save all histograms for a dataset to CSV files."""
    dataset_dir = DATA_ROOT / f"{config}_{split}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_histograms = []

    for column, counts in histograms.items():
        if counts:
            for value, count in counts.items():
                all_histograms.append({
                    "column": column,
                    "value": value,
                    "count": count,
                })

    if all_histograms:
        df = pd.DataFrame(all_histograms)
        df.to_csv(dataset_dir / "histogram.csv", index=False)
        print(f"  Saved histogram.csv with {len(df)} rows")


def load_all_histograms() -> pd.DataFrame:
    """Load all histogram CSVs into a single DataFrame."""
    all_data = []

    for config, split in DATASETS:
        csv_path = DATA_ROOT / f"{config}_{split}" / "histogram.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["dataset"] = config
            df["split"] = split
            df["dataset_label"] = f"{config}/{split.split('_')[0]}"
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def plot_column_analysis(df: pd.DataFrame, column: str, output_dir: Path):
    """Generate stackplot (proportion) and barplot (counts) for a column."""
    output_dir.mkdir(parents=True, exist_ok=True)

    col_data = df[df["column"] == column].copy()
    if col_data.empty:
        print(f"  No data for column: {column}")
        return

    # Pivot for plotting
    pivot = col_data.pivot_table(
        index="dataset_label",
        columns="value",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    # Ensure consistent order of datasets
    dataset_order = [f"{c}/{s.split('_')[0]}" for c, s in DATASETS]
    pivot = pivot.reindex([d for d in dataset_order if d in pivot.index])

    # Order columns by predefined order
    ordered_cols = get_ordered_columns(pivot, column)
    pivot = pivot[ordered_cols]

    # Get viridis colors
    n_colors = len(pivot.columns)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_colors))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Proportion stackplot
    ax1 = axes[0]
    proportions = pivot.div(pivot.sum(axis=1), axis=0) * 100
    proportions.plot(kind="bar", stacked=True, ax=ax1, width=0.8, color=colors)
    ax1.set_ylabel("Proportion (%)")
    ax1.set_xlabel("Dataset")
    ax1.set_title(f"{column} - Proportion by Dataset")
    ax1.legend(title=column, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_ylim(0, 100)

    # Right: Count barplot
    ax2 = axes[1]
    pivot_millions = pivot / 1_000_000
    pivot_millions.plot(kind="bar", ax=ax2, width=0.8, color=colors)
    ax2.set_ylabel("Number of Documents (Millions)")
    ax2.set_xlabel("Dataset")
    ax2.set_title(f"{column} - Document Count by Dataset")
    ax2.legend(title=column, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_analysis.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {column}_analysis.pdf")


def plot_all_columns_grid(df: pd.DataFrame, columns: list[str], output_dir: Path):
    """Generate a grid plot with all columns (2 columns per row)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 2
    n_rows = len(columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))

    dataset_order = [f"{c}/{s.split('_')[0]}" for c, s in DATASETS]

    for i, column in enumerate(columns):
        col_data = df[df["column"] == column].copy()
        if col_data.empty:
            continue

        pivot = col_data.pivot_table(
            index="dataset_label",
            columns="value",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        pivot = pivot.reindex([d for d in dataset_order if d in pivot.index])

        # Order columns by predefined order
        ordered_cols = get_ordered_columns(pivot, column)
        pivot = pivot[ordered_cols]

        # Get viridis colors
        n_colors = len(pivot.columns)
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_colors))

        # Left: Proportion
        ax1 = axes[i, 0]
        proportions = pivot.div(pivot.sum(axis=1), axis=0) * 100
        proportions.plot(kind="bar", stacked=True, ax=ax1, width=0.8, legend=False, color=colors)
        ax1.set_ylabel("Proportion (%)")
        ax1.set_xlabel("")
        ax1.set_title(f"{column} - Proportion")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_ylim(0, 100)

        # Right: Count with legend
        ax2 = axes[i, 1]
        pivot_millions = pivot / 1_000_000
        pivot_millions.plot(kind="bar", ax=ax2, width=0.8, color=colors)
        ax2.set_ylabel("Documents (M)")
        ax2.set_xlabel("")
        ax2.set_title(f"{column} - Count")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.legend(title=column, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output_dir / "all_columns_analysis.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved all_columns_analysis.pdf")


def report_quality_counts(df: pd.DataFrame):
    """Print a pivot table of excellent/good content_quality counts for Finnish and German."""
    LANG_MAP = {"fin_Latn": "Finnish", "deu_Latn": "German", "high_actual": "English"}
    cq = df[(df["column"] == "content_quality") & (df["split"].isin(LANG_MAP))].copy()
    cq["language"] = cq["split"].map(LANG_MAP)

    pivot = cq.pivot_table(
        index=["language", "dataset"],
        columns="value",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    pivot = pivot[["excellent", "good"]].copy()
    pivot["excellent + good"] = pivot["excellent"] + pivot["good"]
    pivot = pivot / 1_000_000
    print(pivot.to_string())


def main():
    print("=" * 60)
    print("Column Analysis for propella-annotations")
    print("=" * 60)

    # Step 1: Load datasets and compute histograms
    print("\nLoading datasets and computing histograms...")
    for config, split in DATASETS:
        print(f"\n{config}/{split}:")
        csv_path = DATA_ROOT / f"{config}_{split}" / "histogram.csv"

        # # Skip if already computed
        if csv_path.exists():
            print(f"  Histogram already exists, skipping...")
            continue

        try:
            histograms = compute_histograms(config, split)
            save_histograms_to_csv(config, split, histograms)
        except Exception as e:
            print(f"  Error: {e}")

    # Step 2: Load all histograms
    print("\n" + "=" * 60)
    print("Loading histogram data...")
    df = load_all_histograms()
    print(f"Total rows: {len(df)}")

    # Step 2.5: Report quality counts for Finnish and German
    print("\n" + "=" * 60)
    print("Content quality report (Finnish & German):")
    report_quality_counts(df)

    # Step 3: Generate individual plots
    print("\n" + "=" * 60)
    print("Generating individual column plots...")
    for column in CATEGORICAL_COLUMNS + LIST_COLUMNS:
        plot_column_analysis(df, column, FIGURES_ROOT)

    # Step 4: Generate grid plot
    print("\n" + "=" * 60)
    print("Generating grid plot...")
    key_columns = [
        "content_quality",
        "information_density",
        "audience_level",
        "commercial_bias",
        "content_safety",
        "educational_value",
        "reasoning_indicators",
        "pii_presence",
    ]
    plot_all_columns_grid(df, key_columns, FIGURES_ROOT)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
