"""
Check statistics of a Hugging Face dataset using the Datasets Server API.
No download required - uses pre-computed statistics but stats are computed on top 50M rows.
"""
from pathlib import Path

import requests
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from fsspec.implementations.http import HTTPFileSystem

FIGURE_PATH = Path(__file__).parent / "figures"
FIGURE_PATH.mkdir(exist_ok=True)
DATASET_ID = "openeurollm/propella-annotations"

DATASETS = [
    ("hplt-3", "fin_Latn"),
    ("hplt-3", "deu_Latn"),
    ("nemotron-cc", "high_actual"),
    ("finepdfs", "fin_Latn"),
    ("finepdfs", "deu_Latn"),
    ("fineweb-2", "deu_Latn"),
    ("fineweb-2", "fin_Latn"),
]

QUALITY_LEVELS = ["excellent", "good", "adequate", "poor", "unacceptable"]


def get_actual_row_count(dataset: str, config: str, split: str) -> int:
    """Get actual row count from parquet metadata (not capped like stats API)."""
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}"
    r = requests.get(url)
    r.raise_for_status()
    parquet_urls = r.json().get(split, [])

    fs = HTTPFileSystem()
    total_rows = 0
    for purl in parquet_urls:
        with fs.open(purl) as f:
            pf = pq.ParquetFile(f)
            total_rows += pf.metadata.num_rows
    return total_rows


def get_dataset_statistics(dataset: str, config: str, split: str) -> dict:
    """Fetch pre-computed statistics from HF Datasets Server API."""
    url = "https://datasets-server.huggingface.co/statistics"
    params = {"dataset": dataset, "config": config, "split": split}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_content_quality_stats(dataset: str, config: str, split: str) -> dict | None:
    """Get content_quality frequency distribution for a dataset."""
    try:
        stats = get_dataset_statistics(dataset, config, split)
        for col_stats in stats.get("statistics", []):
            if col_stats["column_name"] == "content_quality":
                frequencies = col_stats.get("column_statistics", {}).get("frequencies", {})
                total = sum(frequencies.values())
                return {"frequencies": frequencies, "total": total}
        return None
    except Exception as e:
        print(f"  Error fetching {config}/{split}: {e}")
        return None


def format_count(count: int, total: int) -> str:
    """Format count as percentage and millions."""
    pct = 100 * count / total if total > 0 else 0
    if count >= 1_000_000:
        return f"{pct:5.1f}% ({count / 1_000_000:.1f}M)"
    elif count >= 1_000:
        return f"{pct:5.1f}% ({count / 1_000:.0f}K)"
    else:
        return f"{pct:5.1f}% ({count})"


def extract_language(split: str) -> str:
    """Extract language code from split name."""
    if "_" in split:
        return split.split("_")[0]
    return split


def main():
    print(f"Fetching content_quality statistics from {DATASET_ID}...")
    print("=" * 120)

    results = []

    for config, split in DATASETS:
        print(f"  Fetching {config}/{split}...")
        stats = get_content_quality_stats(DATASET_ID, config, split)
        if stats:
            language = extract_language(split)
            # Get actual row count from parquet metadata
            actual_rows = get_actual_row_count(DATASET_ID, config, split)
            sampled_rows = stats["total"]
            is_sampled = actual_rows > sampled_rows * 1.01  # Allow 1% tolerance

            results.append({
                "language": language,
                "dataset": config,
                "split": split,
                "stats": stats,
                "actual_rows": actual_rows,
                "is_sampled": is_sampled,
            })

    # Print table
    print("\n")
    print("=" * 120)
    print("CONTENT QUALITY DISTRIBUTION")
    print("=" * 120)

    # Header
    col_widths = {
        "language": 10,
        "dataset": 15,
        "rows": 22,
        "excellent": 18,
        "good": 18,
        "adequate": 18,
        "poor": 18,
        "unacceptable": 18,
    }

    header = (
        f"{'Language':<{col_widths['language']}} | "
        f"{'Dataset':<{col_widths['dataset']}} | "
        f"{'Rows':>{col_widths['rows']}} | "
        f"{'Excellent':>{col_widths['excellent']}} | "
        f"{'Good':>{col_widths['good']}} | "
        f"{'Adequate':>{col_widths['adequate']}} | "
        f"{'Poor':>{col_widths['poor']}} | "
        f"{'Unacceptable':>{col_widths['unacceptable']}}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for r in results:
        freq = r["stats"]["frequencies"]
        total = r["stats"]["total"]
        actual = r["actual_rows"]

        # Format row count with sampling indicator
        if r["is_sampled"]:
            coverage_pct = 100 * total / actual
            rows_str = f"{actual/1e6:.0f}M ({coverage_pct:.0f}% sampled)"
        else:
            rows_str = f"{actual/1e6:.1f}M"

        row = (
            f"{r['language']:<{col_widths['language']}} | "
            f"{r['dataset']:<{col_widths['dataset']}} | "
            f"{rows_str:>{col_widths['rows']}} | "
            f"{format_count(freq.get('excellent', 0), total):>{col_widths['excellent']}} | "
            f"{format_count(freq.get('good', 0), total):>{col_widths['good']}} | "
            f"{format_count(freq.get('adequate', 0), total):>{col_widths['adequate']}} | "
            f"{format_count(freq.get('poor', 0), total):>{col_widths['poor']}} | "
            f"{format_count(freq.get('unacceptable', 0), total):>{col_widths['unacceptable']}}"
        )
        print(row)

    print("-" * len(header))
    print(f"\nTotal datasets: {len(results)}")
    print("Note: HF stats API caps at 50M rows. Percentages for sampled datasets are based on first 50M rows only.")

    # Generate plots
    plot_quality_distribution(results)
    plot_document_counts(results)


def plot_quality_distribution(results: list):
    """Generate a grouped bar chart showing quality distribution per dataset."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [f"{r['dataset']}/{r['language']}" for r in results]
    x = np.arange(len(labels))
    width = 0.15

    colors = {
        "excellent": "#2ecc71",
        "good": "#3498db",
        "adequate": "#f39c12",
        "poor": "#e74c3c",
        "unacceptable": "#8e44ad",
    }

    for i, quality in enumerate(QUALITY_LEVELS):
        percentages = []
        for r in results:
            freq = r["stats"]["frequencies"]
            total = r["stats"]["total"]
            pct = 100 * freq.get(quality, 0) / total if total > 0 else 0
            percentages.append(pct)

        offset = (i - len(QUALITY_LEVELS) / 2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width, label=quality.capitalize(), color=colors[quality])

    ax.set_xlabel("Dataset / Language")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Content Quality Distribution by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fn = FIGURE_PATH / "quality_distribution.pdf"
    plt.savefig(fn)
    print(f'Saved: {fn}')
    plt.close()


def plot_document_counts(results: list):
    """Generate a stacked bar chart showing document counts per quality bucket."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [f"{r['dataset']}/{r['language']}" for r in results]
    x = np.arange(len(labels))

    colors = {
        "excellent": "#2ecc71",
        "good": "#3498db",
        "adequate": "#f39c12",
        "poor": "#e74c3c",
        "unacceptable": "#8e44ad",
    }

    bottom = np.zeros(len(results))

    for quality in QUALITY_LEVELS:
        counts = []
        for r in results:
            freq = r["stats"]["frequencies"]
            count = freq.get(quality, 0) / 1_000_000  # Convert to millions
            counts.append(count)

        counts = np.array(counts)
        ax.bar(x, counts, bottom=bottom, label=quality.capitalize(), color=colors[quality])
        bottom += counts

    ax.set_xlabel("Dataset / Language")
    ax.set_ylabel("Number of Documents (Millions)")
    ax.set_title("Document Counts by Quality Level (sampled datasets show first 50M only)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add total count labels on top of bars (show actual total if sampled)
    for i, r in enumerate(results):
        sampled_total = r["stats"]["total"] / 1_000_000
        actual_total = r["actual_rows"] / 1_000_000
        if r["is_sampled"]:
            label = f"{sampled_total:.0f}M\n(of {actual_total:.0f}M)"
        else:
            label = f"{actual_total:.1f}M"
        ax.text(i, sampled_total + 1, label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fn = FIGURE_PATH / "document_counts.pdf"
    plt.savefig(fn)
    print(f'Saved: {fn}')
    plt.close()


if __name__ == "__main__":
    main()
