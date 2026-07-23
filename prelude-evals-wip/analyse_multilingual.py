# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Show the number of evaluations available per (data, iter) checkpoint,
broken down by benchmark, from results.csv."""
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# dataviz skill categorical palette (fixed order, light mode)
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
GRIDLINE_COLOR = "#e1e0d9"

# bleu/chrf++ live on a different (0-100) scale and would distort the
# average; acc/acc_norm/exact_match are all in [0, 1] so they mix fine
ACCURACY_METRICS = {"acc", "acc_norm", "exact_match"}

# tokens per training iteration = seq_len * global_batch_size
TOKENS_PER_ITER = {
    "openeurollm/datamix-9b-80-20": 2048 * 2048,
    "openeurollm/prelude-checkpoints": 4096 * 2048,
    "allenai/Olmo-3-1025-7B": 4_194_304,
}

APERTUS_TOKENS_RE = re.compile(r"tokens(\d+(?:\.\d+)?)([BT])")
ITER_RE = re.compile(r"iter_(\d+)")
OLMO_STEP_RE = re.compile(r"stage1-step(\d+)")


def compute_tokens_b(row):
    data, iter_ = row["data"], row["iter"]
    if data == "allenai/Olmo-3-1025-7B":
        m = OLMO_STEP_RE.search(iter_)
        if not m:
            return None
        return int(m.group(1)) * TOKENS_PER_ITER[data] / 1e9
    if data in TOKENS_PER_ITER:
        m = ITER_RE.search(iter_)
        if not m:
            return None
        return int(m.group(1)) * TOKENS_PER_ITER[data] / 1e9
    if data == "swiss-ai/Apertus-8B-2509":
        m = APERTUS_TOKENS_RE.search(iter_)
        if not m:
            return None
        value = float(m.group(1))
        return value * 1000 if m.group(2) == "T" else value
    return None


def average_downstream_performance(df, value_col="score"):
    """Average scores over languages within each benchmark first (e.g. all
    arc_challenge_mt_* tasks -> one ARC Challenge_mt score), then average
    over benchmarks to get a single downstream performance number per
    (data, iter, tokens_B)."""
    per_benchmark = df.groupby(["data", "iter", "tokens_B", "benchmark"], dropna=False)[value_col].mean()
    return (
        per_benchmark.groupby(["data", "iter", "tokens_B"], dropna=False)
        .mean()
        .rename("avg_score")
        .reset_index()
    )


def minmax_normalize_scores(df):
    """Min-max normalize each (benchmark, metric) pair to [0, 1] across the
    whole dataset (all current metrics -- acc, acc_norm, exact_match, bleu,
    chrf++ -- are higher-is-better). This puts every metric on a comparable
    scale before averaging across benchmarks."""
    df = df.copy()
    stats = df.groupby(["benchmark", "metric"])["score"].agg(lo="min", hi="max")
    df = df.join(stats, on=["benchmark", "metric"])
    span = df["hi"] - df["lo"]
    df["score_norm"] = ((df["score"] - df["lo"]) / span.replace(0, pd.NA)).fillna(0.5)
    return df.drop(columns=["lo", "hi"])


def plot_performance_vs_tokens(
    avg_df,
    output_path="avg_performance_vs_tokens.png",
    ylabel="Average downstream performance",
    title="Downstream performance vs. tokens trained",
):
    with_tokens = avg_df.dropna(subset=["tokens_B"])
    flat = avg_df[avg_df["tokens_B"].isna()]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (data, group) in enumerate(with_tokens.groupby("data")):
        group = group.sort_values("tokens_B")
        ax.plot(
            group["tokens_B"] / 1000,
            group["avg_score"],
            marker="o",
            color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
            label=data,
        )

    n_lines = with_tokens["data"].nunique()
    for i, (data, group) in enumerate(flat.groupby("data")):
        ax.axhline(
            group["avg_score"].iloc[0],
            linestyle="--",
            color=CATEGORICAL_COLORS[(n_lines + i) % len(CATEGORICAL_COLORS)],
            label=f"{data} (no checkpoints)",
        )

    ax.set_xlabel("Tokens (T)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color=GRIDLINE_COLOR)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nWrote {output_path}")


RESULTS_CSV = "results.csv"
DOWNLOAD_CMD = "scp lumi:/scratch/project_465002530/users/haider/ML_Evals/campaigns/oellm_public_v2/results.csv ."


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Download it with:\n  {DOWNLOAD_CMD}"
        )
    df = pd.read_csv(RESULTS_CSV)
    df["tokens_B"] = df.apply(compute_tokens_b, axis=1)

    completion = df.pivot_table(index=["data", "iter"], columns="task", values="score", aggfunc="count", fill_value=0)
    n_checkpoints = completion.shape[0]
    n_tasks = completion.shape[1]
    pct_done = completion.values.mean() * 100
    print(f"{len(df)} evaluations collected across {n_checkpoints} checkpoints and {n_tasks} tasks, currently {pct_done:.1f}% done.")

    pivot = df.pivot_table(
        index=["data", "iter", "tokens_B"],
        columns="benchmark",
        values="score",
        aggfunc="count",
        fill_value=0,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(pivot.to_string())

    token_range = df.groupby("data")["tokens_B"].agg(["min", "max"])
    print()
    print(token_range.to_string())

    accuracy_df = df[df["metric"].isin(ACCURACY_METRICS)]
    avg_df = average_downstream_performance(accuracy_df)
    plot_performance_vs_tokens(avg_df)

    normalized_df = minmax_normalize_scores(df)
    avg_norm_df = average_downstream_performance(normalized_df, value_col="score_norm")
    plot_performance_vs_tokens(
        avg_norm_df,
        output_path="normalized_overall_performance.png",
        ylabel="Normalized average downstream performance (min-max, all metrics)",
        title="Normalized downstream performance vs. tokens trained",
    )


if __name__ == "__main__":
    main()
