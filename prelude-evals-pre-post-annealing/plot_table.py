# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Grouped bar chart of the multilingual benchmark table.

Plots the three prelude-8T variants (base / +300BT annealing / +context
extension) against external baselines, one group per benchmark.

OpenSubtitles is BLEU on a 0-100 scale, so it gets its own panel rather than
being squashed into the accuracy axis.

Usage: uv run plot_table.py [-o out.png] [--no-baselines]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Multilingual baselines. results.csv comes from the shared eval campaign;
# see README.md for where to refresh it.
RESULTS_CSV = os.path.join(DATA, "results.csv")
EXTRA_RESULTS_CSVS = [os.path.join(DATA, "results-apertus.csv")]

EXCLUDE_BENCHMARKS = {"Global MGSM"}

# The prelude-8T sweep ran GlobalMMLU's STEM subset
# (global_mmlu_full_<lang>_stem); results.csv holds the full subject set
# (global_mmlu_full_<lang>). Matching them on language lets the benchmark be
# plotted, but the two sides answer different question sets - see the asterisk
# in GLOBAL_MMLU_LABEL.
GLOBAL_MMLU_LABEL = "GlobalMMLU*"

BLEU_BENCHMARK = "OpenSubtitles"

# Files written from the LUMI sweep, in training-progression order.
NEW_MODELS = [
    ("Prelude 8T", "prelude-8T.csv"),
    ("Prelude 8T + 300BT annealing", "prelude-8T-anneal300b.csv"),
    ("Prelude 8T + 300BT ann. + context ext.", "prelude-8T-anneal300b-ctxext.csv"),
]

# Baselines pulled out of results.csv / results-apertus.csv by (data, iter).
# iter=None means the file holds a single released model with no checkpoint.
BASELINES = [
    ("Prelude 4T", "openeurollm/prelude-checkpoints", "iter_0480000"),
    ("Datamix 9b (4T)", "openeurollm/datamix-9b-80-20", "iter_0950000"),
    ("Olmo 3 7B", "allenai/Olmo-3-1025-7B", "stage1-step960000"),
    ("Apertus 8B (v1, 15T)", "swiss-ai/Apertus-8B-2509", "step2627139-tokens15T"),
]


# Row order for both the table and the bar groups.
ROW_ORDER = [
    "Datamix 9b (4T)",
    "Prelude 4T",
    "Prelude 8T",
    "Prelude 8T + 300BT annealing",
    "Prelude 8T + 300BT ann. + context ext.",
    "Olmo 3 7B",
    "Apertus 8B (v1, 15T)",
]


def load_frames(with_baselines=True):
    """Return one long dataframe with a `label` column naming each model."""
    frames = []

    for label, fname in NEW_MODELS:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} - see README.md (step 4)")
        d = pd.read_csv(path)
        d["label"] = label
        frames.append(d)

    if with_baselines:
        pool = pd.concat(
            [pd.read_csv(RESULTS_CSV)]
            + [pd.read_csv(p) for p in EXTRA_RESULTS_CSVS if os.path.exists(p)],
            ignore_index=True,
        )
        for label, data, it in BASELINES:
            d = pool[pool["data"] == data]
            if it is not None:
                d = d[d["iter"] == it]
            if d.empty:
                raise SystemExit(f"no rows for baseline {label} ({data}, {it})")
            d = d.copy()
            d["label"] = label
            frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df = df[~df["benchmark"].isin(EXCLUDE_BENCHMARKS)]

    # Match GlobalMMLU across sources by language, not by raw task name.
    df["task_key"] = df["task"]
    mmlu = df["benchmark"] == "GlobalMMLU"
    df.loc[mmlu, "task_key"] = (
        df.loc[mmlu, "task"].str.replace(r"_stem$", "", regex=True))
    df.loc[mmlu, "benchmark"] = GLOBAL_MMLU_LABEL
    return df


def common_tasks_only(df):
    """Keep only tasks every plotted model has a score for, so each bar
    averages over the identical set of languages."""
    n = df["label"].nunique()
    keep = df.groupby("task_key")["label"].nunique() == n
    return df[df["task_key"].map(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="plot_table.png")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    df = common_tasks_only(load_frames(with_baselines=not args.no_baselines))

    present = set(df["label"])
    labels = [l for l in ROW_ORDER if l in present]

    means = df.groupby(["label", "benchmark"])["score"].mean().unstack()
    means = means.reindex(labels)

    acc = means.drop(columns=[BLEU_BENCHMARK], errors="ignore")
    acc["AVG"] = acc.mean(axis=1)
    bleu = means[[BLEU_BENCHMARK]] if BLEU_BENCHMARK in means else None

    ncols = len(acc.columns) + (1 if bleu is not None else 0)
    fig, axes = plt.subplots(
        1, 2 if bleu is not None else 1,
        figsize=(1.15 * ncols + 3.5, 5.0),
        gridspec_kw={"width_ratios": [len(acc.columns), 1.25]} if bleu is not None else None,
    )
    ax = axes[0] if bleu is not None else axes

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(labels))]
    width = 0.8 / len(labels)

    for i, label in enumerate(labels):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(acc.columns))]
        ax.bar(xs, acc.loc[label], width=width, label=label, color=colors[i])

    ax.set_xticks(range(len(acc.columns)))
    ax.set_xticklabels(acc.columns, rotation=30, ha="right")
    ax.set_ylabel("score (accuracy / exact-match)")
    ax.set_title("Multilingual benchmarks")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(acc.max()) * 1.28)  # headroom for the legend
    ax.legend(fontsize=8, ncol=4, loc="upper center", framealpha=0.95)

    if bleu is not None:
        ax2 = axes[1]
        for i, label in enumerate(labels):
            ax2.bar([i * width - 0.4 + width / 2], bleu.loc[label], width=width,
                    color=colors[i])
        ax2.set_xticks([0])
        ax2.set_xticklabels([BLEU_BENCHMARK], rotation=30, ha="right")
        ax2.set_ylabel("BLEU (0-100)")
        ax2.set_title("Translation")
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_axisbelow(True)

    n_tasks = df.groupby("benchmark")["task_key"].nunique()
    detail = ", ".join(f"{b} {n}" for b, n in n_tasks.items())
    fig.suptitle("Prelude-8T variants vs baselines\n"
                 f"mean over languages common to all models ({detail})\n"
                 "* GlobalMMLU: prelude-8T rows are the STEM subset, baselines the full subject set",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    table = acc.drop(columns=["AVG"])
    if bleu is not None:
        table = table.join(bleu.round(2))
    table["AVG"] = acc["AVG"]          # AVG last; excludes BLEU
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
