# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""English-benchmark counterpart to plot_table.py.

The prelude-8T variants come from the LUMI sweep (prelude-8T*.csv are
multilingual only, so English scores are read from the collected eval CSV).
Baselines come from compare_prelude_ellamind_suite.csv, which is a wide
one-row-per-model table.

Only benchmarks present on BOTH sides are shown.

Usage: uv run plot_table_english.py [-o out.png] [--no-baselines]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# English baselines (wide, one row per model) and the collected scores for the
# three Prelude checkpoints. See README.md for how both are produced.
SUITE_CSV = os.path.join(DATA, "compare_prelude_ellamind_suite.csv")
EVAL_CSV = os.path.join(DATA, "eval_results.csv")

# LUMI sweep model_name -> label, in training-progression order.
NEW_MODELS = [
    ("/scratch/project_465002530/davisali/models/prelude-iter_0953312",
     "Prelude 8T"),
    ("/scratch/project_465002530/davisali/models/prelude-anneal300b_iter_0989075",
     "Prelude 8T + 300BT annealing"),
    ("birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b",
     "Prelude 8T + 300BT ann. + context ext."),
]

# suite `method` -> label. Checkpoints chosen to match plot_table.py where the
# same model exists there.
BASELINES = [
    ("prelude_iter_0480000", "Prelude 4T"),
    ("oellm_datamix_9b_60_40@step900000", "Datamix 9b 60-40*"),
    ("olmo_3_1025_7b@step1473419", "Olmo 3 7B*"),
    ("apertus_8b@step2627139", "Apertus 8B (v1, 15T)"),
]

# hellaswag is run at both 0- and 10-shot in the sweep; the suite does not
# record its shot setting, so pick one explicitly.
HELLASWAG_NSHOT = 10

# The suite's lambada_openai column is broken for non-prelude models:
# llama3_1_8b scores 0.479 and llama3_1_70b 0.503 where the standard lm-eval
# value is ~0.75, while the prelude rows (~0.70) look right. Comparing across
# sources would unfairly penalise every baseline.
EXCLUDE_BENCHMARKS = {"lambada_openai"}


ROW_ORDER = [
    "Datamix 9b 60-40*",
    "Prelude 4T",
    "Prelude 8T",
    "Prelude 8T + 300BT annealing",
    "Prelude 8T + 300BT ann. + context ext.",
    "Olmo 3 7B*",
    "Apertus 8B (v1, 15T)",
]


def load():
    suite = pd.read_csv(SUITE_CSV).set_index("method")
    suite = suite.drop(columns=["average"], errors="ignore")

    ev = pd.read_csv(EVAL_CSV)
    suite = suite.drop(columns=list(EXCLUDE_BENCHMARKS), errors="ignore")
    ev = ev[ev.task.isin(suite.columns)]
    ev = ev[~((ev.task == "hellaswag") & (ev.n_shot != HELLASWAG_NSHOT))]

    rows = {}
    for model, label in NEW_MODELS:
        s = ev[ev.model_name == model].set_index("task")["performance"]
        rows[label] = s

    for method, label in BASELINES:
        if method not in suite.index:
            raise SystemExit(f"{method} not in {SUITE_CSV}")
        rows[label] = suite.loc[method]

    df = pd.DataFrame(rows).T
    df = df.reindex([l for l in ROW_ORDER if l in df.index])
    # keep only benchmarks every model has
    return df.dropna(axis=1, how="any")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="plot_table_english.png")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    df = load()
    if args.no_baselines:
        df = df.loc[[l for _, l in NEW_MODELS]]
    df["AVG"] = df.mean(axis=1)

    fig, ax = plt.subplots(figsize=(1.15 * len(df.columns) + 4.0, 5.0))
    cmap = plt.get_cmap("tab10")
    labels = list(df.index)
    width = 0.8 / len(labels)

    for i, label in enumerate(labels):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(df.columns))]
        ax.bar(xs, df.loc[label], width=width, label=label, color=cmap(i))

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, df.to_numpy().max() * 1.28)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=4, loc="upper center", framealpha=0.95)
    fig.suptitle("Prelude-8T variants vs baselines - English benchmarks\n"
                 f"hellaswag at {HELLASWAG_NSHOT}-shot; baseline shot settings "
                 "undocumented in the suite CSV", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
