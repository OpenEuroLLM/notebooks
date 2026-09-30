# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Grouped bar charts of the multilingual and English benchmark tables.

Plots the three prelude-8T variants (base / +300BT annealing / +context
extension) against external baselines, one group per benchmark, and prints
both tables.

Multilingual: OpenSubtitles is BLEU on a 0-100 scale, so it gets its own panel
rather than being squashed into the accuracy axis.

English: the prelude-8T variants come from the LUMI sweep (prelude-8T*.csv are
multilingual only, so English scores are read from the collected eval CSV).
Baselines come from compare_prelude_ellamind_suite.csv, which is a wide
one-row-per-model table. Only benchmarks present on BOTH sides are shown.

Usage: uv run plot_table.py [-o out.png] [--out-english out.png] [--no-baselines]
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Multilingual baselines. results.csv comes from the shared eval campaign;
# see README.md for where to refresh it.
RESULTS_CSV = os.path.join(DATA, "results.csv")
EXTRA_RESULTS_CSVS = [os.path.join(DATA, "results-apertus.csv")]

# 0-shot generative maths: base models score ~0, so these measure answer
# formatting rather than maths (see README.md).
EXCLUDE_BENCHMARKS = {"Global MGSM", "PolyMath"}

BLEU_BENCHMARK = "OpenSubtitles"

# Prioritized OpenEuroLLM target languages (EU official, co-official, candidate
# members, Icelandic/Norwegian): ISO 639-3 -> (ISO 639-1, name, other codes).
TARGET_LANGUAGES = {
    "bul": ("bg", "bulgarian", []),
    "ces": ("cs", "czech", []),
    "dan": ("da", "danish", []),
    "deu": ("de", "german", []),
    "ell": ("el", "greek", []),
    "eng": ("en", "english", []),
    "est": ("et", "estonian", ["ekk"]),
    "fin": ("fi", "finnish", []),
    "fra": ("fr", "french", []),
    "gle": ("ga", "irish", []),
    "hrv": ("hr", "croatian", []),
    "hun": ("hu", "hungarian", []),
    "ita": ("it", "italian", []),
    "lav": ("lv", "latvian", ["ltg", "lvs"]),
    "lit": ("lt", "lithuanian", []),
    "mlt": ("mt", "maltese", []),
    "nld": ("nl", "dutch", []),
    "pol": ("pl", "polish", []),
    "por": ("pt", "portuguese", []),
    "ron": ("ro", "romanian", []),
    "slk": ("sk", "slovak", []),
    "slv": ("sl", "slovene", ["slovenian"]),
    "spa": ("es", "spanish", []),
    "swe": ("sv", "swedish", []),
    "cat": ("ca", "catalan", []),
    "eus": ("eu", "basque", []),
    "glg": ("gl", "galician", []),
    "bos": ("bs", "bosnian", []),
    "kat": ("ka", "georgian", []),
    "mkd": ("mk", "macedonian", ["north macedonian", "north_macedonian"]),
    "sqi": ("sq", "albanian", ["als"]),
    "srp": ("sr", "serbian", []),
    "tur": ("tr", "turkish", []),
    "ukr": ("uk", "ukrainian", []),
    "isl": ("is", "icelandic", []),
    "nor": ("no", "norwegian", ["nno", "nob", "nb", "nn"]),
}
_TARGET_CODES = {c for k, (iso1, name, other) in TARGET_LANGUAGES.items()
                 for c in (k, iso1, name, *other)}

# Task-name prefixes before the language part of a multilingual task.
_LANG_PREFIX = re.compile(
    r"^(arc_challenge_mt|belebele|global_mmlu_full|include_base_44|sib200|xcsqa"
    r"|opensubtitles_multi40|multiblimp|polymath|hellaswag|global_mgsm"
    r"|global_piqa_completions|global_piqa_prompted|mgsm_native_cot)_"
    r"|^(flores200|xcopa):")


def filter_languages(lang: str) -> bool:
    """True if `lang` is a target language. Accepts ISO 639-3 or 639-1 codes,
    a script-suffixed code (kat_Geor, als_latn) or an English name."""
    lang = lang.lower()
    return lang in _TARGET_CODES or lang.split("_")[0] in _TARGET_CODES


def task_language(task: str) -> str | None:
    """Language part of a multilingual task name; for translation pairs
    (opensubtitles bg_to_en) the non-English side."""
    m = _LANG_PREFIX.match(task)
    if m is None:
        return None
    rest = task[m.end():]
    for sep in ("_to_", "-"):       # opensubtitles bg_to_en, flores200 X-eng_Latn
        if sep in rest:
            src, tgt = rest.split(sep)
            return tgt if src.split("_")[0] in ("en", "eng") else src
    if rest in _TARGET_CODES:       # multi-word names, e.g. north_macedonian
        return rest
    return rest.split("_")[0]

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
    ("Apertus 8B (v1, 15T)", "swiss-ai/Apertus-8B-2509", "step2627139-tokens15T"),
]

# Baselines absent from results.csv, rerun in the same LUMI setup and exported
# by export_results.py. The rerun hit timeouts, so they miss some languages -
# common_tasks_only then narrows every benchmark to what they have.
RERUN_BASELINES = [
    ("Marin 8B", "marin-8b.csv"),
    ("EuroLLM 9B", "eurollm-9b.csv"),
    # Our run of the HF release, used instead of results.csv's Olmo 3 rows.
    ("Olmo 3 7B", "olmo3-7b-ownrun.csv"),
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
    "Marin 8B",
    "EuroLLM 9B",
]


# English baselines (wide, one row per model) and the collected scores for the
# three Prelude checkpoints. See README.md for how both are produced.
SUITE_CSV = os.path.join(DATA, "compare_prelude_ellamind_suite.csv")
EVAL_CSV = os.path.join(DATA, "eval_results.csv")
EVAL_BASELINES_CSV = os.path.join(DATA, "eval_results_baselines.csv")

# LUMI sweep model_name -> label, in training-progression order.
NEW_MODELS_EN = [
    ("/scratch/project_465002530/davisali/models/prelude-iter_0953312",
     "Prelude 8T"),
    ("/scratch/project_465002530/davisali/models/prelude-anneal300b_iter_0989075",
     "Prelude 8T + 300BT annealing"),
    ("birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b",
     "Prelude 8T + 300BT ann. + context ext."),
]

# suite `method` -> label. Checkpoints chosen to match plot_table.py where the
# same model exists in the multilingual table.
BASELINES_EN = [
    ("prelude_iter_0480000", "Prelude 4T"),
    ("oellm_datamix_9b_60_40@step900000", "Datamix 9b 60-40*"),
    ("apertus_8b@step2627139", "Apertus 8B (v1, 15T)"),
]

# Rerun baselines (EVAL_BASELINES_CSV model_name -> label), same setup as the
# Prelude rows.
M = "/scratch/project_465002530/davisali/models"

RERUN_BASELINES_EN = [
    ("marin-community/marin-8b-base", "Marin 8B"),
    ("utter-project/EuroLLM-9B", "EuroLLM 9B"),
    ("allenai/Olmo-3-1025-7B", "Olmo 3 7B"),
    # dclm-core-22 rerun of the four baselines that previously existed only in
    # compare_prelude_ellamind_suite.csv, so the English table no longer falls
    # back to that file's 6-benchmark overlap.
    (f"{M}/prelude-iter_0480000", "Prelude 4T"),
    (f"{M}/datamix-9b-80-20-iter_0950000", "Datamix 9b (4T)"),
    (f"{M}/apertus-8b-step2627139-tokens15T", "Apertus 8B (v1, 15T)"),
]

# hellaswag is run at both 0- and 10-shot in the sweep; the suite does not
# record its shot setting, so pick one explicitly.
HELLASWAG_NSHOT = 10

# The suite's lambada_openai column is broken for non-prelude models:
# llama3_1_8b scores 0.479 and llama3_1_70b 0.503 where the standard lm-eval
# value is ~0.75, while the prelude rows (~0.70) look right. Comparing across
# sources would unfairly penalise every baseline.
EXCLUDE_BENCHMARKS_EN = {"lambada_openai"}


ROW_ORDER_EN = [
    "Datamix 9b (4T)",
    "Datamix 9b 60-40*",
    "Prelude 4T",
    "Prelude 8T",
    "Prelude 8T + 300BT annealing",
    "Prelude 8T + 300BT ann. + context ext.",
    "Olmo 3 7B",
    "Apertus 8B (v1, 15T)",
    "Marin 8B",
    "EuroLLM 9B",
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
        for label, fname in RERUN_BASELINES:
            d = pd.read_csv(os.path.join(DATA, fname))
            d["label"] = label
            frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    # GlobalMMLU is the full subject set for every model; collect before oellm-cli
    # ba1ff8c (#109) kept only the _stem subgroup, so re-collect with a later version.
    df = df[~df["benchmark"].isin(EXCLUDE_BENCHMARKS)]
    lang = df["task"].map(task_language)
    unparsed = sorted(df.loc[lang.isna(), "task"].unique())
    if unparsed:
        raise SystemExit(f"no language found for tasks: {unparsed[:5]}")
    return df[lang.map(filter_languages)]


def common_tasks_only(df):
    """Keep only tasks every plotted model has a score for, so each bar
    averages over the identical set of languages."""
    n = df["label"].nunique()
    keep = df.groupby("task")["label"].nunique() == n
    return df[df["task"].map(keep)]


def to_markdown(df):
    """Markdown table with two decimals; the best (highest) value of each
    row is bolded. Every metric here is higher-is-better."""
    best = df.max(axis=1)
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" + "---:|" * len(df.columns)]
    for label, row in df.iterrows():
        cells = []
        for col, v in row.items():
            if pd.isna(v):
                cells.append("–")
                continue
            cell = f"{v:.2f}"
            if round(v, 2) == round(best[label], 2):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join([str(label)] + cells) + " |")
    return "\n".join(lines)


# Full dclm-core-22. The English table used to be capped at the 6 benchmarks
# that compare_prelude_ellamind_suite.csv also carries; now that the baselines
# have been rerun on our own suite, the whole group is available for every
# same-suite model. agieval_lsat_ar, squadv2 and
# bigbench_language_identification_multiple_choice are absent for everyone
# (the first two are lm-eval 0.4.13 bugs), leaving 18.
DCLM_TASKS = [
    "agieval_lsat_ar", "arc_easy", "arc_challenge", "boolq", "commonsense_qa",
    "copa", "hellaswag", "openbookqa", "piqa",
    "bigbench_language_identification_multiple_choice", "winogrande", "wsc273",
    "lambada_openai", "bigbench_qa_wikidata_generate_until",
    "bigbench_dyck_languages_generate_until", "bigbench_operators_generate_until",
    "bigbench_repeat_copy_logic_generate_until",
    "bigbench_cs_algorithms_generate_until", "coqa", "squadv2", "jeopardy",
]


def load_english_dclm():
    """Full dclm-core-22, same-suite models only (no compare-suite baselines)."""
    ev = pd.concat([pd.read_csv(EVAL_CSV), pd.read_csv(EVAL_BASELINES_CSV)],
                   ignore_index=True)
    ev = ev[ev.task.isin(DCLM_TASKS)]
    n = ev.groupby("task")["n_shot"].nunique()
    ev["bench"] = ev.apply(
        lambda r: f"{r['task']}_{r['n_shot']}s" if n[r["task"]] > 1 else r["task"],
        axis=1)

    rows = {}
    for model, label in NEW_MODELS_EN + RERUN_BASELINES_EN:
        rows[label] = ev[ev.model_name == model].set_index("bench")["performance"]
    df = pd.DataFrame(rows).T
    order = [l for _, l in NEW_MODELS_EN] + [l for _, l in RERUN_BASELINES_EN]
    df = df.reindex([l for l in ROW_ORDER_EN if l in order] +
                    [l for l in order if l not in ROW_ORDER_EN])
    return df.dropna(axis=1, how="any")


def load_english():
    suite = pd.read_csv(SUITE_CSV).set_index("method")
    suite = suite.drop(columns=["average"], errors="ignore")

    ev = pd.concat([pd.read_csv(EVAL_CSV), pd.read_csv(EVAL_BASELINES_CSV)],
                   ignore_index=True)
    suite = suite.drop(columns=list(EXCLUDE_BENCHMARKS_EN), errors="ignore")
    ev = ev[ev.task.isin(suite.columns)]
    ev = ev[~((ev.task == "hellaswag") & (ev.n_shot != HELLASWAG_NSHOT))]

    rows = {}
    for model, label in NEW_MODELS_EN:
        s = ev[ev.model_name == model].set_index("task")["performance"]
        rows[label] = s

    for method, label in BASELINES_EN:
        if method not in suite.index:
            raise SystemExit(f"{method} not in {SUITE_CSV}")
        rows[label] = suite.loc[method]

    for model, label in RERUN_BASELINES_EN:
        rows[label] = ev[ev.model_name == model].set_index("task")["performance"]

    df = pd.DataFrame(rows).T
    df = df.reindex([l for l in ROW_ORDER_EN if l in df.index])
    # keep only benchmarks every model has
    return df.dropna(axis=1, how="any")


def plot_multilingual(out, with_baselines=True):
    df = common_tasks_only(load_frames(with_baselines=with_baselines))

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

    n_tasks = df.groupby("benchmark")["task"].nunique()
    detail = ", ".join(f"{b} {n}" for b, n in n_tasks.items())
    fig.suptitle("Prelude-8T variants vs baselines\n"
                 f"mean over languages common to all models ({detail})",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    table = acc.drop(columns=["AVG"])
    if bleu is not None:
        table = table.join(bleu.round(2))
    table["AVG"] = acc["AVG"]          # AVG last; excludes BLEU
    table = table.T                    # benchmarks as rows, models as columns
    plot_ranks(table, "plot_table_ranks.png", "multilingual")
    print("\nMultilingual")
    print(table.round(2).to_string())
    print()
    print(to_markdown(table))



def plot_english(out, with_baselines=True, full_dclm=True):
    df = load_english_dclm() if full_dclm else load_english()
    if not with_baselines:
        df = df.loc[[l for _, l in NEW_MODELS_EN]]
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
    sub = ("full dclm-core-22, all models run on the same suite"
           if full_dclm else
           f"hellaswag at {HELLASWAG_NSHOT}-shot; baseline shot settings "
           "undocumented in the suite CSV")
    fig.suptitle(f"Prelude-8T variants vs baselines - English benchmarks\n{sub}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plot_ranks(df.T, "plot_table_english_ranks.png", "English")
    print("\nEnglish")
    print(df.T.round(2).to_string())
    print()
    print(to_markdown(df.T))



def plot_ranks(scores, out, what):
    """Rank of each model within every benchmark (1 = best), plus AVG.

    `scores` is benchmarks x models, AVG included. Rank is ordinal magnitude, so
    the cells use one sequential hue rather than categorical colours, and every
    cell carries its rank as a number - colour is the secondary encoding, never
    the only one.
    """
    # every metric here is higher-is-better; ties share the better rank
    ranks = scores.rank(axis=1, ascending=False, method="min").astype(int)

    # AVG last, separated from the per-benchmark rows
    order = [b for b in ranks.index if b != "AVG"] + ["AVG"]
    ranks, scores = ranks.loc[order], scores.loc[order]

    n_row, n_col = ranks.shape
    fig, ax = plt.subplots(figsize=(0.95 * n_col + 5.0, 0.42 * n_row + 2.4))
    ax.imshow(ranks.to_numpy(), cmap="Blues_r", vmin=1, vmax=n_col,
              aspect="auto")

    for i in range(n_row):
        for j in range(n_col):
            r = ranks.iat[i, j]
            # ink stays legible against the ramp; no series colour in text
            ax.text(j, i, str(r), ha="center", va="center", fontsize=9,
                    color="white" if r <= n_col / 2.5 else "#1a1a1a",
                    fontweight="bold" if r == 1 else "normal")

    ax.set_xticks(range(n_col))
    ax.set_xticklabels(ranks.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_row))
    ax.set_yticklabels(ranks.index, fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 2px surface gap between cells
    ax.set_xticks([x - 0.5 for x in range(1, n_col)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_row)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    # divider above AVG
    ax.axhline(n_row - 1.5, color="#1a1a1a", linewidth=1.5)

    ax.set_title(f"Rank among the {n_col} methods per {what} benchmark "
                 f"(1 = best of {n_col}; darker = better)", fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"\n{what} ranks (1 = best)")
    print(ranks.to_string())
    print()
    n_bench = len(ranks) - 1
    print(f"mean rank across the {n_bench} benchmarks:")
    print(ranks.drop(index="AVG").mean().round(2).sort_values().to_string())
    return ranks



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="plot_table.png",
                    help="multilingual chart")
    ap.add_argument("--out-english", default="plot_table_english.png")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--suite-english", action="store_true",
                    help="English table from compare_prelude_ellamind_suite.csv "
                         "(adds Apertus/Datamix/Prelude 4T, but only the 6 "
                         "benchmarks that file shares with dclm-core-22)")
    args = ap.parse_args()

    plot_multilingual(args.out, with_baselines=not args.no_baselines)
    plot_english(args.out_english, with_baselines=not args.no_baselines,
                 full_dclm=not args.suite_english)

if __name__ == "__main__":
    main()
