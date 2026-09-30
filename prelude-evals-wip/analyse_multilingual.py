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

# matplotlib Set3 qualitative colormap, one color per method/model
CATEGORICAL_COLORS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
GRIDLINE_COLOR = "#e1e0d9"

# bleu/chrf++ live on a different (0-100) scale and would distort the
# average; acc/acc_norm/exact_match are all in [0, 1] so they mix fine
ACCURACY_METRICS = {"acc", "acc_norm", "exact_match"}

# PolyMath is excluded since it's not suited for base pretrained models
# Not suited to base pretrained models: PolyMath is competition maths, and
# global PIQA (prompted) scores exact_match on a required "The best answer
# is: X" format, so it measures instruction-following rather than
# commonsense - its mean (0.25) sits below the 0.50 a coin flip would get.
EXCLUDED_TASKS = ["PolyMath", "global PIQA (prompted)"]

# tokens per training iteration = seq_len * global_batch_size
TOKENS_PER_ITER = {
    "openeurollm/datamix-9b-80-20": 2048 * 2048,
    "openeurollm/prelude-checkpoints": 4096 * 2048,
    "allenai/Olmo-3-1025-7B": 4_194_304,
}

APERTUS_TOKENS_RE = re.compile(r"tokens(\d+(?:\.\d+)?)([BT])")
ITER_RE = re.compile(r"iter_(\d+)")
OLMO_STEP_RE = re.compile(r"stage1-step(\d+)")

# language code/name (2-letter, 3-letter, or full lowercase name as used in
# task strings) -> (display name, flag emoji)
LANGUAGE_INFO = {
    "bg": ("Bulgarian", "🇧🇬"), "bul": ("Bulgarian", "🇧🇬"), "bulgarian": ("Bulgarian", "🇧🇬"),
    "cs": ("Czech", "🇨🇿"), "ces": ("Czech", "🇨🇿"),
    "da": ("Danish", "🇩🇰"), "dan": ("Danish", "🇩🇰"),
    "de": ("German", "🇩🇪"), "deu": ("German", "🇩🇪"), "german": ("German", "🇩🇪"),
    "el": ("Greek", "🇬🇷"), "ell": ("Greek", "🇬🇷"), "greek": ("Greek", "🇬🇷"),
    "es": ("Spanish", "🇪🇸"), "spa": ("Spanish", "🇪🇸"), "spanish": ("Spanish", "🇪🇸"),
    "et": ("Estonian", "🇪🇪"), "est": ("Estonian", "🇪🇪"), "ekk": ("Estonian", "🇪🇪"), "estonian": ("Estonian", "🇪🇪"),
    "fi": ("Finnish", "🇫🇮"), "fin": ("Finnish", "🇫🇮"), "finnish": ("Finnish", "🇫🇮"),
    "fr": ("French", "🇫🇷"), "fra": ("French", "🇫🇷"), "french": ("French", "🇫🇷"),
    "hu": ("Hungarian", "🇭🇺"), "hun": ("Hungarian", "🇭🇺"), "hungarian": ("Hungarian", "🇭🇺"),
    "is": ("Icelandic", "🇮🇸"), "isl": ("Icelandic", "🇮🇸"),
    "it": ("Italian", "🇮🇹"), "ita": ("Italian", "🇮🇹"), "italian": ("Italian", "🇮🇹"),
    "lt": ("Lithuanian", "🇱🇹"), "lit": ("Lithuanian", "🇱🇹"), "lithuanian": ("Lithuanian", "🇱🇹"),
    "lv": ("Latvian", "🇱🇻"), "lav": ("Latvian", "🇱🇻"), "lvs": ("Latvian", "🇱🇻"),
    "nb": ("Norwegian", "🇳🇴"), "nob": ("Norwegian", "🇳🇴"), "no": ("Norwegian", "🇳🇴"), "nno": ("Norwegian", "🇳🇴"),
    "nl": ("Dutch", "🇳🇱"), "nld": ("Dutch", "🇳🇱"), "dutch": ("Dutch", "🇳🇱"),
    "pl": ("Polish", "🇵🇱"), "pol": ("Polish", "🇵🇱"), "polish": ("Polish", "🇵🇱"),
    "pt": ("Portuguese", "🇵🇹"), "por": ("Portuguese", "🇵🇹"), "portuguese": ("Portuguese", "🇵🇹"),
    "ro": ("Romanian", "🇷🇴"), "ron": ("Romanian", "🇷🇴"),
    "sk": ("Slovak", "🇸🇰"), "slk": ("Slovak", "🇸🇰"),
    "sl": ("Slovenian", "🇸🇮"), "slv": ("Slovenian", "🇸🇮"),
    "sv": ("Swedish", "🇸🇪"), "swe": ("Swedish", "🇸🇪"),
    "ca": ("Catalan", "🇪🇸"), "cat": ("Catalan", "🇪🇸"),
    "en": ("English", "🇬🇧"), "eng": ("English", "🇬🇧"),
    "eu": ("Basque", "🇪🇸"), "eus": ("Basque", "🇪🇸"), "basque": ("Basque", "🇪🇸"),
    "gl": ("Galician", "🇪🇸"), "glg": ("Galician", "🇪🇸"),
    "sr": ("Serbian", "🇷🇸"), "srp": ("Serbian", "🇷🇸"), "hbs": ("Serbian", "🇷🇸"), "serbian": ("Serbian", "🇷🇸"),
    "he": ("Hebrew", "🇮🇱"),
    "ru": ("Russian", "🇷🇺"), "russian": ("Russian", "🇷🇺"),
    "tr": ("Turkish", "🇹🇷"), "tur": ("Turkish", "🇹🇷"), "turkish": ("Turkish", "🇹🇷"),
    "uk": ("Ukrainian", "🇺🇦"), "ukr": ("Ukrainian", "🇺🇦"), "ukrainian": ("Ukrainian", "🇺🇦"),
    "hr": ("Croatian", "🇭🇷"), "hrv": ("Croatian", "🇭🇷"), "croatian": ("Croatian", "🇭🇷"),
    "als": ("Albanian", "🇦🇱"), "sqi": ("Albanian", "🇦🇱"), "albanian": ("Albanian", "🇦🇱"),
    "bos": ("Bosnian", "🇧🇦"),
    "gle": ("Irish", "🇮🇪"),
    "kat": ("Georgian", "🇬🇪"), "georgian": ("Georgian", "🇬🇪"),
    "mkd": ("North Macedonian", "🇲🇰"), "north macedonian": ("North Macedonian", "🇲🇰"),
    "mlt": ("Maltese", "🇲🇹"),
    "armenian": ("Armenian", "🇦🇲"),
    "azerbaijani": ("Azerbaijani", "🇦🇿"),
    "belarusian": ("Belarusian", "🇧🇾"),
}


def task_language_code(benchmark, task):
    """Extract the raw language code/name embedded in a task string. Each
    benchmark encodes it in a different position (2-letter, 3-letter+script,
    or a full lowercase name), so this dispatches per benchmark."""
    if benchmark in ("ARC Challenge_mt", "Global MGSM", "GlobalMMLU", "Mgsm", "MultiBlimp", "xHellaswag"):
        return task.rsplit("_", 1)[-1]
    if benchmark in ("BeleBele", "PolyMath", "SIB-200", "Xcsqa"):
        return task.split("_")[1]
    if benchmark == "Flores-200":
        left, right = task.split(":", 1)[1].split("-")
        left_code, right_code = left.split("_")[0], right.split("_")[0]
        return right_code if left_code == "eng" else left_code
    if benchmark == "INCLUDE":
        return task.split("include_base_44_", 1)[-1]
    if benchmark == "OpenSubtitles":
        parts = task.split("_")  # opensubtitles_multi40_<src>_to_<tgt>
        src, tgt = parts[2], parts[4]
        return tgt if src == "en" else src
    if benchmark == "XCOPA":
        return task.split(":", 1)[1]
    if benchmark.startswith("global PIQA"):
        return task.split("_")[3]
    return None


def resolve_language(benchmark, task):
    code = task_language_code(benchmark, task)
    if code is None:
        return None
    info = LANGUAGE_INFO.get(code.lower())
    return info[0] if info else None



# results.csv files both global_piqa variants under a single "global PIQA"
# benchmark, but they are not the same measurement: `completions` is
# multiple_choice over two solutions (acc_norm, chance = 0.50), while `prompted`
# is generate_until scored by exact_match on a required "The best answer is: X"
# format (chance = 0.00, since a model that ignores the format scores nothing).
# Averaging them mixes two different floors, so they are split into separate
# benchmarks at load time.
def split_global_piqa(df):
    is_piqa = df["benchmark"] == "global PIQA"
    if not is_piqa.any():
        return df
    df = df.copy()
    variant = df.loc[is_piqa, "task"].str.contains("completions")
    df.loc[is_piqa, "benchmark"] = variant.map(
        {True: "global PIQA (completions)", False: "global PIQA (prompted)"})
    return df


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


FIGURES_DIR = "figures"


def plot_performance_vs_tokens(
    avg_df,
    output_name="avg_performance_vs_tokens",
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

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{output_name}.{ext}")
        fig.savefig(path, dpi=150)
        print(f"\nWrote {path}")


RESULTS_CSV = "results.csv"
EXTRA_RESULTS_CSVS = ["results-apertus.csv"]
DOWNLOAD_CMD = "scp lumi:/scratch/project_465002530/users/haider/ML_Evals/campaigns/oellm_public_v2/results.csv ."


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Download it with:\n  {DOWNLOAD_CMD}"
        )
    dfs = [pd.read_csv(RESULTS_CSV)]
    for path in EXTRA_RESULTS_CSVS:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    df = split_global_piqa(df)
    df = df[~df["benchmark"].isin(EXCLUDED_TASKS)]
    df["tokens_B"] = df.apply(compute_tokens_b, axis=1)
    df["language"] = df.apply(lambda row: resolve_language(row["benchmark"], row["task"]), axis=1)

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

    belebele_df = accuracy_df[accuracy_df["benchmark"] == "BeleBele"]
    avg_belebele_df = average_downstream_performance(belebele_df)
    plot_performance_vs_tokens(
        avg_belebele_df,
        output_name="avg_performance_vs_tokens_belebele",
        ylabel="Average BeleBele performance",
        title="BeleBele performance vs. tokens trained",
    )

    normalized_df = minmax_normalize_scores(df)
    avg_norm_df = average_downstream_performance(normalized_df, value_col="score_norm")
    plot_performance_vs_tokens(
        avg_norm_df,
        output_name="normalized_overall_performance",
        ylabel="Normalized average downstream performance (min-max, all metrics)",
        title="Normalized downstream performance vs. tokens trained",
    )


if __name__ == "__main__":
    main()
