"""Compare Prelude "baby" checkpoint results (eval_results_baby_all.csv) against the
Ellamind eval suite (scores.parquet).

The two sources name tasks differently:
  - eval_results_baby_all.csv uses lm-eval-harness task names, e.g. "arc_challenge".
  - scores.parquet appends a task *formulation* suffix, e.g. "arc_challenge_mc" /
    "_rc" (rank/continuation-choice) / "_bpb" (bits-per-byte) / "_gen" (generative) /
    "_code". A few tasks are also renamed outright (social_iqa -> socialiqa,
    squadv2 -> squad, lambada_openai -> lambada).

Resolution strategy per baby task:
  1. Apply the rename map to get the scores.parquet base name.
  2. Enumerate candidate parquet tasks: {base}_{suffix} for suffix in FORMULATION_SUFFIXES.
  3. Keep candidates whose available metric names overlap (via METRIC_SYNONYMS) with
     the baby task's metric.
  4. If a manual override exists for the task (TASK_OVERRIDES), prefer that formulation.
  5. Otherwise, if still ambiguous, break ties with TIE_BREAK_ORDER (rc preferred, since
     that matches the plain-continuation lm-eval-harness convention used by most of
     these tasks; mmlu and lambada_openai are the two exceptions, handled by overrides).
  6. Tasks with zero candidates (boolq, copa, wsc273, openbookqa, commonsense_qa, ifeval,
     bigbench_*) are simply absent from scores.parquet's suite and are dropped.

Output: a table with one row per method (baby checkpoint, or scores.parquet
model/step) and one column per resolved benchmark, plus an "average" column,
sorted descending by average.
"""

import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

BABY_CSV = "eval_results_baby_all.csv"
SCORES_PARQUET = "scores.parquet"
OUTPUT_CSV = "compare_prelude_ellamind_suite.csv"

RENAME_MAP = {
    "social_iqa": "socialiqa",
    "squadv2": "squad",
    "lambada_openai": "lambada",
}

# Tasks whose score is on a 0-100 scale in the baby csv but 0-1 in scores.parquet.
RESCALE_DIV100 = {"squadv2"}

FORMULATION_SUFFIXES = ["rc", "mc", "gen", "code", "cot", "gen_olmes", "bpb"]

METRIC_SYNONYMS = {
    "exact_match": {"exact_match", "em"},
    "em": {"exact_match", "em"},
    "acc": {"acc"},
    "acc_norm": {"acc_norm"},
    "f1": {"f1"},
    "bits_per_byte": {"bits_per_byte"},
    "pass_at_1": {"pass_at_1", "pass@1"},
    "pass_at_2": {"pass_at_2", "pass@2"},
    "pass_at_4": {"pass_at_4", "pass@4"},
    "pass_at_8": {"pass_at_8", "pass@8"},
    "pass_at_16": {"pass_at_16", "pass@16"},
    "maj_at_1": {"maj_at_1"},
    "maj_at_2": {"maj_at_2"},
    "maj_at_4": {"maj_at_4"},
    "maj_at_16": {"maj_at_16"},
}

# Manual override when metric-based matching leaves more than one candidate
# formulation. Keyed by the *original* baby task name.
TASK_OVERRIDES = {
    "mmlu": "mc",  # standard lm-eval-harness mmlu uses the lettered A/B/C/D prompt
    "lambada_openai": "gen",  # standard lambada_openai is a generative task
}

TIE_BREAK_ORDER = ["rc", "gen", "code", "mc", "cot", "gen_olmes", "bpb"]


def resolve_task(baby_task: str, baby_metric: str, task_to_metrics: dict[str, set[str]]):
    """Return (parquet_task, matched_metric) for a baby task, or None if absent."""
    base = RENAME_MAP.get(baby_task, baby_task)
    synonyms = METRIC_SYNONYMS.get(baby_metric, {baby_metric})

    candidates = {}  # suffix -> (parquet_task, matched_metric)
    for suffix in FORMULATION_SUFFIXES:
        parquet_task = f"{base}_{suffix}"
        metrics = task_to_metrics.get(parquet_task)
        if metrics is None:
            continue
        matched = synonyms & metrics
        if matched:
            candidates[suffix] = (parquet_task, sorted(matched)[0])

    if not candidates:
        return None

    override = TASK_OVERRIDES.get(baby_task)
    if override and override in candidates:
        return candidates[override]

    if len(candidates) == 1:
        return next(iter(candidates.values()))

    for suffix in TIE_BREAK_ORDER:
        if suffix in candidates:
            return candidates[suffix]

    return next(iter(candidates.values()))


def load_baby(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["metric"] = df["metric_name"].str.split(",").str[0]
    df["method"] = df["model_name"].apply(lambda p: "prelude_" + p.rstrip("/").split("/")[-1])
    return df


def build_baby_wide(df: pd.DataFrame, resolution: dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        task = row["task"]
        if task not in resolution:
            continue
        _, matched_metric = resolution[task]
        if matched_metric not in METRIC_SYNONYMS.get(row["metric"], {row["metric"]}):
            continue
        value = row["performance"]
        if task in RESCALE_DIV100:
            value = value / 100.0
        rows.append({"method": row["method"], "benchmark": task, "score": value})
    long = pd.DataFrame(rows)
    return long.pivot_table(index="method", columns="benchmark", values="score", aggfunc="mean")


def get_resolution(baby: pd.DataFrame, scores: pd.DataFrame) -> dict[str, tuple]:
    task_to_metrics = scores.groupby("task")["metric"].apply(set).to_dict()
    baby_tasks = baby[["task", "metric"]].drop_duplicates()
    resolution = {}
    for _, r in baby_tasks.iterrows():
        result = resolve_task(r["task"], r["metric"], task_to_metrics)
        if result is not None:
            resolution[r["task"]] = result
    return resolution


def resolved_scores(df: pd.DataFrame, resolution: dict[str, tuple]) -> pd.DataFrame:
    """scores.parquet rows restricted to the resolved benchmarks/metrics, with a
    "benchmark" column holding the baby-csv task name."""
    parquet_lookup = {v[0]: (k, v[1]) for k, v in resolution.items()}
    sub = df[df["task"].isin(parquet_lookup.keys()) & (df["metric_filter"] == "none")].copy()
    sub["benchmark"] = sub["task"].map(lambda t: parquet_lookup[t][0])
    sub["expected_metric"] = sub["task"].map(lambda t: parquet_lookup[t][1])
    return sub[sub["metric"] == sub["expected_metric"]]


def build_parquet_wide(df: pd.DataFrame, resolution: dict[str, tuple]) -> pd.DataFrame:
    sub = resolved_scores(df, resolution)

    def method_label(r):
        if pd.isna(r["step"]):
            return r["model"]
        return f"{r['model']}@step{int(r['step'])}"

    sub["method"] = sub.apply(method_label, axis=1)
    return sub.pivot_table(index="method", columns="benchmark", values="score", aggfunc="mean")


# eval_results_baby_all.csv has no tokens_trained column, only training step
# ("iter_XXXXXXX"). The baby checkpoints come from the same oellm-autoexp project
# as the oellm_datamix_* models in scores.parquet, which all report exactly
# 4,194,304 tokens/step (batch_size * seq_len). We assume the same constant here;
# if baby's actual batch size/seq len differ, this conversion should be adjusted.
PRELUDE_TOKENS_PER_STEP = 4_194_304

ITERATION_MODELS = ["Apertus 8B", "OELLM datamix 9B 60-40", "Olmo 3 7B"]
FLAT_MODELS = [
    "EuroLLM 9B 2512",
    "Nemotron 3 Nano 30B-A3B",
    "Qwen3 8B",
    "Qwen3.5 9B",
    "Salamandra 7B",
    "Teuken 7B",
]


def latest_iteration_table(
    baby_csv: str = BABY_CSV,
    scores_parquet: str = SCORES_PARQUET,
) -> pd.DataFrame:
    """Benchmark-by-model table (benchmark as index, model as column) using only
    the latest available checkpoint for each of the ITERATION_MODELS/Prelude, and
    the single reported checkpoint for each of the FLAT_MODELS."""
    baby = load_baby(baby_csv)
    scores = pd.read_parquet(scores_parquet)
    resolution = get_resolution(baby, scores)
    sub = resolved_scores(scores, resolution)

    display_to_key = dict(scores[["model_display_name", "model"]].drop_duplicates().values)

    columns = {}

    baby_wide = build_baby_wide(baby, resolution)
    steps = baby_wide.index.str.extract(r"iter_0*(\d+)$")[0].astype(int)
    latest_label = baby_wide.index[steps.values.argmax()]
    columns["Prelude (baby)"] = baby_wide.loc[latest_label]

    for name in ITERATION_MODELS:
        key = display_to_key[name]
        msub = sub[sub["model"] == key]
        wide = msub.pivot_table(index="tokens_trained", columns="benchmark", values="score", aggfunc="mean")
        columns[name] = wide.sort_index().iloc[-1]

    for name in FLAT_MODELS:
        key = display_to_key[name]
        msub = sub[sub["model"] == key]
        wide = msub.pivot_table(columns="benchmark", values="score", aggfunc="mean")
        columns[name] = wide.iloc[0] if len(wide) else pd.Series(dtype=float)

    table = pd.DataFrame(columns)
    table.loc["average"] = table.mean(axis=0, skipna=True)
    return table.round(2)


def plot_iteration(
    baby_csv: str = BABY_CSV,
    scores_parquet: str = SCORES_PARQUET,
    output_path: str = "iteration_plot.pdf",
):
    """Plot average downstream performance vs. #tokens trained for a group of
    models with intermediate checkpoints (ITERATION_MODELS), overlaid with
    horizontal reference lines for models only evaluated at a single, final
    checkpoint (FLAT_MODELS, no tokens_trained trajectory available)."""
    baby = load_baby(baby_csv)
    scores = pd.read_parquet(scores_parquet)
    resolution = get_resolution(baby, scores)
    sub = resolved_scores(scores, resolution)

    display_to_key = dict(scores[["model_display_name", "model"]].drop_duplicates().values)

    fig, ax = plt.subplots(figsize=(9, 6))

    baby_wide = build_baby_wide(baby, resolution)
    baby_wide["step"] = baby_wide.index.str.extract(r"iter_0*(\d+)$")[0].astype(int).values
    baby_wide["tokens_trained"] = baby_wide["step"] * PRELUDE_TOKENS_PER_STEP
    baby_avg = baby_wide.drop(columns=["step", "tokens_trained"]).mean(axis=1, skipna=True)
    baby_avg = baby_avg.reindex(baby_wide.sort_values("tokens_trained").index)
    ax.plot(
        baby_wide.sort_values("tokens_trained")["tokens_trained"],
        baby_avg.values,
        marker="o",
        label="Prelude (baby)",
    )

    for name in ITERATION_MODELS:
        key = display_to_key[name]
        msub = sub[sub["model"] == key]
        wide = msub.pivot_table(index="tokens_trained", columns="benchmark", values="score", aggfunc="mean")
        avg = wide.mean(axis=1, skipna=True).sort_index()
        ax.plot(avg.index, avg.values, marker="o", label=name)

    plot_flat_models = [m for m in FLAT_MODELS if m != "Salamandra 7B"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(plot_flat_models):
        key = display_to_key[name]
        msub = sub[sub["model"] == key]
        wide = msub.pivot_table(columns="benchmark", values="score", aggfunc="mean")
        avg = wide.mean(axis=1, skipna=True)
        value = avg.iloc[0] if len(avg) else float("nan")
        ax.axhline(
            value,
            linestyle="--",
            color=color_cycle[(len(ITERATION_MODELS) + 1 + i) % len(color_cycle)],
            label=f"{name} (no checkpoints)",
        )

    ax.set_xlabel("#Tokens trained")
    ax.set_ylabel("Average downstream performance")
    ax.set_title("Downstream performance vs. tokens trained")
    ax.set_ylim(bottom=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")
    return fig


def main():
    baby = load_baby(BABY_CSV)
    scores = pd.read_parquet(SCORES_PARQUET)

    resolution = get_resolution(baby, scores)
    baby_tasks = baby[["task", "metric"]].drop_duplicates()
    unresolved = sorted(set(baby_tasks["task"]) - set(resolution.keys()))

    print(f"Resolved {len(resolution)}/{len(baby_tasks)} baby tasks against scores.parquet:")
    for task, (parquet_task, metric) in sorted(resolution.items()):
        print(f"  {task:55s} -> {parquet_task:35s} (metric={metric})")
    print(f"\nNot present in scores.parquet, dropped ({len(unresolved)}):")
    for task in sorted(unresolved):
        print(f"  {task}")

    baby_wide = build_baby_wide(baby, resolution)
    parquet_wide = build_parquet_wide(scores, resolution)

    benchmark_cols = sorted(resolution.keys())
    combined = pd.concat([baby_wide, parquet_wide], axis=0)
    combined = combined.reindex(columns=benchmark_cols)
    combined = combined[~combined.index.duplicated(keep="first")]

    combined["average"] = combined[benchmark_cols].mean(axis=1, skipna=True)
    combined = combined.sort_values("average", ascending=False)

    combined.to_csv(OUTPUT_CSV)
    print(f"\nWrote {combined.shape[0]} methods x {len(benchmark_cols)} benchmarks to {OUTPUT_CSV}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print()
    print(combined.to_string())


if __name__ == "__main__":
    sys.exit(main())
