# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Turn the collected LUMI eval CSV into the results.csv schema.

Reads data/eval_results_all_metrics.csv and
data/eval_results_baselines_all_metrics.csv (produced by `oellm-eval collect
--fetch_all_metrics`) and writes one data/<model>.csv per model, using
the same columns, metric choices and CRLF line endings as results.csv so the
files drop straight into the existing analysis pipeline.

Usage: uv run export_results.py
"""
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SRCS = [os.path.join(DATA, "eval_results_all_metrics.csv"),
        os.path.join(DATA, "eval_results_baselines_all_metrics.csv"),
        # Second context-extension run (256k, rope theta 20M, continued from
        # the 300BT anneal), exported from LUMI by luomajou. Already in the
        # all-metrics shape, so it concatenates straight onto the rest.
        os.path.join(DATA, "export_256k-rope20m-cont_iter0500_allmetrics.csv")]
PRELUDE = "openeurollm/prelude-checkpoints"

# model_name in the collected CSV -> (output file, data, iter)
MODELS = {
    "/scratch/project_465002530/davisali/models/prelude-iter_0953312":
        ("prelude-8T.csv", PRELUDE, "iter_0953312"),
    "/scratch/project_465002530/davisali/models/prelude-anneal300b_iter_0989075":
        ("prelude-8T-anneal300b.csv", PRELUDE, "iter_0989075"),
    "/scratch/project_465002530/users/luomajou/oellm-long-ctx/output-prelude-anneal300b-256k-rope20m-cont/hf-checkpoints/prelude-anneal300b-256k-rope20m-cont/iter_0500":
        ("prelude-8T-anneal300b-256k-rope20m.csv",
         "openeurollm/prelude-anneal300b-256k-rope20m-cont", "iter_0500"),
    "birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b":
        ("prelude-8T-anneal300b-ctxext.csv",
         "birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b", ""),
    # Baselines missing from results.csv, rerun in the same LUMI setup.
    "marin-community/marin-8b-base":
        ("marin-8b.csv", "marin-community/marin-8b-base", ""),
    "utter-project/EuroLLM-9B":
        ("eurollm-9b.csv", "utter-project/EuroLLM-9B", ""),
    # Our run of the Olmo 3 HF release; plot_table.py uses it instead of the
    # results.csv Olmo 3 rows.
    "allenai/Olmo-3-1025-7B":
        ("olmo3-7b-ownrun.csv", "allenai/Olmo-3-1025-7B", ""),
}


# task prefix -> (benchmark label, metric), matching results.csv conventions.
# SIB-200 uses acc, not the acc_norm that `collect` picks as its primary
# metric: acc_norm is degenerate there (exactly 0.25 on every language for
# every model), while acc is real.
MAP = {
    "arc_challenge_mt":      ("ARC Challenge_mt", "acc_norm"),
    "belebele":              ("BeleBele",         "acc"),
    "global_mgsm":           ("Global MGSM",      "exact_match"),
    "global_mmlu_full":      ("GlobalMMLU",       "acc"),
    "include_base_44":       ("INCLUDE",          "acc"),
    "opensubtitles_multi40": ("OpenSubtitles",    "bleu"),
    "sib200":                ("SIB-200",          "acc"),
    "xcsqa":                 ("Xcsqa",            "acc_norm"),
    "multiblimp":            ("MultiBlimp",       "acc_norm"),
    "polymath":              ("PolyMath",         "exact_match"),
    # trailing underscore: the English `hellaswag` task is not xHellaswag
    "hellaswag_":            ("xHellaswag",       "acc_norm"),
    "global_piqa_completions": ("global PIQA",    "acc_norm"),
    "global_piqa_prompted":  ("global PIQA",      "exact_match"),
}
# English dclm-core-22 task -> the metric eval_results.csv records as primary.
EN_METRIC = {
    "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "boolq": "acc",
    "commonsense_qa": "acc", "copa": "acc", "hellaswag": "acc_norm",
    "openbookqa": "acc_norm", "piqa": "acc_norm", "winogrande": "acc",
    "wsc273": "acc", "lambada_openai": "acc", "coqa": "f1",
    "bigbench_language_identification_multiple_choice": "acc",
    "bigbench_qa_wikidata_generate_until": "exact_match",
    "bigbench_dyck_languages_generate_until": "exact_match",
    "bigbench_operators_generate_until": "exact_match",
    "bigbench_repeat_copy_logic_generate_until": "exact_match",
    "bigbench_cs_algorithms_generate_until": "exact_match",
    "jeopardy": "exact_match",
}
# (all-metrics source, primary-metric output) for models that arrive outside
# the `collect` run, which writes both shapes itself.
EN_EXPORTS = [("export_256k-rope20m-cont_iter0500_allmetrics.csv",
               "eval_results_256k-rope20m-cont.csv")]

COLS = ["model_path", "size", "data", "lr", "gbsz", "beta2", "seed",
        "schedule", "tokens_B", "iter", "benchmark", "task", "n_shot",
        "metric", "score"]


def main():
    src = pd.concat([pd.read_csv(p) for p in SRCS], ignore_index=True)
    for model, (fname, data, it) in MODELS.items():
        df = src[src.model_name == model]
        if df.empty:
            raise SystemExit(f"no rows for {model} in {SRCS}")
        rows = []
        for prefix, (bench, metric) in MAP.items():
            sel = df[df.task.str.startswith(prefix) & (df.metric_name == metric)]
            for _, r in sel.iterrows():
                rows.append({
                    "model_path": f"{data},revision={it}" if it else data,
                    "size": "", "data": data, "lr": "", "gbsz": "", "beta2": "",
                    "seed": "", "schedule": "", "tokens_B": "", "iter": it,
                    "benchmark": bench, "task": r["task"],
                    "n_shot": int(r["n_shot"]), "metric": metric,
                    "score": round(float(r["performance"]), 5),
                })
        out = pd.DataFrame(rows, columns=COLS).sort_values(["benchmark", "task"])
        out.to_csv(os.path.join(DATA, fname), index=False, lineterminator="\r\n")
        print(f"{fname:38s} {len(out):4d} rows  iter={it or '(none)'}")

    for src_name, out_name in EN_EXPORTS:
        d = pd.read_csv(os.path.join(DATA, src_name))
        keep = d.apply(lambda r: EN_METRIC.get(r["task"]) == r["metric_name"],
                       axis=1)
        d[keep].to_csv(os.path.join(DATA, out_name), index=False)
        print(f"{out_name:38s} {int(keep.sum()):4d} rows  (English primary metric)")


if __name__ == "__main__":
    main()
