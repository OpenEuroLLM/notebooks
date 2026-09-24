# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Turn the collected LUMI eval CSV into the results.csv schema.

Reads data/eval_results_all_metrics.csv (produced by `oellm-eval collect
--fetch_all_metrics`) and writes one data/prelude-8T*.csv per checkpoint, using
the same columns, metric choices and CRLF line endings as results.csv so the
files drop straight into the existing analysis pipeline.

Usage: uv run export_results.py
"""
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SRC = os.path.join(DATA, "eval_results_all_metrics.csv")
PRELUDE = "openeurollm/prelude-checkpoints"

# model_name in the collected CSV -> (output file, data, iter)
MODELS = {
    "/scratch/project_465002530/davisali/models/prelude-iter_0953312":
        ("prelude-8T.csv", PRELUDE, "iter_0953312"),
    "/scratch/project_465002530/davisali/models/prelude-anneal300b_iter_0989075":
        ("prelude-8T-anneal300b.csv", PRELUDE, "iter_0989075"),
    "birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b":
        ("prelude-8T-anneal300b-ctxext.csv",
         "birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b", ""),
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
}
COLS = ["model_path", "size", "data", "lr", "gbsz", "beta2", "seed",
        "schedule", "tokens_B", "iter", "benchmark", "task", "n_shot",
        "metric", "score"]


def main():
    src = pd.read_csv(SRC)
    for model, (fname, data, it) in MODELS.items():
        df = src[src.model_name == model]
        if df.empty:
            raise SystemExit(f"no rows for {model} in {SRC}")
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


if __name__ == "__main__":
    main()
