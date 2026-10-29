# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "pyyaml", "huggingface_hub"]
# ///
"""Which Prelude checkpoint x task evaluations are missing from the HF eval dataset?

Checkpoints: the branches of openeurollm/prelude starting with `iter_` or
`anneal2t` that are on the evaluation schedule. Not every branch is evaluated, so
the schedule is inferred per branch family from the multilingual evals already in
the dataset: the most common gap between evaluated steps (24000 for `iter_*`).
A branch is scheduled when its step is a multiple of that gap, it is already
evaluated, or it is the family's last branch. Branches in --exclude (default: iter_0953312, the 8T checkpoint,
evaluated separately) are dropped from the schedule. A family with no evals yet (anneal2t)
borrows the spacing of the other anneal family; override with --every PREFIX=STEPS.
The same schedule is used for DCLM. Expected tasks are what the current OELLM suites define, expanded
with oellm-evals' own task-group code (task + n_shot):

  multilingual   multilingual-oellm-eu   -> multilingual_scores.parquet
  DCLM           dclm-core-22            -> english_lm_eval_scores.parquet

An evaluation counts as present when the dataset has a row for
(model=openeurollm/prelude-checkpoints, revision=<branch>, task, num_fewshot);
the metric is ignored. Prints a summary and a checkpoint x task table per suite
and writes missing_evals_multiling.csv / missing_evals_dclm.csv (one row per
missing checkpoint/task).

Usage: uv run show_missing.py [--every PREFIX=STEPS ...] [--oellm-repo PATH]
                              [--no-compress] [-o OUT_DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
MODEL_REPO = "openeurollm/prelude"
DATASET = "openeurollm/prelude-base-eval-scores"
DATASET_MODEL = "openeurollm/prelude-checkpoints"
BRANCH_PREFIXES = ("iter_", "anneal2t")
# Evaluated by hand in a different, ad hoc setup, so not tracked here.
DEFAULT_EXCLUDE = ["iter_0953312"]

# suite name -> (task group, dataset file, output csv)
SUITES = {
    "multilingual": ("multilingual-oellm-eu", "multilingual_scores.parquet",
                     "missing_evals_multiling.csv"),
    "DCLM": ("dclm-core-22", "english_lm_eval_scores.parquet",
             "missing_evals_dclm.csv"),
}


def checkpoints() -> list[str]:
    """Branch names of MODEL_REPO to evaluate: iter_* by step, then anneal2t*."""
    names = [b.name for b in HfApi().list_repo_refs(MODEL_REPO).branches
             if b.name.startswith(BRANCH_PREFIXES)]
    return sorted(names, key=lambda n: (not n.startswith("iter_"), n))


def family(branch: str) -> str:
    return branch.rsplit("iter_", 1)[0] + "iter_"


def step(branch: str) -> int:
    return int(branch.rsplit("iter_", 1)[1])


def schedule(branches: list[str], evaluated: set[str], overrides: dict[str, int],
             exclude: set[str]) -> list[str]:
    """The branches on the evaluation schedule (see module docstring)."""
    steps = {}
    for rev in evaluated:
        steps.setdefault(family(rev), set()).add(step(rev))
    inferred = {f: int(pd.Series(sorted(v)).diff().dropna().mode().max())
                for f, v in steps.items() if len(v) >= 3}

    def spacing(fam: str) -> tuple[int, str]:
        for prefix, n in overrides.items():
            if fam.startswith(prefix):
                return n, "--every"
        if fam in inferred:
            return inferred[fam], f"{len(steps[fam])} evaluated branches"
        analog = next((f for f in inferred if f.startswith("anneal")), None) \
            if fam.startswith("anneal") else None
        if analog is None:
            sys.exit(f"no evals to infer the spacing of {fam}*; pass --every {fam}=STEPS")
        return inferred[analog], f"assumed from {analog}*, no evals yet"

    keep, print_rows = [], []
    for fam in sorted({family(b) for b in branches}, key=lambda f: f != "iter_"):
        fam_b = [b for b in branches if family(b) == fam]
        every, why = spacing(fam)
        last = max(fam_b, key=step)
        # the family's last branch is picked before excluding, so dropping the
        # 8T checkpoint does not promote iter_0952800 to "last"
        chosen = [b for b in fam_b if step(b) % every == 0 or b in evaluated or b == last]
        chosen = [b for b in chosen if b not in exclude]
        keep += chosen
        print_rows.append(f"  {fam + '*':44s} every {every:6d} steps ({why}): "
                          f"{len(chosen)} of {len(fam_b)} branches scheduled")
    print("evaluation schedule inferred from the multilingual evals in the dataset:")
    print("\n".join(print_rows))
    return sorted(keep, key=lambda n: (not n.startswith("iter_"), n))


def expected_tasks(oellm_repo: Path, group: str) -> pd.DataFrame:
    sys.path.insert(0, str(oellm_repo))
    from oellm.task_groups import _expand_task_groups  # needs only yaml
    return pd.DataFrame(
        [(r.task, r.n_shot) for r in _expand_task_groups([group])],
        columns=["task", "n_shot"]).drop_duplicates()


def held(fname: str) -> pd.DataFrame:
    """Prelude rows of one dataset file: revision, task, parent_task, num_fewshot."""
    path = hf_hub_download(DATASET, fname, repo_type="dataset")
    d = pd.read_parquet(path, columns=["model", "revision", "task", "parent_task",
                                       "num_fewshot", "score"])
    return d[(d.model == DATASET_MODEL) & d.score.notna()]


def find_missing(ckpts: list[str], expected: pd.DataFrame,
                 have: pd.DataFrame) -> pd.DataFrame:
    """One row per (checkpoint, task, n_shot) the dataset lacks."""
    grid = pd.MultiIndex.from_product(
        [ckpts, range(len(expected))], names=["checkpoint", "i"]).to_frame(index=False)
    grid = grid.join(expected.reset_index(drop=True), on="i").drop(columns="i")
    present = pd.MultiIndex.from_frame(
        have[["revision", "task", "num_fewshot"]].astype({"num_fewshot": int}))
    key = pd.MultiIndex.from_frame(grid[["checkpoint", "task", "n_shot"]])
    return grid[~key.isin(present)].reset_index(drop=True)


def label_benchmarks(missing: pd.DataFrame, have: pd.DataFrame, many_shots: set[str]):
    """Column label per task: the dataset's benchmark group when it has one
    (multilingual), otherwise the task name, with n_shot where a task has several."""
    parent = have.drop_duplicates("task").set_index("task").parent_task
    out = missing.copy()
    out["benchmark"] = [parent.get(t, t) for t in out.task]
    out["column"] = [f"{t} ({n})" if t in many_shots else t
                     for t, n in zip(out.task, out.n_shot)]
    return out


def compress(table: pd.DataFrame) -> pd.DataFrame:
    """Merge runs of consecutive checkpoints with an identical row into one."""
    same = (table != table.shift()).any(axis=1).cumsum()
    rows = {}
    for _, grp in table.groupby(same, sort=False):
        first, last = grp.index[0], grp.index[-1]
        name = first if first == last else f"{first} .. {last} (n={len(grp)})"
        rows[name] = grp.iloc[0]
    return pd.DataFrame(rows).T


def report(name: str, ckpts: list[str], expected: pd.DataFrame, have: pd.DataFrame,
           out_dir: Path, out_csv: str, no_compress: bool) -> None:
    n_exp = len(expected)
    missing = find_missing(ckpts, expected, have)
    shots = expected.groupby("task").n_shot.nunique()
    missing = label_benchmarks(missing, have, set(shots[shots > 1].index))
    # task present, but only at another n_shot (e.g. Flores-200 0-shot vs 4-shot)
    other = set(zip(have.revision, have.task))
    missing["has_other_nshot"] = [(c, t) in other for c, t in zip(missing.checkpoint, missing.task)]
    # benchmark per expected task, for the denominators
    exp = label_benchmarks(expected.assign(checkpoint=""), have, set(shots[shots > 1].index))
    never = sorted(set(expected.task) - set(have.task))

    print(f"\n{'=' * 78}\n{name}: {n_exp} expected (task, n_shot) per checkpoint, "
          f"{len(ckpts)} checkpoints -> {len(ckpts) * n_exp} evaluations\n{'=' * 78}")
    n_miss = len(missing)
    print(f"missing: {n_miss} evaluations ({n_miss / (len(ckpts) * n_exp):.1%}); "
          f"{missing.checkpoint.nunique()} of {len(ckpts)} checkpoints lack at least one, "
          f"{len(ckpts) - missing.checkpoint.nunique()} are complete")
    for fam in BRANCH_PREFIXES:
        c = [x for x in ckpts if x.startswith(fam)]
        m = missing[missing.checkpoint.isin(c)]
        print(f"  {fam + '*':10s} {len(c):4d} checkpoints, {m.checkpoint.nunique():4d} incomplete, "
              f"{len(m):7d} evaluations missing")
    n_other = int(missing.has_other_nshot.sum())
    if n_other:
        print(f"  of these, {n_other} are tasks the dataset has at a different n_shot "
              f"({', '.join(sorted(missing[missing.has_other_nshot].benchmark.unique()))})")
    if never:
        print(f"  note: {len(never)} expected tasks are in the dataset for NO checkpoint "
              f"(failed everywhere, or named differently): {', '.join(never[:6])}"
              + (" ..." if len(never) > 6 else ""))

    by_bench = exp.groupby("benchmark", sort=False).size().to_frame("tasks_expected")
    by_bench["checkpoints_missing"] = missing.groupby("benchmark").checkpoint.nunique()
    by_bench["evals_missing"] = missing.groupby("benchmark").size()
    by_bench["of_which_other_nshot"] = missing.groupby("benchmark").has_other_nshot.sum()
    by_bench = by_bench.fillna(0).astype(int)
    print("\nper benchmark (checkpoints_missing = checkpoints lacking >= 1 of its tasks):")
    print(by_bench.to_string())

    # multilingual has ~400 task columns, so its columns are benchmarks (missing
    # tasks per cell); DCLM's 21 tasks fit as columns (1 = missing)
    by_task = name != "multilingual"
    cols = list(dict.fromkeys(exp.column if by_task else exp.benchmark))
    counts = missing.pivot_table(index="checkpoint", columns="column" if by_task else "benchmark",
                                 values="task", aggfunc="size", fill_value=0)
    counts = counts.reindex(index=ckpts, columns=cols, fill_value=0)   # complete ones stay, as '.' 
    what = "1 = missing" if by_task else "missing tasks per benchmark"
    shown = counts if no_compress else compress(counts)
    print(f"\ncheckpoint x {'task' if by_task else 'benchmark'} "
          f"({what}; '.' = complete"
          + ("" if no_compress else "; identical consecutive rows merged, use --no-compress for all") + ")")
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.max_rows", None):
        print(shown.astype(int).astype(str).replace("0", ".").to_string())

    path = out_dir / out_csv
    missing[["checkpoint", "benchmark", "task", "n_shot", "has_other_nshot"]].to_csv(path, index=False)
    print(f"\nwrote {path} ({n_miss} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oellm-repo", type=Path, default=HERE.parents[1] / "oellm-evals",
                    help="oellm-evals checkout, for the task-group definitions")
    ap.add_argument("--every", action="append", default=[], metavar="PREFIX=STEPS",
                    help="evaluation spacing for branches starting with PREFIX "
                         "(default: inferred), e.g. --every anneal2t=4800")
    ap.add_argument("--exclude", action="append", default=None, metavar="BRANCH",
                    help=f"branch to leave out (repeatable; default: {' '.join(DEFAULT_EXCLUDE)})")
    ap.add_argument("--no-compress", action="store_true",
                    help="print one table row per checkpoint")
    ap.add_argument("-o", "--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    branches = checkpoints()
    print(f"{len(branches)} branches on {MODEL_REPO}: "
          + ", ".join(f"{sum(c.startswith(p) for c in branches)} {p}*" for p in BRANCH_PREFIXES))
    overrides = {k: int(v) for k, v in (e.split("=") for e in args.every)}
    exclude = set(DEFAULT_EXCLUDE if args.exclude is None else args.exclude)
    ckpts = schedule(branches, set(held(SUITES["multilingual"][1]).revision), overrides,
                     exclude)
    for name, (group, fname, out_csv) in SUITES.items():
        report(name, ckpts, expected_tasks(args.oellm_repo, group), held(fname),
               args.out_dir, out_csv, args.no_compress)


if __name__ == "__main__":
    main()
