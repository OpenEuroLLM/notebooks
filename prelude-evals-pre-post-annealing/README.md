# Prelude 8T — pre/post annealing evaluation

Compares four Prelude checkpoints against external baselines on English and
multilingual benchmarks:

| label | checkpoint |
|---|---|
| Prelude 8T | `prelude-iter_0953312` (7997 B tokens) |
| Prelude 8T + 300BT annealing | `prelude-anneal300b_iter_0989075` (8297 B tokens) |
| Prelude 8T + 300BT ann. + 256k rope20m cont. | `prelude-anneal300b-256k-rope20m-cont/iter_0500` (luomajou's run, LUMI) |
| Prelude 8T + 300BT ann. + context ext. | `birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b` |

The 256k-rope20m row was evaluated outside our `collect` run and arrives as
`data/export_256k-rope20m-cont_iter0500_allmetrics.csv`, copied from
`/scratch/project_465002530/users/luomajou/` on LUMI. It covers the same task
suite, so it does not narrow `common_tasks_only`; `export_results.py` also
derives `data/eval_results_256k-rope20m-cont.csv` for the English table.

Baselines: Datamix 9b, Prelude 4T, Olmo 3 7B, Apertus 8B (v1 base, 15T).

## Quick start

Everything needed is committed under `data/`, so the plots reproduce with no
setup beyond [uv](https://docs.astral.sh/uv/):

```bash
uv run plot_table.py   # both tables + plot_table.png, plot_table_english.png
```

Prints the Multilingual and English tables to stdout and writes one PNG for
each. Flags: `-o <file>` / `--out-english <file>` to change the output paths,
`--no-baselines` to show only the three Prelude checkpoints.

## How the results were produced

Steps 1–4 only need repeating for a **new** checkpoint; `data/` already holds
the output of every step.

### 1. Run the evaluations on LUMI

```bash
export ENVPATH=/scratch/project_465002530/users/davisali/oellm-venv/
export MODELS=/scratch/project_465002530/davisali/models

uv run oellm-eval schedule \
    --models "$MODELS/prelude-iter_0953312,$MODELS/prelude-anneal300b_iter_0989075,birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b" \
    --task_groups "dclm-core-22,multilingual-oellm-eu" \
    --n_shot 0 \
    --venv_path $ENVPATH
```

`dclm-core-22` gives the English suite, `multilingual-oellm-eu` the
multilingual one. Results land in
`/scratch/project_465002530/oellm-cli-shared-evals/$USER/<timestamp>/results/`.

The venv **must** have the ROCm build of torch — LUMI is AMD, and a default
`pip install torch` pulls the CUDA build, which silently falls back to CPU and
makes every job hit the wall clock. Use `setup_env_lumi.sh` in the `oellm-eval`
repo, and sanity-check with:

```bash
srun -A project_465002530 -p small-g --gres=gpu:1 -t 5 \
  $ENVPATH/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Budget roughly **75 GPU-hours per checkpoint** (7 h English + 68 h
multilingual), or about double that for the 256k-context model.

### 2. Pull the results

```bash
rsync -a --include='*/' --include='*.json' --include='jobs.csv' --exclude='*' \
  lumi:/scratch/project_465002530/oellm-cli-shared-evals/davisali/<timestamp> \
  ./pulled/
```

The `--include`/`--exclude` filters skip `slurm_logs/`, which is hundreds of MB.

### 3. Collect the JSONs into a CSV

From the `oellm-eval` checkout:

```bash
oellm-eval collect --results_dir ./pulled \
    --output_csv eval_results.csv                       # primary metric per task
oellm-eval collect --results_dir ./pulled --fetch_all_metrics true \
    --output_csv eval_results_all_metrics.csv           # every metric
```

Copy both into `data/`. Add `--check` to get a `*_missing.csv` listing
evaluations that produced no result.

### 4. Export to the `results.csv` schema

```bash
uv run export_results.py
```

Reads `data/eval_results_all_metrics.csv` and writes one
`data/prelude-8T*.csv` per checkpoint, matching the column layout, per-benchmark
metric choice and CRLF line endings of `results.csv`.

### 5. Plot

See **Quick start**.

## Annealing trajectory (intermediate checkpoints)

`data/anneal300b-trajectory/` holds 15 intermediate checkpoints of the 300BT
anneal (`openeurollm/prelude` branches `anneal300b_iter_0955200` ...
`0988800`, every 2400 iters), evaluated by tudormateiu with
`multilingual-oellm-eu` only - **no English dclm-core-22**. Every checkpoint
has the identical 579 rows (same tasks, shots and metrics, no gaps), copied
unchanged from
`/scratch/project_465002530/users/tudormateiu/prelude_evaluations/evals/results/`
(`scp 'lumi:<that dir>/*.csv' data/anneal300b-trajectory/`).

```bash
uv run plot_anneal_trajectory.py   # plot_anneal_trajectory.png + tables
```

One panel per benchmark, with our Prelude 8T (iter 953312) and final anneal
(iter 989075) runs overlaid. Same metric mapping and language filter as
`plot_table.py`; each benchmark is averaged over the (task, n_shot) pairs all
17 points share (279). Iter 988800 (Tudor's run) and 989075 (ours) agree to
within 0.001 on AVG, so the two eval setups are consistent. SIB-200 is very
noisy between neighbouring checkpoints (0.32-0.49), so read its panel, and the
AVG dips it causes, with care.

Not used: Tudor's runs also include `flores200` chrF++ (70 tasks), `xcopa` and
`mgsm_native`, which the 8T / anneal runs here lack, so they have no anchor.

## Files

```
plot_table.py           multilingual + English tables and charts
plot_anneal_trajectory.py   multilingual scores along the anneal
export_results.py       step 4
data/
  prelude-8T*.csv                     the three checkpoints, results.csv schema
  anneal300b-trajectory/              15 intermediate anneal checkpoints (Tudor)
  eval_results.csv                    collected scores, primary metric
  eval_results_all_metrics.csv        collected scores, all metrics
  results.csv                         multilingual baselines
  results-apertus.csv                 Apertus multilingual baseline
  compare_prelude_ellamind_suite.csv  English baselines (wide format)
```

`results.csv`, `results-apertus.csv` and `compare_prelude_ellamind_suite.csv`
are copies of the shared campaign files in `../prelude-evals-wip/`. Refresh
`results.csv` with:

```bash
scp lumi:/scratch/project_465002530/users/haider/ML_Evals/campaigns/oellm_public_v2/results.csv data/
```

## Caveats baked into the scripts

Each is a named constant at the top of the relevant script, with a comment.

- **`Global MGSM` excluded** (`plot_table.py`). 0-shot MGSM measures whether a
  model emits the `Answer:` format, not whether it can do the maths. Every base
  model scores 0.01–0.04, including Apertus 8B v1 at 0.018; only the
  instruction-tuned Apertus v1.5 scored 0.44. Re-run at 5- or 8-shot to make it
  meaningful.
- **`PolyMath` excluded** (`plot_table.py`) for the same reason: 0-shot
  generative maths, every model 0.00–0.12.
- **`lambada_openai` excluded** (`plot_table.py`). The column in
  `compare_prelude_ellamind_suite.csv` is broken for non-prelude models —
  `llama3_1_8b` 0.479 and `llama3_1_70b` 0.503, against ~0.75 in standard
  lm-eval — while the prelude rows (~0.70) look right.
- **GlobalMMLU is the full subject set** for every model. `oellm-eval collect`
  before `ba1ff8c` (#109) kept only the first group in each result JSON, which
  for GlobalMMLU is the `_stem` subgroup — collect with a later version.
- **SIB-200 uses `acc`**, not the `acc_norm` that `oellm-eval collect` picks:
  `acc_norm` is exactly 0.25 on every language for every model.
- **Olmo 3 7B is our own run** of the HF release (`allenai/Olmo-3-1025-7B`) in
  both tables, not the `results.csv` / suite checkpoints.
- **Datamix is marked `*` in the `--suite-english` table**: it is
  `9b_60_40@step900000` there (the suite has no `9b_80_20`). Apertus and
  Prelude 4T match across both tables.
- Every benchmark is averaged over **only the languages/tasks all plotted
  models have**, so groups are like-for-like. Counts appear in the chart
  subtitle.
- `AVG` excludes OpenSubtitles BLEU, which is on a 0–100 scale.
