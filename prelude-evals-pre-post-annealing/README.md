# Prelude 8T — pre/post annealing evaluation

Compares three Prelude checkpoints against external baselines on English and
multilingual benchmarks:

| label | checkpoint |
|---|---|
| Prelude 8T | `prelude-iter_0953312` (7997 B tokens) |
| Prelude 8T + 300BT annealing | `prelude-anneal300b_iter_0989075` (8297 B tokens) |
| Prelude 8T + 300BT ann. + context ext. | `birgermoell/oellm-9b-256k-theta64m-prelude-anneal300b` |

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

## Files

```
plot_table.py           multilingual + English tables and charts
export_results.py       step 4
data/
  prelude-8T*.csv                     the three checkpoints, results.csv schema
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
- **`lambada_openai` excluded** (`plot_table.py`). The column in
  `compare_prelude_ellamind_suite.csv` is broken for non-prelude models —
  `llama3_1_8b` 0.479 and `llama3_1_70b` 0.503, against ~0.75 in standard
  lm-eval — while the prelude rows (~0.70) look right.
- **`GlobalMMLU*`** is matched across sources by *language*. The Prelude rows
  are the STEM subset (`global_mmlu_full_<lang>_stem`), the baselines the full
  subject set, so the two sides answer different questions. Treat that column
  as indicative.
- **SIB-200 uses `acc`**, not the `acc_norm` that `oellm-eval collect` picks:
  `acc_norm` is exactly 0.25 on every language for every model.
- **Baselines marked `*` in the English table** differ from the multilingual
  one — Datamix is `9b_60_40@step900000` (the English suite has no `9b_80_20`)
  and Olmo 3 is `@step1473419`. Apertus and Prelude 4T match across both.
- Every benchmark is averaged over **only the languages/tasks all plotted
  models have**, so groups are like-for-like. Counts appear in the chart
  subtitle.
- `AVG` excludes OpenSubtitles BLEU, which is on a 0–100 scale.
