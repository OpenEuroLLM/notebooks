# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub",
#   "matplotlib",
#   "pandas",
#   "pyarrow",
# ]
# ///
"""Final-checkpoint results of Prelude vs baselines on English, Multilingual,
German, Norwegian and Finnish.

For each category: a markdown table (models x benchmark groups, plus AVG, best
value of each column in bold) and a grouped bar plot
figures/plot_results_<category>_all.png (English and Multilingual also
_decontaminated_only / _non_decontaminated_only). The tables and figures are
also written to README.md (decontaminated_only for English and Multilingual,
all for German, Norwegian and Finnish), whose text above "# Results" is kept and
everything from it on regenerated. Each
model is shown at a single checkpoint: its released model when there is one,
else its latest checkpoint.

Sources (HF datasets):
  English / German  openeurollm/prelude-base-eval-scores scores.parquet
                    (Prelude) + ellamind/eval-scores-ref (baselines). Tasks
                    are averaged within their task group of the suite's
                    subtask_tree, so e.g. the bigbench subtasks count once.
  Multilingual      multilingual_scores.parquet, OpenEuroLLM target languages
                    only, EXCLUDE_BENCHMARKS dropped.
  Norwegian         noreval_scores.parquet (LTG NorEval dashboard results),
                    0-shot, each task's main metric (prompt variants already
                    aggregated upstream).
  Finnish           finnish_lm_eval_results/ (FIN-bench v2 lm-eval JSONs),
                    5-shot, each benchmark averaged over its 5 prompts.
Tasks are kept only when every model of the category has them. Translation
metrics (BLEU / chrF, 0-100) go to a separate panel and are left out of AVG.

Usage: uv run plot_results.py
"""
import datetime
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
README = os.path.join(HERE, "README.md")

REPO_ID = "openeurollm/prelude-base-eval-scores"
ELLAMIND_REPO_ID = "ellamind/eval-scores-ref"
# Suite-level aggregates: a leaf directly under one of these is its own group.
SUITES = {f"{l}_base_{s}" for l in ("eng", "deu") for s in ("main", "full", "easy")}

# 0-shot generative maths: base models score ~0, so these measure answer
# formatting rather than maths (see plot_table.py
# in prelude-evals-pre-post-annealing).
EXCLUDE_BENCHMARKS = {"Global MGSM", "PolyMath"}

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


# --- multilingual: tokens from revision names --------------------------------
# tokens per optimiser step = global batch size * sequence length
TOKENS_PER_STEP = {
    "openeurollm/prelude-checkpoints": 4096 * 2048,
    "openeurollm/datamix-9b-80-20": 2048 * 2048,
    "allenai/Olmo-3-1025-7B": 4_194_304,
}
STEP_RE = re.compile(r"(?:iter_|stage1-step)(\d+)")
APERTUS_TOKENS_RE = re.compile(r"tokens(\d+(?:\.\d+)?)([BT])")


def tokens_trained(model: str, revision: str | None) -> float | None:
    """Tokens seen at a checkpoint, from its revision name; None for released
    models without checkpoints."""
    if revision is None:
        return None
    if model == "swiss-ai/Apertus-8B-2509":
        m = APERTUS_TOKENS_RE.search(revision)
        return float(m.group(1)) * (1e12 if m.group(2) == "T" else 1e9)
    m = STEP_RE.search(revision)
    return int(m.group(1)) * TOKENS_PER_STEP[model]


def leaf_groups(df: pd.DataFrame) -> dict[str, str]:
    """Leaf task -> the task group whose subtask_tree lists it directly
    (smallest such group when several do). Suite aggregates are skipped."""
    parent = {}
    groups = df[(df["task_type"] == "task_group") & df["subtask_tree"].notna()
                & ~df["task"].isin(SUITES)].drop_duplicates("task")
    for group, tree in zip(groups["task"], groups["subtask_tree"]):
        children = json.loads(tree).get(group, [])
        for child in children:
            if child not in parent or len(children) < parent[child][1]:
                parent[child] = (group, len(children))
    return {leaf: g for leaf, (g, _) in parent.items()}


# Per task, the first of these the task reports is used.
METRIC_PREFERENCE = ["acc_norm", "acc", "exact_match", "em", "errant"]
TRANSLATION_METRICS = ["chrf", "chrf++", "bleu"]   # side panel, not in AVG

# series id -> name. Rows / bars: open-weights, then fully open models (each by
# AVG, weakest first), then Prelude.
ENG_DEU_MODELS = {
    "prelude_anneal2t": "Prelude 9B + 2T anneal",
    "eurollm_9B_2512": "EuroLLM 9B 2512",
    "apertus_8b": "Apertus 8B",
    "olmo_3_1025_7b": "Olmo 3 7B",
    "marin_8b": "Marin 8B",
    #"qwen3_8b": "Qwen3 8B",    # only including one Qwen (3.5 9B)
    "qwen35_9b": "Qwen3.5 9B",
    "llama3_1_8b": "Llama 3.1 8B",
}
MULTILINGUAL_MODELS = {
    "openeurollm/prelude-checkpoints": "Prelude 9B",
    "utter-project/EuroLLM-9B": "EuroLLM 9B",
    "swiss-ai/Apertus-8B-2509": "Apertus 8B",
    "allenai/Olmo-3-1025-7B": "Olmo 3 7B",
    "marin-community/marin-8b-base": "Marin 8B",
    #"Qwen/Qwen3-8B-Base": "Qwen3 8B",    # only including one Qwen (3.5 9B)
    "Qwen/Qwen3.5-9B-Base": "Qwen3.5 9B",
    "meta-llama/Llama-3.1-8B": "Llama 3.1 8B",
}
# NorEval covers ~80 models; the baselines are picked automatically as those
# also used for English / German / Multilingual (see norwegian_models).
NOREVAL_PRELUDE = {"openeurollm/prelude-base": "Prelude 9B + 2T anneal"}
NOREVAL_SHOTS = 0
# Shown names where they differ from the internal one (both Prelude rows are
# "Prelude-base 9B", told apart by the tokens in the label).
DISPLAY = {"Prelude 9B": "Prelude-base 9B",
           "Prelude 9B + 2T anneal": "Prelude-base 9B"}
# NorEval main metrics on a 0-1 scale; BLEU only for the translation tasks.
# Summarisation / rewriting / truthfulqa-gen (ROUGE-L, BLEU) are dropped.
NOREVAL_METRICS = ["acc", "f1", "em_first", "errant_f05"]

# Finnish: FIN-bench v2 lm-eval results (finnish_lm_eval_results/<model>/*.json),
# 5-shot, each benchmark averaged over its 5 prompt variants (group_* rows).
# Keyed by the lm-eval `pretrained` (and revision for Prelude). Left out:
# Llama-Poro-2-8B-Instruct (instruction tuned), OLMo 2 7B (superseded by
# Olmo 3) and the smaller TurkuNLP gpt3-finnish models.
FINNISH_DIR = "finnish_lm_eval_results"
FINNISH_MODELS = {
    "openeurollm/prelude@anneal2t_multilingual1_dominant1_iter_1192093":
        "Prelude 9B + 2T anneal",
    "utter-project/EuroLLM-9B": "EuroLLM 9B",
    "swiss-ai/Apertus-8B-2509": "Apertus 8B",
    "allenai/Olmo-3-1025-7B": "Olmo 3 7B",
    "marin-community/marin-8b-base": "Marin 8B",
    "BSC-LT/salamandra-7b": "Salamandra 7B",
    "TurkuNLP/gpt3-finnish-8B": "FinGPT3 8B",
    "meta-llama/Llama-3.1-8B": "Llama 3.1 8B",
    "Qwen/Qwen3.5-9B-Base": "Qwen3.5 9B",
}
FINNISH_GROUPS = {   # group task -> (shown name, metric)
    "group_arc_challenge_fi_multi_mcf_fbv2": ("ARC Challenge FI", "acc_norm"),
    "group_belebele_multi_mcf_fbv2": ("Belebele FI", "acc_norm"),
    "group_finbench_multi_mcf_analogies_fbv2": ("FIN-bench Analogies", "acc_norm"),
    "group_finbench_multi_mcf_emotions_1k_fbv2": ("FIN-bench Emotions", "acc_norm"),
    "group_finbench_multi_mcf_general_knowledge_fbv2":
        ("FIN-bench General Knowledge", "acc_norm"),
    "group_finbench_multi_mcf_hhh_alignment_fbv2": ("FIN-bench HHH Alignment", "acc_norm"),
    "group_finbench_multi_mcf_similarities_abstraction_fbv2":
        ("FIN-bench Similarities", "acc_norm"),
    "group_goldenswag_multi_cf_fbv2": ("GoldenSwag FI", "acc_norm"),   # cloze
    "group_ogx_truthfulqax_multi_mc1_fbv2": ("TruthfulQA FI", "acc_norm"),
    "group_scandisent_fi_multi_mcf_fbv2": ("ScandiSent FI", "acc_norm"),
    "group_sib200_fi_multi_mcf_fbv2": ("SIB-200 FI", "acc_norm"),
    "group_squad_multi_gen_fbv2": ("SQuAD FI", "f1"),
}


# Lighten a color by given amount
def lighten(color, amount=0.5):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return tuple(c + (1 - c) * amount for c in rgb)


# One colour per model across all plots (models never shown together
# share one). Every Prelude checkpoint is the same blue with a
# dark-blue outline (8T and 10T never share a plot).
PRELUDE_BLUE = "#4c8fd6"
PRELUDE_EDGE = "#1d4f8c"   # darker shade of PRELUDE_BLUE for the outline
OPEN, WEIGHTS = "fully open", "open weights"
STYLES = {   # name -> (colour, style)
    "Prelude 9B": (PRELUDE_BLUE, OPEN),              # 8T, before the anneal
    "Prelude 9B + 2T anneal": (PRELUDE_BLUE, OPEN),  # 10T, final
    "Apertus 8B": ("#da291c", OPEN),    # Swiss flag red
    "Olmo 3 7B": ("#e5679f", OPEN),   # Olmo brand pink
    "Marin 8B": (lighten("#192026", 0.3), OPEN),    # Marin logo blue-gray
    # EuroLLM is Apache 2.0; counted as open (data not released)
    "EuroLLM 9B": (lighten("#ec5926", 0.3), OPEN),    # Utter logo orange
    "EuroLLM 9B 2512": (lighten("#ec5926", 0.3), OPEN),
    "Qwen3 8B": (lighten("#6950e8"), WEIGHTS),    # Qwen logo color
    "Qwen3.5 9B": (lighten("#6950e8"), WEIGHTS),    # Qwen logo color
    "Llama 3.1 8B": (lighten("#0081fb"), WEIGHTS),    # Meta logo blue
    # Apache 2.0, training data documented; counted as open like EuroLLM
    "Salamandra 7B": (lighten("#f2b705", 0.2), OPEN),   # salamander yellow
    "FinGPT3 8B": (lighten("#2a9d8f", 0.2), OPEN),      # TurkuNLP, Finnish only
}


# --- display names for benchmark groups ---------------------------------------
GROUP_NAMES = {
    "ARC Challenge_mt": "ARC Challenge MT",
    "BeleBele": "Belebele",
    "GlobalMMLU": "Global MMLU",
    "MultiBlimp": "MultiBLiMP",
    "Xcsqa": "XCSQA",
    "global PIQA": "Global PIQA",
    "xHellaswag": "xHellaSwag",
    "noreval_multiblimp": "MultiBLiMP SME",
    "heldout": "GPQA",   # English held-out group is only GPQA diamond + main
    "INCLUDE": "INCLUDE", "SIB-200": "SIB-200", "OpenSubtitles": "OpenSubtitles",
}
GROUP_NAMES.update({n: n for n, _ in FINNISH_GROUPS.values()})   # already styled
WORDS = {   # per "_"-separated word; unlisted words are capitalised
    "agieval": "AGIEval", "mc": "MC", "rc": "RC", "coqa": "CoQA",
    "deepmind": "DeepMind", "drop": "DROP", "gsm": "GSM", "all": "",
    "heldout": "Held-out", "mmlu": "MMLU", "mmmlu": "MMMLU", "filtered": "",
    "stem": "STEM", "nonstem": "Non-STEM", "naturalqs": "NaturalQs",
    "qa": "QA", "squad": "SQuAD", "glp": "GLP", "include": "INCLUDE",
    "ask": "ASK", "gec": "GEC", "ncb": "NCB", "nocola": "NoCoLA",
    "norbelebele": "NorBelebele", "norcommonsenseqa": "NorCommonsenseQA",
    "norec": "NoReC", "noridiom": "NorIdiom", "noropenbookqa": "NorOpenBookQA",
    "nofact": "no-fact", "norquad": "NorQuAD", "nortruthfulqa": "NorTruthfulQA",
    "nrk": "NRK", "slide": "SLIDE", "norsumm": "NorSumm",
    "nob": "NB", "nno": "NN", "eng": "EN", "sme": "SME",
}
LANG = {"nob": "NB", "nno": "NN", "eng": "EN", "sme": "SME"}
PAIR_RE = re.compile(r"^(tatoeba|norsumm)_([a-z]{3})_([a-z]{3})(?:_translation)?$")


def pretty_group(group: str) -> str:
    """Display name: ARC Challenge_mt -> ARC Challenge MT, deu_mmlu_pro_mc ->
    MMLU Pro MC, tatoeba_eng_nno -> Tatoeba EN→NN."""
    if group in GROUP_NAMES:
        return GROUP_NAMES[group]
    if m := PAIR_RE.match(group):
        return f"{WORDS.get(m[1], m[1].capitalize())} {LANG[m[2]]}→{LANG[m[3]]}"
    group = re.sub(r"^deu_", "", group).replace("no_fact", "nofact")
    words = [WORDS.get(w, w.capitalize()) for w in group.split("_")]
    return " ".join(w for w in words if w)


def pick_metric(df: pd.DataFrame, preference: list[str]) -> pd.DataFrame:
    """Keep one metric per task: the first of `preference` it reports."""
    df = df[df["metric"].isin(preference)]
    rank = df["metric"].map({m: i for i, m in enumerate(preference)})
    return df[rank == rank.groupby(df["task"]).transform("min")]


def latest(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of each series' final checkpoint: the release (no checkpoint id)
    when present, else the one with most tokens (ties: highest checkpoint)."""
    ckpt = df["checkpoint"].fillna("<release>")
    best = (df.assign(ckpt=ckpt, release=df["checkpoint"].isna(),
                      t=df["tokens"].fillna(-1))
            .sort_values(["release", "t", "ckpt"])
            .groupby("series")["ckpt"].last())
    return df[ckpt == df["series"].map(best)]


def finish(df: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    """Latest checkpoint per series, row labels with the tokens seen, and only
    tasks every model has."""
    df = latest(df[df["series"].isin(models)])
    tok = df.groupby("series")["tokens"].first()
    label = {s: DISPLAY.get(models[s], models[s])
             + (f" ({tok[s] / 1e12:.1f}T)" if pd.notna(tok.get(s))
                             else "") for s in df["series"].unique()}
    df = df.assign(label=df["series"].map(label), name=df["series"].map(models),
                   order=df["series"].map(list(models).index))
    n = df["label"].nunique()
    df = df[df.groupby("task")["label"].transform("nunique") == n]
    return df.sort_values("order")


# --- loaders: long frames with series, checkpoint, tokens, group, task, score,
# translation (bool)

def load_eng_deu(language: str) -> pd.DataFrame:
    cols = ["model", "step", "task", "metric", "metric_filter", "score",
            "task_type", "tokens_trained", "language", "subtask_tree"]
    df = pd.concat([
        pd.read_parquet(hf_hub_download(repo, "scores.parquet",
                                        repo_type="dataset"), columns=cols)
        for repo in (REPO_ID, ELLAMIND_REPO_ID)], ignore_index=True)
    df = df[df["model"].isin(ENG_DEU_MODELS) & (df["language"] == language)]
    parent = leaf_groups(df)
    df = df[(df["task_type"] == "benchmark") & (df["metric_filter"] == "none")]
    df = pick_metric(df, METRIC_PREFERENCE)
    df = df.assign(series=df["model"],
                   checkpoint=df["step"].map(lambda s: None if pd.isna(s)
                                             else str(int(s))),
                   tokens=df["tokens_trained"],
                   group=df["task"].map(parent).fillna(df["task"]),
                   translation=False)
    return finish(df, ENG_DEU_MODELS)


def load_multilingual() -> pd.DataFrame:
    df = pd.read_parquet(hf_hub_download(
        REPO_ID, "multilingual_scores.parquet", repo_type="dataset"))
    # one task under two names: include_base_44_north macedonian (baselines
    # such as Qwen/Llama/Apertus) and ..._north_macedonian (late Prelude
    # checkpoints, EuroLLM, Olmo, Marin)
    df = df.assign(task=df["task"].str.replace(" ", "_"))
    df = df[~df["parent_task"].isin(EXCLUDE_BENCHMARKS)]
    df = df[df["task"].map(task_language).map(filter_languages)]
    anneal = df["revision"].str.startswith("anneal300b", na=False)
    df = df.assign(series=df["model"].mask(anneal, "prelude_anneal300b"))
    df = df[df["series"].isin(MULTILINGUAL_MODELS)]
    rev = df["revision"]
    df = df.assign(
        checkpoint=rev.where(rev.notna(), None),
        # anneal revisions count iterations globally (anneal300b_iter_0989075
        # is 8.3T), so the pretraining tokens/step applies to them too
        tokens=[None if pd.isna(r) else tokens_trained(m, r)
                for m, r in zip(df["model"], rev)],
        group=df["parent_task"],
    )
    main = pick_metric(df, METRIC_PREFERENCE).assign(translation=False)
    trans = pick_metric(df[~df["task"].isin(main["task"])],
                        TRANSLATION_METRICS).assign(translation=True)
    return finish(pd.concat([main, trans]), MULTILINGUAL_MODELS)


def norm_name(name: str) -> str:
    """Comparable model name: "OLMo3 7B" and "Olmo 3 7B" -> "olmo37b";
    HF ids lose the org and a -Base / -pt suffix."""
    name = re.sub(r"[-_](base|pt)$", "", name.split("/")[-1], flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def norwegian_models(df: pd.DataFrame) -> dict[str, str]:
    """NorEval model id -> our model name, for Prelude and every baseline of
    the other categories that NorEval has. The HF id is matched before the
    display name, so EuroLLM-9B-2512 maps to "EuroLLM 9B 2512" rather than
    the original "EuroLLM 9B"."""
    ours = [n for n in (*ENG_DEU_MODELS.values(), *MULTILINGUAL_MODELS.values())
            if not n.startswith("Prelude")]
    by_norm = {norm_name(n): n for n in ours}
    models = dict(NOREVAL_PRELUDE)
    for model, display in df[["model", "model_display_name"]].drop_duplicates().values:
        for key in (norm_name(model), norm_name(display)):
            if key in by_norm and model not in models:
                models[model] = by_norm[key]
    order = {n: i for i, n in enumerate(dict.fromkeys(
        [*NOREVAL_PRELUDE.values(), *ours]))}
    return dict(sorted(models.items(), key=lambda kv: order[kv[1]]))


def load_norwegian() -> pd.DataFrame:
    df = pd.read_parquet(hf_hub_download(
        REPO_ID, "noreval_scores.parquet", repo_type="dataset"))
    models = norwegian_models(df)
    print("NorEval baselines:", ", ".join(f"{m} -> {n}" for m, n in models.items()))
    df = df[(df["metric_filter"].fillna("none") == "none")
            & (df["num_fewshot"] == NOREVAL_SHOTS)]
    main = pick_metric(df, NOREVAL_METRICS).assign(translation=False)
    trans = pick_metric(df[~df["task"].isin(main["task"])],
                        TRANSLATION_METRICS).assign(translation=True)
    trans = trans[trans["task"].str.contains("tatoeba|translation")]
    df = pd.concat([main, trans])
    df = df.assign(series=df["model"],
                   checkpoint=df["step"].map(lambda s: None if pd.isna(s)
                                             else str(int(s))),
                   tokens=df["tokens_trained"],
                   group=df["task"])
    return finish(df, models)


def load_finnish() -> pd.DataFrame:
    from huggingface_hub import snapshot_download
    root = snapshot_download(REPO_ID, repo_type="dataset",
                             allow_patterns=[f"{FINNISH_DIR}/*/*.json"])
    rows = []
    for path in sorted(glob.glob(os.path.join(root, FINNISH_DIR, "*", "*.json"))):
        with open(path) as f:
            result = json.load(f)
        args = result["config"]["model_args"]
        if isinstance(args, str):     # "pretrained=...,revision=..."
            args = dict(kv.split("=", 1) for kv in args.split(","))
        key = args["pretrained"] + (f"@{args['revision']}" if args.get("revision") else "")
        for task, res in result["results"].items():
            if task in FINNISH_GROUPS:
                name, metric = FINNISH_GROUPS[task]
                rows.append((key, args.get("revision"), name, task,
                             res[f"{metric},none"], result["date"]))
    df = pd.DataFrame(rows, columns=["series", "checkpoint", "group", "task",
                                     "score", "date"])
    # a re-run of a task: keep the latest
    df = df.sort_values("date").drop_duplicates(["series", "task"], keep="last")
    prelude = df["series"].str.startswith("openeurollm/prelude@")
    tokens = [tokens_trained("openeurollm/prelude-checkpoints", c) if p else None
              for c, p in zip(df["checkpoint"], prelude)]
    df = df.assign(checkpoint=df["checkpoint"].where(prelude, None), tokens=tokens,
                   translation=False)
    return finish(df, FINNISH_MODELS)


# --- table and plot ----------------------------------------------------------

def scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(models x groups incl. AVG, models x translation groups or None).
    Mean over tasks within a group, then over groups for AVG."""
    def table(d):
        return d.groupby(["label", "group"])["score"].mean().unstack()
    acc = table(df[~df["translation"]])
    acc["AVG"] = acc.mean(axis=1)
    # fully open models by AVG (weakest first), then Prelude, then -- after a
    # gap in the plots -- the open-weight models, also by AVG
    first = df.drop_duplicates("label").set_index("label")
    style = first["name"].map(lambda n: STYLES[n][1])
    prelude = first.index.str.startswith("Prelude")
    def by_avg(mask):
        return list(acc.loc[first.index[mask], "AVG"].sort_values().index)
    labels = (by_avg(~prelude & (style != WEIGHTS))
              + list(first.loc[prelude, "order"].sort_values().index)
              + by_avg(~prelude & (style == WEIGHTS)))
    acc = acc.reindex(labels)
    tr = df[df["translation"]]
    return acc, (table(tr).reindex(labels) if not tr.empty else None)


def to_markdown(df: pd.DataFrame, names: dict[str, str]) -> str:
    """Two decimals (one for 0-100 scores); best value of each column bold.
    A heading row opens each openness group (fully open / open weights)."""
    lines = ["| model | " + " | ".join(df.columns) + " |",
             "|---|" + "---:|" * len(df.columns)]
    current = None
    for label, row in df.iterrows():
        group = GROUP_TITLES[STYLES[names[label]][1] == WEIGHTS]
        if group != current:
            lines.append(f"| ***{group}*** |" + " |" * len(df.columns))
            current = group
        cells = []
        for col, v in row.items():
            if pd.isna(v):
                cells.append("–")
                continue
            digits = 1 if df[col].max() > 1.5 else 2
            cell = f"{v:.{digits}f}"
            if round(v, digits) == round(df[col].max(), digits):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


INK, MUTED, GRID, AVG_BAND = "#1f1f1e", "#6b6a64", "#e4e3dd", "#f3f2ee"
GROUP_TITLES = {False: "Fully open models", True: "Close/open-weight models"}
GROUP_GAP = 0.8          # extra space between the two groups, in bar widths
LEGEND_BAND_IN = 1.0     # inches above the axes for the legends (title above)


def bars(ax, table: pd.DataFrame, names: dict[str, str]):
    """Grouped bars, one group per column; a hairline gap between bars, a wider
    one between fully open and open-weight models. AVG sits on a light band."""
    weights = [STYLES[names[l]][1] == WEIGHTS for l in table.index]
    gap = GROUP_GAP if any(weights) and not all(weights) else 0
    slot = 0.86 / (len(table) + gap)
    for i, (label, row) in enumerate(table.iterrows()):
        offset = slot * (i + 0.5 + (gap if weights[i] else 0))
        xs = [j - 0.43 + offset for j in range(len(table.columns))]
        color, style = STYLES[names[label]]
        if style == WEIGHTS:
            hatchcolor = lighten(color, 0.4)
            kw = dict(facecolor=color, edgecolor="white", hatch='\\\\\\',
                      linewidth=0, hatch_linewidth=0.75, hatchcolor=hatchcolor)
        else:
            kw = dict(facecolor=color, edgecolor="white")
        if names[label].startswith("Prelude"):   # outline only our model
            kw.update(edgecolor=PRELUDE_EDGE, linewidth=0.5, hatchcolor="white")
        ax.bar(xs, row, width=slot * 0.86, label=label, zorder=2, **kw)
    if "AVG" in table.columns:
        j = list(table.columns).index("AVG")
        ax.axvspan(j - 0.5, j + 0.5, color=AVG_BAND, zorder=0, lw=0)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=9.5, color=INK)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "AVG":
            tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(table.columns) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0, pad=4)


def legend_boxes(fig, ax, names: dict[str, str], fontsize: float = 10):
    """Two legend boxes in the band above the axes, right-aligned with them,
    two rows each, in bar order: fully open models (Prelude included) |
    open-weight models.
    Placed right to left in figure coordinates; an empty box is skipped.
    Returns the top of the legends in figure coordinates."""
    handles, labels = ax.get_legend_handles_labels()
    weights = [STYLES[names[l]][1] == WEIGHTS for l in labels]
    boxes = [(GROUP_TITLES[flag], [i for i, w in enumerate(weights) if w == flag])
             for flag in (False, True)]
    renderer = fig.canvas.get_renderer()
    pos = ax.get_position()
    right, top = fig.axes[-1].get_position().x1, pos.y1
    for a in fig.axes:          # clear panel titles such as "Translation"
        if a.get_title(loc="left"):
            top = max(top, a.title.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()).y1 if a.get_title() else
                a._left_title.get_window_extent(renderer).transformed(
                    fig.transFigure.inverted()).y1)
    top += 0.01
    legend_top = top
    for title, idx in reversed([(t, idx) for t, idx in boxes if idx]):
        leg = fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
                         title=title, title_fontsize=fontsize, fontsize=fontsize,
                         loc="lower right", bbox_to_anchor=(right, top),
                         ncol=-(-len(idx) // 2), frameon=False,
                         columnspacing=1.4, handlelength=1.8, handleheight=1.0,
                         labelcolor=INK)
        leg.get_title().set_color(MUTED)
        leg._legend_box.align = "left"
        box = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        right, legend_top = box.x0 - 0.01, max(legend_top, box.y1)
    return legend_top


def plot(name: str, df: pd.DataFrame, acc: pd.DataFrame,
         trans: pd.DataFrame | None, out: str):
    names = df.drop_duplicates("label").set_index("label")["name"].to_dict()
    ncols = len(acc.columns) + (len(trans.columns) if trans is not None else 0)
    height = 5.6
    fig, axes = plt.subplots(
        1, 2 if trans is not None else 1,
        figsize=(max(0.95 * ncols + 3.5, 11), height), squeeze=False,
        gridspec_kw={"width_ratios": [len(acc.columns), len(trans.columns) + 0.4],
                     "wspace": 0.08}
        if trans is not None else None)
    ax = axes[0, 0]
    bars(ax, acc, names)
    ax.set_ylabel("score (accuracy / exact match)", color=MUTED, fontsize=9.5)
    top = acc.max().max()
    ax.set_ylim(0, max(1.0, top + 0.03))
    if trans is not None:
        ax2 = axes[0, 1]
        bars(ax2, trans, names)
        ax2.set_ylim(0, trans.max().max() * 1.12)
        metrics = sorted(set(df.loc[df["translation"], "metric"]))
        ax2.set_ylabel(f"{' / '.join(m.upper() for m in metrics)} (0-100)",
                       color=MUTED, fontsize=9.5)
        ax2.set_title("Translation", loc="left", fontsize=10.5, color=INK, pad=6)
    fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_BAND_IN / height))
    legend_top = legend_boxes(fig, ax, names)   # after layout: positions final
    fig.text(ax.get_position().x0, legend_top + 0.1 / height, f"{name} benchmarks",
             ha="left", va="bottom", fontsize=14, fontweight="semibold", color=INK)
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")


README_INTRO = """# Prelude evaluations

Compares OpenEuroLLM Prelude 9B with fully open models (open weights and
training data) and open-weight models of similar size on English,
Multilingual, German, Norwegian and Finnish benchmarks; run it with
`uv run plot_results.py`. The script downloads the scores from the HF datasets
`openeurollm/prelude-base-eval-scores` and `ellamind/eval-scores-ref`, takes
each model's released model (else its latest checkpoint), keeps only the tasks
every model of a category has, and averages tasks within each benchmark group
and then over groups (AVG, which excludes BLEU/chrF translation scores). It
writes one bar plot per category to `figures/` and regenerates the Results
section below, where the best value of each column is bold and, in the plots,
fully open models (white diagonal hatching, Prelude last, outlined) are
followed by open-weight models (black diagonal hatching).

## Update model card

The Evaluation section of the
[prelude-base model card](https://huggingface.co/openeurollm/prelude-base) is
`model_card_evaluation.md`; edit it there, regenerate the figures with
`uv run plot_results.py`, then run

```bash
uv run update_model_card_evaluation_section.py                  # show the diff only
uv run update_model_card_evaluation_section.py --commit-change  # push it
```

The script replaces the model card's Evaluation section (up to the next heading
of the same level) with the markdown file and always prints the diff. With
`--commit-change` it uploads the new README.md and the referenced figures to the
model repo in one commit, which needs an HF token with write access
(`hf auth login`).

"""


def write_readme(sections: list[tuple[str, str, str]]):
    """Keep README.md up to the "# Results" heading (or start it from
    README_INTRO) and regenerate everything from that heading on."""
    head = README_INTRO
    if os.path.exists(README):
        text = open(README).read()
        if "\n# Results" in text:
            head = text[:text.index("\n# Results") + 1]
    today = datetime.date.today().isoformat()
    parts = [head + "# Results\n\n"
             f"Automatically generated by `plot_results.py` on {today}; do "
             "not edit below this line.\n"]
    for name, figure, table in sections:
        parts.append(f"\n## {name}\n\n![{name} results]({figure})\n\n{table}\n")
    with open(README, "w") as f:
        f.write("".join(parts))
    print(f"wrote {README}")


# Tasks of the benchmarks that were decontaminated from the training data
# (English: MMLU, Lambada, ARC, HellaSwag, CommonsenseQA, PIQA, GSM8K, MATH,
# GPQA; Multilingual: ARC MT, Belebele, Global MMLU, INCLUDE, MultiBLiMP,
# xHellaSwag, XCSQA). MMLU Pro and Global PIQA are not on the list.
DECONTAMINATED_RE = re.compile(
    r"^(mmlu_(hum|other|ss|stem)_|lambada|arc_(challenge|easy)|hellaswag"
    r"|x?csqa|piqa_mc|gsm8k|minerva_math|gpqa|belebele|global_mmlu|include_"
    r"|multiblimp)")
# Only applied where the task names match the decontaminated benchmarks; the
# German (translated) and Norwegian (NorEval) suites only get the "all" plot.
DECONTAMINATION_SPLITS = {"English", "Multilingual"}
SUBSETS = {   # suffix -> (title suffix, task filter)
    "all": ("", lambda t: True),
    "decontaminated_only": (" (decontaminated only)",
                            lambda t: bool(DECONTAMINATED_RE.match(t))),
    "non_decontaminated_only": (" (non-decontaminated only)",
                                lambda t: not DECONTAMINATED_RE.match(t)),
}


def report(name: str, df: pd.DataFrame, figure: str) -> str:
    """Print the markdown table, write the plot; return the table."""
    acc, trans = scores(df)
    print(f"\nBenchmark {name}\n")
    table = acc.drop(columns="AVG")
    if trans is not None:
        table = table.join(trans)
    table["AVG"] = acc["AVG"]           # AVG last; excludes translation
    names = df.drop_duplicates("label").set_index("label")["name"].to_dict()
    md = to_markdown(table, names)
    print(md)
    print()
    plot(name, df, acc, trans, os.path.join(HERE, figure))
    return md


def main():
    os.makedirs(FIGURES, exist_ok=True)
    groups = {
        "English": lambda: load_eng_deu("eng"),
        "Multilingual": load_multilingual,
        "German": lambda: load_eng_deu("deu"),
        "Norwegian": load_norwegian,
        "Finnish": load_finnish,
    }
    sections = []
    for name, load in groups.items():
        df = load()
        df = df.assign(group=df["group"].map(pretty_group))
        subsets = (SUBSETS if name in DECONTAMINATION_SPLITS
                   else {"all": SUBSETS["all"]})
        for suffix, (title, keep) in subsets.items():
            figure = f"figures/plot_results_{name.lower()}_{suffix}.png"
            md = report(name + title, df[df["task"].map(keep)], figure)
            if suffix == ("decontaminated_only" if len(subsets) > 1 else "all"):
                sections.append((name + title, figure, md))
    write_readme(sections)


if __name__ == "__main__":
    main()
