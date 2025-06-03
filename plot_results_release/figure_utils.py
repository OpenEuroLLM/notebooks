from pathlib import Path
import pandas as pd


metrics = [
    "mmlu/acc",
    # "mmlu_pro/exact_match,custom-extract",
    "copa/acc",
    "lambada_openai/acc",
    # "openbookqa/acc",
    "openbookqa/acc_norm",
    "winogrande/acc",
    "arc_challenge/acc_norm",
    # "arc_challenge/acc",
    "arc_easy/acc_norm",
    # "arc_easy/acc",
    "boolq/acc",
    "commonsense_qa/acc",
    "hellaswag/acc_norm",
    "piqa/acc_norm",
    # "piqa/acc",
    # "social_iqa/acc",
]

bench_sel = [x.split("/")[0] for x in metrics]

hp_cols = [
    'size',
    'dataset',
    'tokenizer',
    'n_tokens',
    'global_batch_size',
    'seq_length',
    'lr_decay_style',
    'lr',
    'lr_warmup_iters']


def load_data():
    df = pd.read_csv(Path(__file__).parent / "data/results-22-05.csv.zip")
    df["n_iter"] = df.model_path.apply(lambda x: int(x.split("_")[-1]))
    df["metric_name"] = df.apply(lambda row: row["benchmark"] + "/" + row["metric"], axis=1)
    df = df[df.metric_name.isin(metrics)]
    return df
