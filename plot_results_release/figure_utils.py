import json
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


def sanitize(x):
    if x.endswith("run-1"):
        x = x[:-5]
    if "machine-LEONARDO":
        res = x.split("machine-LEONARDO")[0]
    # res = x if x.endswith("LEONARDO") else "_".join(x.split("_")[:-1])
    res = res.replace("_", "")
    res = res.lower()
    assert len(res) > 0
    return res


def _add_hp(root, df):
    with open(root / "log_dir_name_mapping.jsonl", "r") as f:
        mapping_rows = []
        for line in f:
            mapping_rows.append(json.loads(line))
    mapping = {row["log_file_name"]: row for row in mapping_rows}

    with open("_eval_models.txt", "w") as f:
        f.write("\n".join(sorted(df.model_name.unique())))

    map_sanitize_to_original_name = {
        sanitize(model): model for model in mapping.keys()
    }
    sanitize_models_mapping = [sanitize(x) for x in sorted(mapping.keys())]

    # checks that all models appearing in evaluation appears in mapping.json
    for model in df.model_name.unique():
        if sanitize(model) not in sanitize_models_mapping:
            print(model, sanitize(model))

    df_hp = pd.DataFrame([x["config"] for x in df.apply(lambda row: mapping[map_sanitize_to_original_name[sanitize(row["model_name"])]], axis=1).tolist()])

    df = pd.concat([df, df_hp], axis=1)
    return df


def format_large_number(num: float):
    """
    Returns:
        str: Formatted string representation (e.g., "50B", "1T")
    """
    if num >= 1e12:
        return f"{int(num / 1e12)}T"
    elif num >= 1e9:
        return f"{int(num / 1e9)}B"
    else:
        return str(num)  # For smaller numbers, just return as is

def load_data():
    root = Path(__file__).parent
    df = pd.read_csv(root / "data/results-04-06.csv.zip")
    df = _add_hp(root, df)
    df["n_iter"] = df.model_path.apply(lambda x: int(x.split("_")[-1]))
    df["metric_name"] = df.apply(lambda row: row["benchmark"] + "/" + row["metric"], axis=1)
    df = df[df.metric_name.isin(metrics)]

    df["n_tokens"] = df.apply(lambda row: format_large_number(row["seq_length"] * row["global_batch_size"] * row["train_iters"]), axis=1)

    def _size(hidden_size, num_layers, ffn_hidden_size, **kwargs):
        model_size = (
            (50432 * hidden_size)
            + (
                num_layers
                * (
                    (4 * hidden_size**2)
                    + 2 * hidden_size * ffn_hidden_size
                    + ffn_hidden_size * hidden_size
                )
            )
        ) / 1_000_000_000

        if model_size < 1.0:
            return float(
                f"{model_size:.2f}"
            )
        else:
            return float(
                f"{model_size:.1f}"
            )

    df["size"] = df.apply(lambda row: _size(**row), axis=1)
    return df


if __name__ == '__main__':
    load_data()