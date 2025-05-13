import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class ModelConfig:
    n_iter: int
    size: float
    data: str
    tokenizer: str
    n_tokens: int
    global_bs: int
    context: int
    schedule: str
    lr: float
    warmup: int


def parse_model_path(path):
    # Extract the filename part (after the last /)
    parts = path.split('/')

    # Find the part containing all the model config (long string with parameters)
    # /leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/open-sci-ref_model-1.7b_data-DCLM_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14070018/hf/iter_0114000
    config_string = None
    for part in parts:
        if "model-" in part or "data-" in part:
            config_string = part
            break
    assert not config_string is None, path

    iter_match = re.search(r'iter_(\d+)', path)
    n_iter = int(iter_match.group(1)) if iter_match else None

    size_match = re.search(r'model-(\d+\.?\d*)b', config_string)
    size = float(size_match.group(1)) if size_match else None

    data_match = re.search(r'data-([^_]+)', config_string)
    data = data_match.group(1) if data_match else None

    tokenizer_match = re.search(r'tokenizer-([^_]+)', config_string)
    tokenizer = tokenizer_match.group(1) if tokenizer_match else None

    tokens_match = re.search(r'samples-(\d+)B', config_string)
    n_tokens = int(tokens_match.group(1)) if tokens_match else None

    bs_match = re.search(r'global_bs-(\d+)', config_string)
    global_bs = int(bs_match.group(1)) if bs_match else None

    context_match = re.search(r'context-(\d+)', config_string)
    context = int(context_match.group(1)) if context_match else None

    schedule_match = re.search(r'schedule-(\w+)', config_string)
    schedule = schedule_match.group(1) if schedule_match else None

    lr_match = re.search(r'lr-(\d+e-\d+)', config_string)
    lr = float(lr_match.group(1)) if lr_match else None

    warmup_match = re.search(r'warmup-(\d+)', config_string)
    warmup = int(warmup_match.group(1)) if warmup_match else None

    return ModelConfig(
        n_iter=n_iter,
        size=size,
        data=data,
        tokenizer=tokenizer,
        n_tokens=n_tokens,
        global_bs=global_bs,
        context=context,
        schedule=schedule,
        lr=lr,
        warmup=warmup
    )


def load_data(
        path: str | Path,
        metrics: list[str] | None = None,
        filter_model_path: Callable[[str], bool] = None,
):
    if metrics is None:
        metrics = [
            "mmlu/acc",
            "mmlu_pro/exact_match,custom-extract",
            "copa/acc",
            "lambada_openai/acc",
            "openbookqa/acc",
            "winogrande/acc",
            "arc_challenge/acc_norm",
            "arc_easy/acc_norm",
            "boolq/acc",
            "commonsense_qa/acc",
            "hellaswag/acc_norm",
            "piqa/acc_norm",
            "social_iqa/acc",
        ]

    rows = []
    rows_baselines = []
    path = Path(path)
    assert path.exists(), f"provided path {path} does not exists."
    json_files = list(Path(path).rglob("*results*.json"))

    for json_file in json_files:
        with open(json_file, "r") as f:
            json_dict = json.load(f)
            model_path = json_dict["model_name"]
            # allow to filter smol/qwen evals
            for col in metrics:
                benchmark_name, metric = col.split("/")
                if benchmark_name in json_dict["results"]:
                    metric_col = f"{metric},none" if benchmark_name != "mmlu_pro" else metric

                    model_name = Path(model_path).parent.parent.name
                    model_name = "".join(model_name.split("_")[:-1])

                    row = {
                        "model_name": model_name,
                        "model_path": model_path,
                        # "n_iter": hp["n_iter"],
                        "benchmark": benchmark_name,
                        "metric": metric,
                        "total_evaluation_time_seconds": float(json_dict["total_evaluation_time_seconds"]),
                        "value": json_dict["results"][benchmark_name][metric_col],
                    }
                    # hp = parse_model_path(model_path).__dict__
                    # row.update(hp)
                    # TODO how to distinguish best models from the hub with ours?
                    # TODO pass filter
                    if filter_model_path is None or filter_model_path(model_path):
                        rows.append(row)
    return pd.DataFrame(rows)

def load_mapping():
    with open(Path(__file__).parent / "mapping.json", "r") as f:
        dict_mapping = json.load(f)
    rows = []
    for key, value in dict_mapping.items():
        row = value["config"].copy()
        row["checkpoint_path"] = key
        hidden_size = row["hidden_size"]
        num_layers = row["num_layers"]
        ffn_hidden_size = row["ffn_hidden_size"]
        row["model_size"] = (
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
        model_size = row["model_size"]
        row["model_size_str"] = (
            f"{model_size:.2f}b" if model_size < 1 else f"{model_size:.1f}b"
        )

        rows.append(row)
    return pd.DataFrame(rows), dict_mapping

def main():
    # Example usage
    path = "/leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/open-sci-ref_model-1.7b_data-DCLM_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14070018/hf/iter_0114000"
    config = parse_model_path(path)
    print(config)

    df = load_data(
        path=Path("/Users/salinasd/Documents/code/Megatron-LM-Open-Sci/data/json_files_selected/"),
        filter_model_path=lambda model_path: "leonardo" in model_path,
    )
    print(df.shape)


if __name__ == '__main__':
    main()

