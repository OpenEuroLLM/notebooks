import json
from pathlib import Path

import pandas as pd

from evaldata_utils import load_mapping
from figure_utils import bench_sel


def load_model_checkpoints_list():
    with open("models-14-05.txt", "r") as f:
        res = f.readlines()

    return list(sorted([Path(x).name.strip() for x in res]))

def load_model_evals():
    path = Path(__file__).parent.parent / "data" / "results-22-05.csv.zip"
    df_all = pd.read_csv(path)
    return df_all, list(sorted(df_all.model_name.unique()))


# TODO
models_checkpoints = load_model_checkpoints_list()
df_evals, models_evals = load_model_evals()
_, dict_mapping = load_mapping()

def sanitize(x):
    if x.endswith("run-1"):
        x = x[:-5]
    res = x if x.endswith("LEONARDO") else "_".join(x.split("_")[:-1])
    res = res.replace("_", "")
    res = res.lower()
    assert len(res) > 0
    return res

mapping = {sanitize(k): v for k, v in dict_mapping.items()}

with open("model-evals-14-05.txt", "w") as f:
    f.write("\n".join(models_evals))

models_evals = list(set([sanitize(x) for x in models_evals]))
models_checkpoints = list(set([sanitize(x) for x in models_checkpoints]))
print(f"Loaded {len(models_checkpoints)} models from checkpoints")
print(f"Loaded {len(models_evals)} models from evaluations")


intersection = sorted(set(models_evals).difference(models_checkpoints))
print(f"{len(intersection)} models that have an evaluation but no converted checkpoint:")
print("\n".join(intersection))

intersection = sorted(set(models_checkpoints).difference(models_evals))
print(f"\n{len(intersection)} models that have a converted checkpoint but no evaluation:")
print("\n".join(intersection))



df_task_model_count = df_evals.pivot_table(index="benchmark", columns="model_path", aggfunc="count", fill_value=0)["lr"].loc[bench_sel].T
df_task_model_count[df_task_model_count>1] = 1

series_model_count = df_task_model_count.sum(axis=1)
print("Task number distribution missing per checkpoint")
print(series_model_count.value_counts().sort_index())

df_task_model_count["sum"] = df_task_model_count.sum(axis=1)
df_task_model_count.sort_values(by="sum", inplace=True)
print(df_task_model_count.head())


def n_few_shot(task: str):
    if 'mmlu' in task:
        return 5
    elif task in ["copa", "openbookqa", "lambada_openai", "winogrande", "social_iqa"]:
        return 0
    elif task in ["commonsense_qa", "piqa", "arc_challenge", "arc_easy", "hellaswag", "boolq"]:
        return 10
    else:
        raise ValueError(task)

rows = []
missing_checkpoint_tasks = {}
for model, row in df_task_model_count.T.items():
    missing_tasks = [benchmark for benchmark, count in row.to_dict().items() if count == 0]
    if len(missing_tasks) > 0:
        missing_checkpoint_tasks[model] = missing_tasks
        for task in missing_tasks:
            rows.append({
                "model": model,
                "task": task,
                "n_few_shot": n_few_shot(task)
            })

n_missing_model_task = sum(len(x) for x in missing_checkpoint_tasks.values())
print(f"{n_missing_model_task} tasks missing saving in missing-tasks.csv")

pd.DataFrame(rows).sort_values(by=["model", "n_few_shot", "task"]).to_csv("missing-tasks.csv", index=False)

print(missing_checkpoint_tasks)
print(sum(len(x) for x in missing_checkpoint_tasks.values()))
