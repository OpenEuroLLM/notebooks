import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols


def flops(n_tokens_B: float, n_params_T: int):
    return 6 * n_tokens_B * 1e9 * n_params_T * 1e12

# IMPORT TODO check those numbers, some are from memory
flops_baselines = {
    'EuroLLM-1.7B': flops(n_tokens_B=1.7, n_params_T=4),
    'Qwen2.5-0.5B': flops(n_tokens_B=0.5, n_params_T=18),
    'Qwen2.5-1.5B': flops(n_tokens_B=1.5, n_params_T=18),
    'Qwen2.5-3B': flops(n_tokens_B=3, n_params_T=18),
    'SmolLM2-1.7B': flops(n_tokens_B=2, n_params_T=11),
    'gemma-2-2b': flops(n_tokens_B=2.6, n_params_T=2),
}

df = load_data()


for n_tokens in ["300B", "1T", "all"]:
    df_all = df.copy()
    if n_tokens != "all":
        df_all = df_all[df_all.n_tokens == n_tokens]
    df_pivot = df_all.pivot_table(
        columns=["n_tokens", "model_name", "n_iter"],
        index="benchmark",
        values="value",
    ).loc[bench_sel]

    # remove checkpoint which are missing one evaluation
    df_pivot = df_pivot.dropna(axis=1)

    # gets average performance for every model
    df_avg = df_pivot.mean().reset_index()
    avg_model = {}
    for model_name in df_avg.model_name.unique():
        df_model = df_avg[df_avg.model_name == model_name].sort_values("n_iter")
        avg_model[model_name] = df_model[0].values[-1]
    avg_model = {k: v for k,v in avg_model.items()}

    df_avg_model = pd.DataFrame(index=avg_model.keys(), data={"perf": avg_model.values()})
    df_hp = df_all.drop_duplicates("model_name").set_index("model_name").loc[avg_model.keys(), hp_cols]
    df_avg_model = pd.concat([df_avg_model, df_hp], axis=1)
    n_tokens_mapping = {"300B": 300 * 1e9, "50B": 50 * 1e9, "1T": 1e12}
    df_avg_model["FLOPS"] = 6 * df_avg_model["n_tokens"].apply(lambda x: n_tokens_mapping[x])  * df_avg_model["size"] * 1e9

    df_baselines = pd.read_csv("results-baselines.csv.zip")
    df_baselines_pivot = df_baselines.pivot_table(index="model_name", columns="benchmark", values="value").loc[:, bench_sel]
    baselines_avg = df_baselines_pivot.mean(axis=1)
    df_baselines_flops_perf = pd.DataFrame({"FLOPS": flops_baselines, "Avg performance": baselines_avg})

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    dd = df_avg_model.pivot_table(
        index="FLOPS",
        values="perf",
        columns="dataset",
    )
    for col in dd.columns:
        ax = dd.loc[:, col].dropna().plot(ax=ax, label=col, marker="*")

    df_qwen = df_baselines_flops_perf[df_baselines_flops_perf.index.str.contains("Qwen")]
    df_qwen.plot(x="FLOPS", y="Avg performance", ax=ax, label="Qwen", color="black", ls="--", marker="o")
    for baseline in baselines_avg.keys():
        if "Qwen" not in baseline and baseline in flops_baselines:
            ax.plot(flops_baselines[baseline], baselines_avg[baseline], 'o', label=baseline)

    ax.set_xscale("log")
    ax.set_ylabel("Avg performance")
    ax.grid()

    # df_baselines_flops_perf.plot(ax=ax, marker="x")

    ax.legend(ncols=1, loc="center left", bbox_to_anchor=(1.0, 0.5), )
    ax.set_title(f"Scaling comparison with reference models trained on {n_tokens}");
    plt.tight_layout()
    plt.show()