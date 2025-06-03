import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols


def flops(n_tokens_B: float, n_params_T: float):
    return 6 * n_tokens_B * 1e9 * n_params_T * 1e12


flops_baselines = {
    'EuroLLM-1.7B': flops(n_tokens_B=1.7, n_params_T=4),
    'EuroLLM-9B': flops(n_tokens_B=9, n_params_T=4),
    # Not sure which one is correct
    # "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens"
    # https://qwenlm.github.io/blog/qwen2.5/
    # We are not authorized to share the details right now but the rough number is over 3T tokens for Qwen1.5 and over
    # 7T tokens for Qwen2.
    # https://github.com/QwenLM/Qwen3/issues/562
    'Qwen2.5-0.5B': flops(n_params_T=0.5, n_tokens_B=18),
    'Qwen2.5-1.5B': flops(n_params_T=1.5, n_tokens_B=18),
    'Qwen2.5-3B': flops(n_params_T=3, n_tokens_B=18),
    'Qwen2.5-7B': flops(n_params_T=7, n_tokens_B=18),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-135M
    'SmolLM2-135M': flops(n_params_T=0.135, n_tokens_B=2),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-360M
    'SmolLM2-360M': flops(n_params_T=0.36, n_tokens_B=4),
    'SmolLM2-1.7B': flops(n_params_T=2, n_tokens_B=11),
    # https://arxiv.org/html/2408.00118v1 We train Gemma 2 27B on 13 trillion tokens of primarily-English data,
    # the 9B model on 8 trillion tokens, and the 2B on 2 trillion tokens
    'gemma-2-2b': flops(n_params_T=2.6, n_tokens_B=2),
}

x_col = "Training FLOPs"
y_col = "Average downstream performance"
df = load_data()

df.replace(
    {
        'Nemotron-cc-2024-HQ-real-synth-mix': 'Nemotron-cc'
    },
    inplace=True
)

for n_tokens in ["300B", "1T", "all"]:
    df_all = df.copy()
    if n_tokens != "all":
        df_all = df_all[df_all.n_tokens == n_tokens]

    # for all training budget and models, we take the latest value
    df_all.sort_values(by="n_iter", inplace=True)
    df_pivot = df_all.pivot_table(
        columns=["n_tokens", "model_name"],
        index="benchmark",
        values="value",
        aggfunc="last",
    ).loc[bench_sel]

    # we remove checkpoint which are missing an evaluation for any of the selected benchmarks
    df_pivot = df_pivot.dropna(axis=1)

    # gets average performance for every model
    df_avg = df_pivot.mean().reset_index(name=y_col).loc[:, ["model_name", y_col]]
    df_hp = df_all.drop_duplicates("model_name").set_index("model_name").loc[df_avg.model_name, hp_cols]
    df_avg = pd.concat([df_avg.set_index("model_name"), df_hp], axis=1)
    n_tokens_mapping = {"300B": 300 * 1e9, "50B": 50 * 1e9, "1T": 1e12}
    df_avg[x_col] = 6 * df_avg["n_tokens"].apply(lambda x: n_tokens_mapping[x]) * df_avg["size"] * 1e9

    # read results from baseline
    df_baselines = pd.read_csv("results-baselines.csv.zip")
    df_baselines_pivot = df_baselines.pivot_table(
        index="model_name", columns="benchmark", values="value"
    ).loc[:, bench_sel]

    # compute average performance for all
    baselines_avg = df_baselines_pivot.mean(axis=1)
    df_baselines_flops_perf = pd.DataFrame({x_col: flops_baselines, "Avg performance": baselines_avg})

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    dd = df_avg.pivot_table(
        index=x_col,
        values=y_col,
        columns="dataset",
    )
    for col in dd.columns:
        ax = dd.loc[:, col].dropna().plot(ax=ax, label=col, marker="*")

    baselines = ["Qwen", "SmolLM2", 'EuroLLM']
    for baseline in baselines:
        df_baseline = df_baselines_flops_perf[df_baselines_flops_perf.index.str.contains(baseline)].copy()
        df_baseline.sort_values(x_col, inplace=True)
        df_baseline.plot(x=x_col, y="Avg performance", ax=ax, label=baseline, ls="--", marker="o")
        for baseline in baselines_avg.keys():
            if baseline not in baseline and baseline in flops_baselines:
                ax.plot(flops_baselines[baseline], baselines_avg[baseline], 'o', label=baseline)

    ax.set_xscale("log")
    ax.set_ylabel(y_col)
    ax.grid()

    # df_baselines_flops_perf.plot(ax=ax, marker="x")

    ax.legend(ncols=1, loc="center left", bbox_to_anchor=(1.0, 0.5), )
    ax.set_title(f"Scaling comparison with reference models trained on {n_tokens}");
    plt.tight_layout()
    plt.show()
