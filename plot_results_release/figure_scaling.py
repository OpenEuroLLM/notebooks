import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np


def flops(n_tokens_T: float, n_params_B: float):
    return 6 * n_tokens_T * 1e9 * n_params_B * 1e12

flops_baselines = {
    'EuroLLM-1.7B': flops(n_tokens_T=1.7, n_params_B=4),
    'EuroLLM-9B': flops(n_tokens_T=9, n_params_B=4),
    # Not sure which one is correct
    # "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens"
    # https://qwenlm.github.io/blog/qwen2.5/
    # We are not authorized to share the details right now but the rough number is over 3T tokens for Qwen1.5 and over
    # 7T tokens for Qwen2.
    # https://github.com/QwenLM/Qwen3/issues/562
    'Qwen2.5-0.5B': flops(n_params_B=0.5, n_tokens_T=18),
    'Qwen2.5-1.5B': flops(n_params_B=1.5, n_tokens_T=18),
    'Qwen2.5-3B': flops(n_params_B=3, n_tokens_T=18),
    'Qwen2.5-7B': flops(n_params_B=7, n_tokens_T=18),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-135M
    'SmolLM2-135M': flops(n_params_B=0.135, n_tokens_T=2),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-360M
    'SmolLM2-360M': flops(n_params_B=0.36, n_tokens_T=4),
    'SmolLM2-1.7B': flops(n_params_B=2, n_tokens_T=11),
    # https://arxiv.org/html/2408.00118v1 We train Gemma 2 27B on 13 trillion tokens of primarily-English data,
    # the 9B model on 8 trillion tokens, and the 2B on 2 trillion tokens
    'gemma-2-2b': flops(n_params_B=2.6, n_tokens_T=2),
    'apple/DCLM-7B': flops(n_params_B=7, n_tokens_T=4),
    'TRI-ML/DCLM-1B': flops(n_params_B=1.4, n_tokens_T=4),
}

x_col = "Training FLOPs"
y_col = "Average downstream performance"
df = load_data()

df.replace(
    {
        'Nemotron-cc-2024-HQ-real-synth-mix': 'Nemotron-cc-hq'
    },
    inplace=True
)

# read results from baseline
df_baselines = pd.read_csv("/p/project/laionize/marianna/megatron/notebooks/plot_results_release/data/baselines-24-07.csv.zip")
df_baselines_pivot = df_baselines.pivot_table(
    index="model_name", columns="benchmark", values="value"
).loc[:, bench_sel]


# compute average performance for all
baselines_avg = df_baselines_pivot.mean(axis=1)
df_baselines_flops_perf = pd.DataFrame({x_col: flops_baselines, "Avg performance": baselines_avg})

for n_tokens in [
    # "300B",
    # "1T",
    "all",
]:
    df_all = df.copy()
    df_all.dataset = df_all.dataset.apply(lambda x: "DCLM-open-sci" if "DCLM" in x else x)
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    dd = df_avg.pivot_table(
        index=x_col,
        values=y_col,
        columns="dataset",
    )

    n_colors = len(dd.columns)
    color_map = cm.get_cmap('tab20', n_colors)  # or 'nipy_spectral', 'hsv', 'tab10', etc.
    colors = [mcolors.to_hex(color_map(i)) for i in range(n_colors)]
    for col, color in zip(dd.columns, colors):
        dd_plot = dd.loc[:, col].dropna()
        ax = dd_plot.plot(ax=ax, label=col, marker="*", color=color)
        if "Nemotron" in col:
            for (x, y), s in zip(dd_plot.reset_index().values, [0.13, 0.4, 1.3, 1.7]):
                ax.text(
                    x=x * 0.68, y=y * 1.04, s=f"{s}B",
                    #color=plt.gca().lines[-1].get_color()
                    color="black"
                )

    baselines = ["Qwen2.5", "SmolLM2", 'EuroLLM', "DCLM"]
    for baseline in baselines:
        df_baseline = df_baselines_flops_perf[df_baselines_flops_perf.index.str.contains(baseline)].copy()
        df_baseline.sort_values(x_col, inplace=True)
        df_baseline.plot(x=x_col, y="Avg performance", ax=ax, label=baseline, ls="--", marker="o")


        def _n_params(method: str):
            return method.split("-")[-1]

        # TODO customize offset
        sizes = [_n_params(t) for t in df_baseline.index]
        if baseline == "EuroLLM":
            offset = (0.65, 1.02)
        elif "Qwen" in baseline:
            offset = (1.15, 0.99)
        else:
            offset = (0.68, 0.94)
        print(baseline, offset)
        for (x, y), s in zip(df_baseline.values, sizes):
            ax.text(
                x=x * offset[0], y=y * offset[1], s=s,
                color=plt.gca().lines[-1].get_color(),
                # color="black"
            )

        # for baseline in baselines_avg.keys():
        #     if baseline in flops_baselines:
        #         ax.plot(flops_baselines[baseline], baselines_avg[baseline], 'o', label=baseline)

    ax.set_xscale("log")
    ax.set_ylabel(y_col)
    ax.grid(visible=True)

    # df_baselines_flops_perf.plot(ax=ax, marker="x")

    ax.legend(ncols=1, loc="center left", bbox_to_anchor=(1.0, 0.5), )
    ax.set_title(f"Scaling comparison with reference models trained on {n_tokens}");
    plt.tight_layout()
    # plt.show()
    plt.savefig("/p/project/laionize/marianna/megatron/notebooks/plot_results_release/results.png")
