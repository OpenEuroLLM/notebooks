import matplotlib.pyplot as plt
import numpy as np

from figure_utils import load_data, bench_sel, metrics, figure_path

df_all = load_data()
# compute average per iterationdf_sub.loc[:, ["size", "benchmark", "value"]] ".groupby("n_iter").mean()

# allows to have 5x2 plot
metrics.remove("commonsense_qa/acc")


fig, axes = plt.subplots(2, len(bench_sel) // 2, figsize=(16, 6), sharey=False, sharex=True)
axes = np.ravel(axes)
df_plot = df_all.copy()
df_plot.dataset = df_plot.dataset.apply(lambda s: s.replace('Nemotron-cc-2024-HQ-real-synth-mix', 'Nemotron-cc-hq'))
df_plot.dataset = df_plot.dataset.str.lower()
n_tokens = "300B"
config = {
    # "size": 1.7,
    # "tokenizer": "GPT-NeoX",
    #"global_batch_size": 1008,
    "n_tokens": n_tokens,
    "seq_length": 4096,
    # "lr_decay_style": "WSD",
    "lr_warmup_iters": 25000,
}
mask = None
for key, value in config.items():
    if mask is None:
        mask = (df_all.loc[:, key] == value)
    else:
        mask &= (df_all.loc[:, key] == value)

size = 1.7
df_sub = df_plot.loc[(mask) & (df_plot.loc[:, "size"] == size)].copy()
df_sub["tokens"] = df_plot["n_iter"] * 1008 * 4096

for i, metric in enumerate(metrics):
    ax = axes[i]
    bench_sel = metric.split("/")[0]
    df_iter = df_sub.loc[df_sub.metric_name == metric, :].pivot_table(
        index=["dataset", "tokens"], columns="benchmark", values="value"
    ).loc[:, bench_sel]

    #df_iter_pivot = df_iter.reset_index().pivot_table(index="tokens", columns="dataset", values=0)
    df_iter_pivot = df_sub.loc[df_sub.metric_name == metric, :].pivot_table(
        index="tokens", columns="dataset", values="value"
    )
    dataset_order = [
        "Nemotron-cc-hq",
        "DCLM",
        "HPLT-2.0",
        "FineWeb-Edu-1.4T",
        "Pile",
        "SlimPajama",
        "CommonCorpus",
        "C4",
    ]
    dataset_order = [x.lower() for x in dataset_order]
    # fix order to have same colors across plots
    df_iter_pivot = df_iter_pivot.loc[:, [x for x in dataset_order if x in df_iter_pivot.columns]]
    plot_result = df_iter_pivot.plot(
        ax=ax,
    )
    ax.grid()
    ax.set_title(f"{bench_sel}")
    ax.set_xlabel("Number of tokens");
    # ax.set_xlabel("Number of iterations");
    #ax.set_ylabel(f"Average performance on {len(bench_sel)} tasks");
    if i == 0 or i == 5:
        ax.set_ylabel(f"Downstream performance");

    # if i == len(metrics) - 1:
    #     ax.legend(
    #         #loc="lower center",
    #         loc="lower left",
    #         #loc="lower right",
    #         ncols=1,
    #         bbox_to_anchor=[0.1, -0.1],
    #     )
    # else:
    # Store the lines and labels from the first plot
    if i == 0:
        lines = plot_result.get_lines()
        labels = df_iter_pivot.columns.tolist()

    ax.get_legend().remove()


# Create a single legend outside the plot
# fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.02, 0.5))
fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.01, 0.5))

fig.suptitle(f"Performance while training for different datasets", y=0.97);

# fig.savefig('samplefigure', bbox_extra_artists=(lgd,), )
plt.savefig(figure_path() / "300B_perf_per_dataset.png", bbox_inches='tight')
plt.show(bbox_inches='tight')
