import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols

df_all = load_data()
# compute average per iterationdf_sub.loc[:, ["size", "benchmark", "value"]] ".groupby("n_iter").mean()
sizes = [0.13, 0.4, 1.3, 1.7]
fig, axes = plt.subplots(1, len(sizes), figsize=(11, 4), sharey=True)

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

for i, (ax, size) in enumerate(zip(axes, sizes)):
    df_sub = df_plot.loc[(mask) & (df_plot.loc[:, "size"] == size)].copy()
    df_sub["tokens"] = df_plot["n_iter"] * 1008 * 4096
    df_iter = df_sub.pivot_table(index=["dataset", "tokens"], columns="benchmark", values="value").loc[:, bench_sel].mean(axis=1)

    df_iter_pivot = df_iter.reset_index().pivot_table(index="tokens", columns="dataset", values=0)
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
    df_iter_pivot.columns = [x.capitalize() for x in df_iter_pivot.columns]
    df_iter_pivot = df_iter_pivot.dropna(how="any") 
    df_iter_pivot.plot(
        ax=ax, #marker="."
    )
    ax.grid()
    ax.set_title(f"{size}B");
    ax.set_xlabel("Number of tokens");
    # ax.set_xlabel("Number of iterations");
    #ax.set_ylabel(f"Average performance on {len(bench_sel)} tasks");
    ax.set_ylabel(f"Average downstream performance");
    show_legend_all = False
    if show_legend_all:
        ax.legend(
            #loc="lower center",
            #loc="upper left",
            loc="lower right",
            ncols=1,
        )
    else:
        #if i == len(axes) - 1:
        if i == 0:
            ax.legend(
                #loc="lower center",
                loc="upper left",
                #loc="lower right",
                ncols=1,
            )
        else:
            ax.get_legend().remove()
    ax.set_xlim(0, 3*1e11)
fig.suptitle(f"Performance while training for different datasets", y=0.97);
plt.tight_layout()
plt.show()