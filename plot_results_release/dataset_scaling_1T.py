import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols

df_all = load_data("27-06")
# compute average per iterationdf_sub.loc[:, ["size", "benchmark", "value"]] ".groupby("n_iter").mean()

fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True)

df_plot = df_all.copy()
df_plot.dataset = df_plot.dataset.apply(lambda s: s.replace('Nemotron-cc-2024-HQ-real-synth-mix', 'Nemotron-cc-hq'))
df_plot.dataset = df_plot.dataset.str.lower()
n_tokens = "1T"

size = 1.7

models = [
    #'open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13977373',
    'open-sci-ref_model-1.7b_data-FineWeb-Edu-1.4T_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14066868',
    'open-sci-ref_model-1.7b_data-DCLM_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14070018',
    # "open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13977373",
    'open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX-2_samples-1000B_global_bs-1008_context-4096_rotary-100000_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14917488'
]
df_sub = df_plot[df_plot.model_name.isin(models)].copy()
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
df_iter_pivot.plot(
    ax=ax,
)
ax.grid()
ax.set_title(f"{size}B");
ax.set_xlabel("Number of tokens");
# ax.set_xlabel("Number of iterations");
#ax.set_ylabel(f"Average performance on {len(bench_sel)} tasks");
ax.set_ylabel(f"Average downstream performance");

ax.legend(
    #loc="lower center",
    loc="upper left",
    #loc="lower right",
    ncols=1,
)
fig.suptitle(f"Average performance while training for different datasets", y=0.97);
plt.tight_layout()
plt.show()