from pathlib import Path

import pandas as pd

from figure_utils import load_data, bench_sel

df = load_data()

df.replace(
    {
        'Nemotron-cc-2024-HQ-real-synth-mix': 'Nemotron-cc'
    },
    inplace=True
)
df_all = df.copy()
df_all = df_all[(df_all.n_tokens == "1T") | (df_all.model_name == "open-sci-ref_model-1.7b_data-FineWeb-Edu-1.4T_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_12872088")]

# for all training budget and models, we take the latest value
df_all.sort_values(by="n_iter", inplace=True)

df_pivot = df_all.pivot_table(
    index="model_name",
    columns="benchmark",
    values="value",
    aggfunc="last",
).loc[:, bench_sel]

# we remove checkpoint which are missing an evaluation for any of the selected benchmarks
df_pivot = df_pivot.dropna(axis=0)

rename_dict = {
    "open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-1000B_global_bs-252_context-16384_rotary-1000000_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14904513": "OpenSci-ref-1.7B-nemotron-1T",
    "open-sci-ref_model-1.7b_data-DCLM_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14070018": "OpenSci-ref-1.7B-DCLM-1T",
    "open-sci-ref_model-1.7b_data-FineWeb-Edu-1.4T_tokenizer-GPT-NeoX_samples-1000B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_14066868": "OpenSci-ref-1.7B-FineWeb-Edu-1T",
    "open-sci-ref_model-1.7b_data-FineWeb-Edu-1.4T_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_12872088": "OpenSci-ref-1.7B-FineWeb-Edu-300B"
}

for m in rename_dict.keys():
    assert m in df_all.model_name.unique(), m

df_pivot = df_pivot.loc[rename_dict.keys()]
df_pivot = df_pivot.rename(index=rename_dict)

df_baselines = pd.read_csv(Path(__file__).parent / "data" / "results-baselines.csv.zip")
df_pivot_baselines = df_baselines.pivot_table(index="model_name", columns="benchmark", values="value")
#bench_sel = df_pivot.columns.tolist()

df_pivot_both = pd.concat([df_pivot, df_pivot_baselines]).loc[:, bench_sel]

#commonsense_qa,piqa,arc_challenge,arc_easy,hellaswag,boolq
cols = [
    'copa',
    'lambada_openai',
    'openbookqa',
    'winogrande',
    'mmlu',
    'commonsense_qa',
    'piqa',
    'hellaswag',
    'arc_easy',
    'arc_challenge',
    'boolq',
]
df_pivot_both = df_pivot_both.loc[:, cols]

df_pivot_both["AVG"] = df_pivot_both.mean(axis=1)
df_pivot_both.sort_values(by="AVG", inplace=True, ascending=False)

methods_token = {
    #'Qwen2.5-7B',
    #'EuroLLM-9B',
    #'Qwen2.5-3B',
    'gemma-2-2b': 2,
    'Qwen2.5-1.5B': 18,
    'Qwen2.5-1.5B': 18,
    'OpenSci-ref-1.7B-nemotron-1T': 1,
    'SmolLM2-1.7B': 11,
    'OpenSci-ref-1.7B-DCLM-1T': 1,
    'OpenSci-ref-1.7B-FineWeb-Edu-1T': 1,
    'OpenSci-ref-1.7B-FineWeb-Edu-300B': 0.3,
    'SmolLM-1.7B': 1,
    # 'Qwen2.5-0.5B',
    # TODO check
    'ablation-model-fineweb-edu': 0.3,
    #  'SmolLM2-360M',
    'EuroLLM-1.7B': 4,
    # TODO check
    'ablation-model-c4': 0.3,
}

# copa,openbookqa,lambada_openaiwinogrande,social_iqa
df_pivot_both.rename(columns={
    'copa': 'copa[0]',
    'lambada_openai': "lambada[0]",
    'openbookqa': 'openbookqa[0]',
    'mmlu': 'mmlu[5]',
    'winogrande': 'wino[10]',
    'arc_challenge': 'arc-challenge[10]',
    'arc_easy': 'arc-easy[10]',
    'boolq': 'boolq[10]',
    'commonsense_qa': 'commonsense[10]',
    'hellaswag': 'hellaswag[10]',
    'piqa': 'piqa[10]'
}, inplace=True)

df_pivot_both = df_pivot_both.loc[methods_token.keys()]
df_pivot_both["#Tokens"] = [methods_token[model] for model in df_pivot_both.index]

# put #tokens first
cols = df_pivot_both.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df_pivot_both = df_pivot_both.loc[:, cols]
df_pivot_both.index.name = "model"
print(df_pivot_both.to_string(float_format="%.2f"))
