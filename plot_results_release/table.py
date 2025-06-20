from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols

df = load_data()

df.replace(
    {
        'Nemotron-cc-2024-HQ-real-synth-mix': 'Nemotron-cc'
    },
    inplace=True
)
df_all = df.copy()
df_all = df_all[df_all.n_tokens == "1T"]

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
    "open-sci-refmodel-1.7bdata-DCLMtokenizer-GPT-NeoXsamples-1000Bglobalbs-1008context-4096schedule-WSDlr-4e-3warmup-25000machine-LEONARDO": "OpenSci-ref-1.7B-DCLM-1T",
    "open-sci-refmodel-1.7bdata-FineWeb-Edu-1.4Ttokenizer-GPT-NeoXsamples-1000Bglobalbs-1008context-4096schedule-WSDlr-4e-3warmup-25000machine-LEONARDO": "OpenSci-ref-1.7B-FineWeb-Edu-1T",
}
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


methods = [
    #'Qwen2.5-7B',
    #'EuroLLM-9B',
    #'Qwen2.5-3B',
    'gemma-2-2b',
    'Qwen2.5-1.5B',
    'OpenSci-ref-1.7B-nemotron-1T',
    'SmolLM2-1.7B',
    'OpenSci-ref-1.7B-DCLM-1T',
    'OpenSci-ref-1.7B-FineWeb-Edu-1T',
    'SmolLM-1.7B',
    # 'Qwen2.5-0.5B',
    'ablation-model-fineweb-edu',
    #  'SmolLM2-360M',
    'EuroLLM-1.7B',
    'ablation-model-c4',
    #   'SmolLM2-135M'
]

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
print(df_pivot_both.loc[methods].to_string(float_format="%.2f"))
