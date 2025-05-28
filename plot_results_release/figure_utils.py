import pandas as pd

bench_sel = [
    'mmlu',
    'copa',
    'lambada_openai',
    'openbookqa',
    'winogrande',
    'arc_challenge',
    'arc_easy',
    'boolq',
    'commonsense_qa',
    'hellaswag',
    'piqa',
    #'social_iqa',
]
hp_cols = [
 'size',
 'dataset',
 'tokenizer',
 'n_tokens',
 'global_batch_size',
 'seq_length',
 'lr_decay_style',
 'lr',
 'lr_warmup_iters']

def load_data():
    df = pd.read_csv("../data/results-22-05.csv.zip")
    df["n_iter"] = df.model_path.apply(lambda x: int(x.split("_")[-1]))
    return df