import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import re
from typing import Tuple, Dict, List
from datetime import datetime

import re
from typing import Tuple, Dict, List
from datetime import datetime

file_path = Path(__file__).parent


"""
I have a log file log.out that looks like this:

10.1.0.137
SLURM_GPUS_PER_NODE:  4
SLURM_JOB_NUM_NODES:  63
NUM_GPUS:  252
MICRO_BATCH_SIZE:  4
GRADIENT_ACCUMULATION_STEPS:  1
GLOBAL_BATCH_SIZE:  1008
SEQUENCE LENGTH:  4096
TOKENS_GLOBAL_BATCH_SIZE:  4128768
TOTAL TOKENS:  1000000000000
TOTAL TOKENS LABEL:  1000B
TRAIN_ITERS:  242204
LR_WARMUP_ITERS:  25000
LR_DECAY_ITERS:  242204
LR_WSD_DECAY_ITERS:  48440
LR_WARMUP_ITERS:  25000
LR:  4e-3
pretrain_gpt.py --num-layers 24 --hidden-size 2048 --ffn-hidden-size 8192 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 4 --global-batch-size 1008 --train-iters 242204 --weight-decay 0.05 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --clip-grad 1.0 --lr-decay-style WSD --lr-warmup-iters 25000 --lr-decay-iters 242204 --lr 4e-3 --min-lr 0 --lr-wsd-decay-style linear --lr-wsd-decay-iters 48440 --data-cache-path /leonardo_scratch/large/userexternal/jjitsev0/MEGATRON_CACHEDIR --use-flash-attn --bf16 --qk-layernorm --tensorboard-dir /leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/tensorboard --ckpt-format torch_dist --position-embedding-type rope --rotary-base 100000 --rotary-percent 1.0 --normalization RMSNorm --norm-epsilon 1e-5 --init-method-std 0.02 --swiglu --distributed-backend nccl --use-distributed-optimizer --overlap-param-gather --overlap-grad-reduce --recompute-activations --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --sequence-parallel --data-path /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_0 /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_1 /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_2 /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_3 /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_4 /leonardo_work/EUHPC_E03_068/shared/datasets/language/tokenized/Nemotron-cc-2024-HQ-real-synth-mix/GPT-NeoX/merged_5 --tokenizer-model /leonardo_work/EUHPC_E03_068/shared/models/EleutherAI/gpt-neox-20b --tokenizer-type HuggingFaceTokenizer --split 989,10,1 --num-workers 4 --log-interval 50 --save-interval 2000 --eval-interval 242204 --log-throughput --save /leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/checkpoints/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_samples-1000B_global_bs-1008_context-4096_rotary-100000_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO --load /leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/checkpoints/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_samples-1000B_global_bs-1008_context-4096_rotary-100000_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO --eval-iters 1 --tensorboard-dir /leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/checkpoints/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_samples-1000B_global_bs-1008_context-4096_rotary-100000_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO/tensorboard

[lrdn1664:0]:lrdn1664:2815228:2816901 [0] NCCL INFO Connected all trees
[lrdn1828:0]:lrdn1828:683528:685202 [0] NCCL INFO Connected all trees
[lrdn0137:0]:Number of parameters in transformer layers in billions:  1.61
[lrdn0137:0]:Number of parameters in embedding layers in billions: 0.10
[lrdn0137:0]:Total number of parameters in billions: 1.71
[lrdn0137:0]:Number of parameters in most loaded shard in billions: 1.7138
[lrdn0137:0]:Theoretical memory footprints: weight and optimizer=9884.48 MB
[lrdn0137:0]:[Rank 0] (after 50 iterations) memory (MB) | allocated: 9909.07763671875 | max allocated: 50277.15625 | reserved: 53386.0 | max reserved: 53386.0
[lrdn3321:3]: [2025-03-31 05:54:07] iteration       50/  242204 | consumed samples:        50400 | elapsed time per iteration (ms): 1552.8 | throughput per GPU (TFLOP/s/GPU): 134.0 | learning rate: 8.000000E-06 | global batch size:  1008 | lm loss: 6.258318E+00 | loss scale: 1.0 | grad norm: 1.520 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 05:55:20] iteration      100/  242204 | consumed samples:       100800 | elapsed time per iteration (ms): 1457.9 | throughput per GPU (TFLOP/s/GPU): 142.7 | learning rate: 1.600000E-05 | global batch size:  1008 | lm loss: 4.536442E+00 | loss scale: 1.0 | grad norm: 2.019 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 05:56:24] iteration      150/  242204 | consumed samples:       151200 | elapsed time per iteration (ms): 1283.2 | throughput per GPU (TFLOP/s/GPU): 162.1 | learning rate: 2.400000E-05 | global batch size:  1008 | lm loss: 4.044766E+00 | loss scale: 1.0 | grad norm: 1.866 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 05:57:25] iteration      200/  242204 | consumed samples:       201600 | elapsed time per iteration (ms): 1211.9 | throughput per GPU (TFLOP/s/GPU): 171.7 | learning rate: 3.200000E-05 | global batch size:  1008 | lm loss: 3.736206E+00 | loss scale: 1.0 | grad norm: 1.143 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 05:58:26] iteration      250/  242204 | consumed samples:       252000 | elapsed time per iteration (ms): 1230.1 | throughput per GPU (TFLOP/s/GPU): 169.1 | learning rate: 4.000000E-05 | global batch size:  1008 | lm loss: 3.510861E+00 | loss scale: 1.0 | grad norm: 1.195 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 05:59:55] iteration      300/  242204 | consumed samples:       302400 | elapsed time per iteration (ms): 1767.4 | throughput per GPU (TFLOP/s/GPU): 117.7 | learning rate: 4.800000E-05 | global batch size:  1008 | lm loss: 3.323009E+00 | loss scale: 1.0 | grad norm: 1.123 | number of skipped iterations:   0 | number of nan iterations:   0 |
[lrdn3321:3]: [2025-03-31 06:00:57] iteration      350/  242204 | consumed samples:       352800 | elapsed time per iteration (ms): 1239.9 | throughput per GPU (TFLOP/s/GPU): 167.8 | learning rate: 5.600000E-05 | global batch size:  1008 | lm loss: 3.181144E+00 | loss scale: 1.0 | grad norm: 0.935 | number of skipped iterations:   0 | number of nan iterations:   0 |

I want you to write a function in python def parse_log(path: str) -> Tuple[dict, list[dict]]

That returns a dictionary and a list of dictionary. The first dictionary should extract all hyperparameters (eg {"SLURM_GPUS_PER_NODE": 4, ..., "LR": 4e-3}). 

The list of dictionary should parse the metrics

eg. a line with 

[lrdn3321:3]: [2025-04-01 05:54:05] iteration    73700/  242204 | consumed samples:     74289600 | elapsed time per iteration (ms): 1147.1 | throughput per GPU (TFLOP/s/GPU): 181.4 | learning rate: 4.000000E-03 | global batch size:  1008 | lm loss: 1.271845E+00 | loss scale: 1.0 | grad norm: 0.018 | number of skipped iterations:   0 | number of nan iterations:   0 |

should add an element to the list {"iteration": 73700, "consumed_samples": 74289600, ...} where the other fields are the elapsed time per iteration, the throughput, the loss etc.
"""
def parse_log_jobid(path: str) -> Tuple[list[str], Dict, List[Dict]]:
    """
    Parse a training log file to extract hyperparameters and metrics.

    Args:
        path: Path to the log file

    Returns:
        Tuple containing:
        - Dictionary of hyperparameters
        - List of dictionaries containing training metrics
    """
    hyperparams = {}
    metrics = []
    lines = []
    # Define the specific hyperparameters we want to extract
    target_hyperparams = {
        'SLURM_GPUS_PER_NODE',
        'SLURM_JOB_NUM_NODES',
        'NUM_GPUS',
        'MICRO_BATCH_SIZE',
        'GRADIENT_ACCUMULATION_STEPS',
        'GLOBAL_BATCH_SIZE',
        'SEQUENCE LENGTH',
        'TOKENS_GLOBAL_BATCH_SIZE',
        'TOTAL TOKENS',
        'TOTAL TOKENS LABEL',
        'TRAIN_ITERS',
        'LR_WARMUP_ITERS',
        'LR_DECAY_ITERS',
        'LR_WSD_DECAY_ITERS',
        'LR'
    }

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            lines.append(line)

            # Parse hyperparameters (simple key: value format)
            if ':' in line and not line.startswith('['):
                # Handle lines like "SLURM_GPUS_PER_NODE:  4"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()

                    # Only parse if it's one of our target hyperparameters
                    if key in target_hyperparams:
                        # Try to convert to appropriate type
                        if value.isdigit():
                            hyperparams[key] = int(value)
                        elif value.replace('.', '').replace('e-', '').replace('E-', '').replace('+', '').replace('-',
                                                                                                                 '').isdigit():
                            # Handle scientific notation and floats
                            try:
                                if 'e' in value.lower():
                                    hyperparams[key] = float(value)
                                else:
                                    hyperparams[key] = float(value)
                            except ValueError:
                                hyperparams[key] = value
                        else:
                            hyperparams[key] = value

            # Parse training metrics lines
            elif line.startswith('[') and 'iteration' in line:
                # Use regex to extract all the metrics from the line
                # Pattern to match the entire metrics line
                pattern = r'\[([^\]]+)\]:\s*\[([^\]]+)\]\s*iteration\s+(\d+)/\s*(\d+)\s*\|\s*consumed samples:\s*(\d+)\s*\|\s*elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|\s*throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)\s*\|\s*learning rate:\s*([\d.E+-]+)\s*\|\s*global batch size:\s*(\d+)\s*\|\s*lm loss:\s*([\d.E+-]+)\s*\|\s*loss scale:\s*([\d.]+)\s*\|\s*grad norm:\s*([\d.]+)\s*\|\s*number of skipped iterations:\s*(\d+)\s*\|\s*number of nan iterations:\s*(\d+)'

                match = re.search(pattern, line)
                if match:
                    node_id = match.group(1)
                    timestamp_str = match.group(2)

                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        timestamp = timestamp_str

                    metric_dict = {
                        'node_id': node_id,
                        'timestamp': timestamp,
                        'iteration': int(match.group(3)),
                        'total_iterations': int(match.group(4)),
                        'consumed_samples': int(match.group(5)),
                        'elapsed_time_per_iteration_ms': float(match.group(6)),
                        'throughput_per_gpu_tflops': float(match.group(7)),
                        'learning_rate': float(match.group(8)),
                        'global_batch_size': int(match.group(9)),
                        'lm_loss': float(match.group(10)),
                        'loss_scale': float(match.group(11)),
                        'grad_norm': float(match.group(12)),
                        'skipped_iterations': int(match.group(13)),
                        'nan_iterations': int(match.group(14))
                    }

                    metrics.append(metric_dict)

    return lines, hyperparams, pd.DataFrame(metrics)

def parse_model_name_jobid(model_name_with_jobid: str):
    if model_name_with_jobid.endswith("_NODE"):
        model_name_with_jobid = model_name_with_jobid.replace("_NODE", "")
    try:
        jobid = int(model_name_with_jobid.split("_")[-1])
    except:
        jobid = None
    model_name = "_".join(model_name_with_jobid.split("_")[:-1])
    return model_name, jobid

def parse_model(model_name) -> tuple[dict, pd.DataFrame]:
    root = file_path / "leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/"

    # load all jobids
    dfs = []

    # sort to ensure jobids appears in increasing order
    paths = sorted([path for path in root.rglob("*.out") if model_name in path.stem])
    for path in paths:
        model_name_path, jobid = parse_model_name_jobid(path.stem)
        if jobid is not None:
            lines, hp_dict, df_metrics = parse_log_jobid(path)
            if len(df_metrics) > 0:
                df_metrics["model_name"] = model_name
                df_metrics["jobid"] = jobid
                df_metrics["path"] = path
                dfs.append(df_metrics)
            else:
                # print(f"No metrics in {path}")
                continue
            # print(f"did not find metric in {path}")
        else:
            print(f"could not parse path jobid: {path}")


    # print(f"Parsed {len(dfs)} jobids")
    if len(dfs) > 0:
        df_metrics = pd.concat(dfs, ignore_index=True)
        df_metrics["cum_runtime_s"] = df_metrics["elapsed_time_per_iteration_ms"].cumsum() / 1000
        df_metrics["cum_total_gputime_s"] = df_metrics["elapsed_time_per_iteration_ms"].cumsum() / 1000 * hp_dict[
            "NUM_GPUS"]

    else:
        print(f"Could not parse model {model_name}")
        df_metrics = None
    return hp_dict, df_metrics
    # merge dataframes


def load_mapping():
    with open(file_path.parent / "plot_results_release/log_dir_name_mapping.jsonl", "r") as f:
        mapping_rows = []
        for line in f:
            mapping_rows.append(json.loads(line))
    return {parse_model_name_jobid(Path(row["log_file_name"]).stem)[0]: row for row in mapping_rows}

def load_all_metrics():
    mapping = load_mapping()

    df_metrics = []
    hp_dicts = {}
    slurm_hp_dicts = {}
    for model_name, hp_config in mapping.items():
        hp_dict, df_model_metrics = parse_model(model_name)
        hp_dicts[model_name] = mapping[model_name]["config"]
        slurm_hp_dicts[model_name] = hp_dict
        # 'open-sci-ref_model-1.7b_data-C4_tokenizer-GPT-NeoX_samples-300B_global_bs-2048_context-2048_schedule-WSD_lr-5e-3_warmup-30000_machine-LEONARDO'
        if df_model_metrics is None:
            print(f"Could not parse model {model_name}")

        df_metrics.append(df_model_metrics)

    # FIX me Could not parse model open-sci-ref_model-1.7b_data-C4_tokenizer-GPT-NeoX_samples-300B_global_bs-2048_context-2048_schedule-WSD_lr-5e-3_warmup-30000_machine-LEONARDO_11627851_NODE
    df = pd.concat(df_metrics, ignore_index=True)
    print(f'Found {len(df["model_name"].unique())} models.')
    return hp_dicts, slurm_hp_dicts, df

if __name__ == "__main__":
    lines, hyperparams, dd = parse_log_jobid(file_path / "leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/open-sci-ref_model-1.7b_data-C4_tokenizer-GPT-NeoX_samples-300B_global_bs-2048_context-2048_schedule-WSD_lr-5e-3_warmup-30000_machine-LEONARDO_11722144.out")
    # hp_dict, df_model_metrics = parse_model('open-sci-ref_model-1.7b_data-C4_tokenizer-GPT-NeoX_samples-300B_global_bs-2048_context-2048_schedule-WSD_lr-5e-3_warmup-30000_machine-LEONARDO')

    hp_dicts, slurm_hp_dicts, df = load_all_metrics()