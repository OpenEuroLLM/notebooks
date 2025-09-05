import pandas as pd
import matplotlib.pyplot as plt

from figure_utils import load_data, bench_sel, hp_cols
import argparse
import os

# Import nice plottiong settings
import settings


def flops(n_tokens_T: float, n_params_B: float):
    return 6 * n_tokens_T * 1e9 * n_params_B * 1e12


def parse_token_budget(s) -> float:
    """'300B = 300 * 1e9; 1T = 1e12."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().upper()
    if s.endswith("T"):
        return float(s[:-1]) * 1e12
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    raise ValueError(f"Unexpected token format for n_tokens: {s}")


def format_latex_scientific(x: float, sig: int = 2) -> str:
    """Return '$m \\cdot 10^{e}$' with 'sig' significant digits for LaTeX."""
    if x == 0:
        return "$0$"
    s = f"{x:.{sig}e}"  # e.g., '1.80e+19'
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exp)  # handles leading '+' and zeros
    return rf"${mantissa} \cdot 10^{{{exp}}}$"


# - group: oellm-core-zero-shot
#     task:
#       - copa
#       - openbookqa
#       - lambada_openai
#       - winogrande
#       - social_iqa
#     num_fewshot: 0
#     aggregate_metric_list:
#       - metric: acc
#   - group: oellm-core-five-shot
#     task:
#       - mmlu
#     num_fewshot: 5
#     aggregate_metric_list:
#       - metric: acc
#     metadata:
#       version: 1.0
#   - group: oellm-core-ten-shot
#     task:
#       - commonsense_qa
#       - piqa
#       - hellaswag
#       - arc_easy
#       - arc_challenge
#       - boolq
#     num_fewshot: 10
#     aggregate_metric_list:
#       - metric: acc

eval_settings = {
    "mmlu": {"num_fewshot": 5},
    "copa": {"num_fewshot": 0},
    "openbookqa": {"num_fewshot": 0},
    "lambada_openai": {"num_fewshot": 0},
    "winogrande": {"num_fewshot": 0},
    "social_iqa": {"num_fewshot": 0},
    "commonsense_qa": {"num_fewshot": 10},
    "piqa": {"num_fewshot": 10},
    "hellaswag": {"num_fewshot": 10},
    "arc_easy": {"num_fewshot": 10},
    "arc_challenge": {"num_fewshot": 10},
    "boolq": {"num_fewshot": 10},
}


colors = [
    "#1F77B4",  # (Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#D62728",  # Red
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#7F7F7F",  # Gray
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
    "#000000",  # Black
    "#3366FF",  # Light Blue
    "#6666FF",  # Light Blue
    "#FF66FF",  # Light Purple
    "#FF9933",  # Light Orange
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling results")
    parser.add_argument(
        "--n_tokens_list",
        type=str,
        nargs="+",
        default=["all"],
        help="List of n_tokens to plot. Default: ['all']. Can be: ['all', '300B', '50B', '1T']",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default="figures",
        help="Path to save the figures. Default: 'figures'",
    )

    parser.add_argument(
        "--figure_ext",
        type=str,
        default=["pdf"],
        nargs="+",
        help="File extension for the saved figures. Default: 'pdf'",
    )

    parser.add_argument(
        "--table_path",
        type=str,
        default="tables",
        help="Path to save the tables. Default: 'tables'",
    )

    parser.add_argument(
        "--table_ext",
        type=str,
        default=["csv"],
        nargs="+",
        help="File extension for the saved tables. Default: 'csv'",
    )

    args = parser.parse_args()

    n_tokens_list = args.n_tokens_list
    figure_path = args.figure_path
    os.makedirs(figure_path, exist_ok=True)
    figure_ext = args.figure_ext

    if not isinstance(figure_ext, list):
        figure_ext = [figure_ext]

    figure_ext = [ext.lower() for ext in figure_ext]
    table_ext = args.table_ext
    table_path = args.table_path
    os.makedirs(table_path, exist_ok=True)
    if not isinstance(table_ext, list):
        table_ext = [table_ext]
    table_ext = [ext.lower() for ext in table_ext]

    baselines_metadata = {
        "EuroLLM-1.7B": {"n_tokens_T": 4, "n_params_B": 1.7},
        "EuroLLM-9B": {"n_tokens_T": 4, "n_params_B": 9},
        "Qwen2.5-0.5B": {"n_params_B": 0.5, "n_tokens_T": 18},
        "Qwen2.5-1.5B": {"n_params_B": 1.5, "n_tokens_T": 18},
        "Qwen2.5-3B": {"n_params_B": 3, "n_tokens_T": 18},
        "Qwen2.5-7B": {"n_params_B": 7, "n_tokens_T": 18},
        "SmolLM2-135M": {"n_params_B": 0.135, "n_tokens_T": 2},
        "SmolLM2-360M": {"n_params_B": 0.36, "n_tokens_T": 4},
        "SmolLM2-1.7B": {"n_params_B": 1.71, "n_tokens_T": 11},
        "gemma-2-2b": {"n_params_B": 2.6, "n_tokens_T": 2},
        "apple/DCLM-7B": {"n_params_B": 7, "n_tokens_T": 4},
        "TRI-ML/DCLM-1B": {"n_params_B": 1.4, "n_tokens_T": 4},
        "ablation-model-fineweb-edu": {"n_params_B": 1.7, "n_tokens_T": 0.35},
        "ablation-model-c4": {"n_params_B": 1.7, "n_tokens_T": 0.35},
    }

    flops_baselines = {
        "EuroLLM-1.7B": flops(n_tokens_T=1.7, n_params_B=4),
        "EuroLLM-9B": flops(n_tokens_T=9, n_params_B=4),
        # Not sure which one is correct
        # "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens"
        # https://qwenlm.github.io/blog/qwen2.5/
        # We are not authorized to share the details right now but the rough number is over 3T tokens for Qwen1.5 and over
        # 7T tokens for Qwen2.
        # https://github.com/QwenLM/Qwen3/issues/562
        "Qwen2.5-0.5B": flops(n_params_B=0.5, n_tokens_T=18),
        "Qwen2.5-1.5B": flops(n_params_B=1.5, n_tokens_T=18),
        "Qwen2.5-3B": flops(n_params_B=3, n_tokens_T=18),
        "Qwen2.5-7B": flops(n_params_B=7, n_tokens_T=18),
        # https://huggingface.co/HuggingFaceTB/SmolLM2-135M
        "SmolLM2-135M": flops(n_params_B=0.135, n_tokens_T=2),
        # https://huggingface.co/HuggingFaceTB/SmolLM2-360M
        "SmolLM2-360M": flops(n_params_B=0.36, n_tokens_T=4),
        # https://arxiv.org/pdf/2502.02737v1 for SmolLM2 numbers
        "SmolLM2-1.7B": flops(n_params_B=1.71, n_tokens_T=11),
        # https://arxiv.org/html/2408.00118v1 We train Gemma 2 27B on 13 trillion tokens of primarily-English data,
        # the 9B model on 8 trillion tokens, and the 2B on 2 trillion tokens
        "gemma-2-2b": flops(n_params_B=2.6, n_tokens_T=2),
        "apple/DCLM-7B": flops(n_params_B=7, n_tokens_T=4),
        "TRI-ML/DCLM-1B": flops(n_params_B=1.4, n_tokens_T=4),
        "ablation-model-fineweb-edu": flops(n_params_B=1.7, n_tokens_T=0.35),
        "ablation-model-c4": flops(n_params_B=1.7, n_tokens_T=0.35),
    }

    x_col = "Training FLOPs"
    y_col = "Average downstream performance"
    df = load_data()

    df.replace({"Nemotron-cc-2024-HQ-real-synth-mix": "Nemotron-cc-hq"}, inplace=True)

    # read results from baseline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_baselines = pd.read_csv(
        os.path.join(current_dir, "data/baselines-24-07.csv.zip")
    )

    print(df_baselines.model_name.unique())
    df_baselines_pivot = df_baselines.pivot_table(
        index=["model_name"], columns="benchmark", values="value"
    ).loc[:, bench_sel]

    baselines_avg = df_baselines_pivot.mean(axis=1)
    df_baselines_flops_perf = pd.DataFrame(
        {x_col: flops_baselines, "Avg performance": baselines_avg}
    )
    df_baselines_pivot["average"] = df_baselines_pivot.mean(axis=1)
    df_baselines_pivot["dataset"] = df_baselines_pivot.index.get_level_values(
        "model_name"
    )

    for n_tokens in n_tokens_list:
        df_all = df.copy()
        df_all.dataset = df_all.dataset.apply(
            lambda x: "DCLM-open-sci" if "DCLM" in x else x
        )
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
        df_hp = (
            df_all.drop_duplicates("model_name")
            .set_index("model_name")
            .loc[df_avg.model_name, hp_cols]
        )
        df_avg = pd.concat([df_avg.set_index("model_name"), df_hp], axis=1)
        n_tokens_mapping = {"300B": 300 * 1e9, "50B": 50 * 1e9, "1T": 1e12}
        df_avg[x_col] = (
            6
            * df_avg["n_tokens"].apply(lambda x: n_tokens_mapping[x])
            * df_avg["size"]
            * 1e9
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        # dd = df_avg.pivot_table(
        #     index=x_col,
        #     values=y_col,
        #     columns="dataset",
        # )
        

        df = (
            df.set_index("model_name")
            .join(
                df_avg[["Average downstream performance"]]
                .reset_index()
                .set_index("model_name"),
                how="left",
            )
            .reset_index()
        )

        df = df.loc[
            df.groupby(["dataset", "benchmark", "n_tokens", "size", "model_name"])[
                "n_iter"
            ].idxmax()
        ]

        df_os = (
            df.loc[
                df.groupby(["dataset", "benchmark", "n_tokens", "size"])[
                    "Average downstream performance"
                ].idxmax()
            ][
                [
                    "dataset",
                    "benchmark",
                    "n_tokens",
                    "size",
                    "value",
                    "Average downstream performance",
                ]
            ]
            .pivot(
                index=["dataset", "n_tokens", "size", "Average downstream performance"],
                columns="benchmark",
                values="value",
            )
            .reset_index()
        )

        df_os = df_os.dropna(axis=0)

        df_os = df_os.rename(
            columns={
                "Average downstream performance": "average",
            }
        )

        # Compute = 6 * N * D, where N is parameters (absolute count), D is tokens (absolute count)
        df_os["Compute (FLOPS)"] = (
            6
            * df_os["size"].astype(float)
            * 1e9
            * df_os["n_tokens"].apply(parse_token_budget)
        )
        
        df_os.dataset = df_os.dataset.apply(
            lambda x: "DCLM-open-sci" if "DCLM" in x else x
        )

        df_os = df_os[~df_os[["dataset","n_tokens"]].apply(lambda x: x[0]=="C4" and x[1]=="50B", axis=1)]

        dd = df_os.pivot_table(
            index="Compute (FLOPS)",
            values="average",
            columns="dataset",
        )

        df_baselines_pivot = df_baselines_pivot.reset_index()
        df_baselines_pivot["n_tokens"] = df_baselines_pivot["model_name"].apply(
            lambda x: baselines_metadata[x]["n_tokens_T"]
            if x in baselines_metadata
            else None
        )
        df_baselines_pivot["size"] = df_baselines_pivot["model_name"].apply(
            lambda x: baselines_metadata[x]["n_params_B"]
            if x in baselines_metadata
            else None
        )
        df_baselines_pivot = df_baselines_pivot.dropna(
            subset=["n_tokens", "size"]
        ).drop(columns=["model_name"])
        df_baselines_pivot["n_tokens"] = df_baselines_pivot["n_tokens"].apply(
            lambda x: f"{x}T" if x >= 1 else f"{x * 1000}B"
        )
        df_merged = pd.concat([df_os, df_baselines_pivot], axis=0).sort_values(
            by=["average"]
        )

        results_table = df_merged

        results_table[bench_sel] = results_table[bench_sel].apply(
            lambda x: x.round(3) if x.dtype == "float64" else x
        )

        results_table["average"] = results_table["average"].apply(
            lambda x: round(x, 3) if isinstance(x, float) else x
        )

        columns_to_rename = {
            "dataset": "Model",
            "n_tokens": "Training tokens",
            "size": "Parameters (B)",
            "average": "Average performance",
        }

        benchmarks_settings = {
            bench: f'{bench}[{eval_settings[bench]["num_fewshot"]}]'
            for bench in bench_sel
        }
        columns_to_rename.update(benchmarks_settings)

        results_table = results_table.rename(columns=columns_to_rename)

        # Compute = 6 * N * D, where N is parameters (absolute count), D is tokens (absolute count)
        results_table["Compute (FLOPS)"] = (
            6
            * results_table["Parameters (B)"].astype(float)
            * 1e9
            * results_table["Training tokens"].apply(parse_token_budget)
        )

        # Add compute FLOPS column
        # results_table = results_table[
        #     [
        #         "Model",
        #         "Training tokens",
        #         "Parameters (B)",
        #         "Compute (FLOPS)",
        #         "Average performance",
        #     ]
        # ]
        # order by few shot benchmarks
        bench_sel_order = sorted(
            list(benchmarks_settings.values()),
            key=lambda x: int(x.split("[")[1].split("]")[0]),
        )

        col_order = (
            [
                "Model",
                "Training tokens",
                "Parameters (B)",
                "Compute (FLOPS)",
            ]
            + ["Average performance"]
            + bench_sel_order
        )

        results_table = results_table[col_order]

        for ext in table_ext:
            print(f"Saving results table for {n_tokens} tokens in {ext} format")
            if ext == "csv":
                print(results_table.columns)
                results_table["Compute (FLOPS)"] = results_table[
                    "Compute (FLOPS)"
                ].apply(
                    lambda x: f"{x:.2e}"  # format as scientific notation
                )
                results_table.to_csv(
                    os.path.join(table_path, f"scaling_{n_tokens}.{ext}"), index=False
                )
            elif ext == "latex":
                # Format the compute column for LaTeX
                rt = results_table.copy()
                rt["Compute (FLOPS)"] = rt["Compute (FLOPS)"].apply(
                    lambda x: f"{format_latex_scientific(float(x))}"  # format as scientific notation
                )
                rt.to_latex(
                    os.path.join(table_path, f"scaling_{n_tokens}.{ext}"),
                    index=False,
                    float_format="%.2f",
                    escape=False,  # allow LaTeX math strings
                )
            elif ext == "html":
                results_table.to_html(
                    os.path.join(table_path, f"scaling_{n_tokens}.{ext}"),
                    index=False,
                    float_format="%.2f",
                )
            else:
                print(
                    f"[WARNING] Unsupported table format: {ext}. Supported formats are: csv, latex, html."
                )

        dataset_order = [
          "Nemotron-cc-hq",
          "DCLM-open-sci",
          "HPLT-2.0",
          "FineWeb-Edu-1.4T",
          "Pile",
          "SlimPajama",
          "CommonCorpus",
          "C4",
        ]
        # Reorder dataset 
        for col, color in zip(dataset_order, colors):
            dd_plot = dd.loc[:, col].dropna()
            ax = dd_plot.plot(ax=ax, label=col, marker="*", color=color)
            if "Nemotron" in col:
                for (x, y), s in zip(
                    dd_plot.reset_index().values,
                    ["0.13B", "0.4B", "1.3B", "1.7B", "1.7B (1T)"],
                ):
                    ax.text(
                        x=x * 0.68,
                        y=y * 1.04,
                        s=s,
                        # color=plt.gca().lines[-1].get_color()
                        color="black",
                    )

        baselines = ["Qwen2.5", "SmolLM2", "EuroLLM", "DCLM"]

        for baseline, color in zip(baselines, colors[len(dd.columns) :]):
            df_baseline = df_baselines_flops_perf[
                df_baselines_flops_perf.index.str.contains(baseline)
            ].copy()
            df_baseline.sort_values(x_col, inplace=True)
            df_baseline.plot(
                x=x_col,
                y="Avg performance",
                ax=ax,
                label=baseline,
                ls="--",
                marker="o",
                color=color,
            )

            def _n_params(method: str):
                return method.split("-")[-1]

            # TODO customize offset
            sizes = [_n_params(t) for t in df_baseline.index]
            if baseline == "EuroLLM":
                offset = (1.2, 1.0)
            elif "Qwen" in baseline:
                offset = (1.15, 0.99)
            elif "DCLM" in baseline:
                offset = (0.9, 1.02)
            else:
                offset = (0.68, 0.94)
            for (x, y), s in zip(df_baseline.values, sizes):
                ax.text(
                    x=x * offset[0],
                    y=y * offset[1],
                    s=s,
                    color=plt.gca().lines[-1].get_color(),
                )

        ax.set_xscale("log")
        ax.set_ylabel(y_col)
        ax.grid(visible=True)

        # first legend: OpenSci
        handles, labels = ax.get_legend_handles_labels()
        open_ref_order = dataset_order
        ext_order = baselines

        open_ref = [(h,l) for h,l in zip(handles,labels) if l in open_ref_order]
        ext = [(h,l) for h,l in zip(handles,labels) if l in ext_order]
        
        # replace 'DCLM-open-sci' with 'DCLM'
        open_ref = [(h,l) if l!='DCLM-open-sci' else (h,'DCLM') for h,l in open_ref]

        leg1 = ax.legend(
            [h for h,_ in open_ref],
            [l for _,l in open_ref],
            title="Open-sci-ref-0.01",
            loc="center left",
            bbox_to_anchor=(1.0,0.66)
        )
        leg1._legend_box.align = "left"   # left-align title
        ax.add_artist(leg1)  # keep it when adding second legend

        # second legend: External
        leg2 = ax.legend(
            [h for h,_ in ext],
            [l for _,l in ext],
            title="External Models",
            loc="center left",
            bbox_to_anchor=(1.0,0.2)
        )


        # ax.legend(
        #     ncols=1,
        #     loc="center left",
        #     bbox_to_anchor=(1.0, 0.5),
        # )
        
        # ax.set_title(f"Scaling comparison with reference models trained on 300B and 1T tokens");
        plt.tight_layout()

        for ext in figure_ext:
            fig.savefig(
                os.path.join(figure_path, f"scaling_{n_tokens}.{ext}"),
                bbox_inches="tight",
                dpi=300,
                bbox_extra_artists=(leg1, leg2),
            )
            print(
                f"Figure saved to {os.path.join(figure_path, f'scaling_{n_tokens}.{ext}')}"
            )
        plt.close(fig)
