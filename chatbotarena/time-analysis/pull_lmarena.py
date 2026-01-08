import pickle
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles problematic Plotly objects gracefully"""

    def find_class(self, module, name):
        # For plotly objects that might cause issues, return a dummy class
        if module.startswith('plotly.graph_objs'):
            # Create a dummy class that accepts any arguments
            class DummyPlotlyObject:
                def __init__(self, *args, **kwargs):
                    pass
            return DummyPlotlyObject

        # For everything else, use normal unpickling
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # Handle persistent IDs if needed
        return super().persistent_load(pid) if hasattr(super(), 'persistent_load') else None


def safe_pickle_load(file_path):
    """Safely load a pickle file, handling Plotly compatibility issues"""
    with open(file_path, 'rb') as f:
        try:
            # First try normal pickle load
            return pickle.load(f)
        except Exception:
            # If that fails, try with SafeUnpickler
            f.seek(0)
            try:
                return SafeUnpickler(f).load()
            except Exception as e:
                raise e


def download_csv_from_lmarena():
    repo_id = "lmarena-ai/lmarena-leaderboard"
    allow_patterns = ["*.csv"]

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Check if there are already csv files in the output directory
    if list(output_dir.glob("*.csv")):
        print(f"CSV files already exist in {output_dir}. Skipping download.")
        return output_dir

    print(f"Downloading CSV files from {repo_id} to {output_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            allow_patterns=allow_patterns, 
            local_dir=output_dir, 
            repo_type="space"
        )
        print(f"Successfully downloaded CSV files to {output_dir}")
    except Exception as e:
        print(f"Error downloading CSV files: {e}")
    
    return output_dir

def download_pickle_from_lmarena():
    repo_id = "lmarena-ai/lmarena-leaderboard"
    allow_patterns = ["*.pkl"]

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Check if there are already pkl files in the output directory
    if list(output_dir.glob("*.pkl")):
        print(f"Pickle files already exist in {output_dir}. Skipping download.")
        return output_dir

    print(f"Downloading Pickle files from {repo_id} to {output_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            allow_patterns=allow_patterns, 
            local_dir=output_dir, 
            repo_type="space"
        )
        print(f"Successfully downloaded Pickle files to {output_dir}")
    except Exception as e:
        print(f"Error downloading Pickle files: {e}")
    
    return output_dir

def load_df():
    data_dir = Path(__file__).parent / "data"
    csv_files = list(data_dir.glob("leaderboard_table_*.csv"))
    
    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame()

    all_dfs = []
    for f in csv_files:
        # Extract date from filename: leaderboard_table_YYYYMMDD.csv
        match = re.search(r"(\d{8})", f.name)
        if match:
            date_str = match.group(1)
            date = datetime.strptime(date_str, "%Y%m%d")
            
            df = pd.read_csv(f)
            df['date'] = date
            all_dfs.append(df)
        else:
            print(f"Could not extract date from filename: {f.name}")

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def load_df_elo(filename: str) -> pd.DataFrame:
    """Load elo ratings from a single pickle file.

    Loads only the leaderboard_table_df from the "full" category.

    Args:
        filename: Path to the pickle file

    Returns:
        DataFrame with columns: model (index), elo columns, and date
    """
    f = Path(filename)

    # Extract date from filename: elo_results_YYYYMMDD.pkl
    match = re.search(r"(\d{8})", f.name)
    if not match:
        print(f"Could not extract date from filename: {f.name}")
        return pd.DataFrame()

    # Load pickle file
    try:
        data = safe_pickle_load(f)
    except Exception as e:
        print(f"Skipping pickle file {f.name}: {str(e)[:100]}")
        return pd.DataFrame()

    # Extract leaderboard_table_df from "full" category
    try:
        # full_data = data["full"]
        # if not isinstance(data, dict):# or 'full' not in data:
        #     return pd.DataFrame()
        #
        # if not isinstance(full_data, dict) or 'leaderboard_table_df' not in full_data:
        #     return pd.DataFrame()

        if "full" in data and "leaderboard_table_df" in data["full"]:
            df = data["full"]["leaderboard_table_df"]
        elif "text" in data and "leaderboard_table_df" in data["text"]["full"]:
            df = data["text"]["full"]["leaderboard_table_df"]
        else:
            df = pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        # Reset index to make model a column
        df = df.reset_index(names="model").copy()

        date_str = match.group(1)
        df['date'] = datetime.strptime(date_str, "%Y%m%d")

        return df

    except (KeyError, TypeError, AttributeError) as e:
        print(f"Could not extract full/leaderboard_table_df from {f.name}: {e}")
        return pd.DataFrame()


def load_all_elo() -> pd.DataFrame:
    """Load all elo ratings from pickle files in the data directory.

    Returns:
        Combined DataFrame with all elo ratings
    """
    data_dir = Path(__file__).parent / "data"
    pkl_files = list(data_dir.glob("elo_results_*.pkl"))

    if not pkl_files:
        print("No Pickle files found.")
        return pd.DataFrame()

    all_dfs = []
    for f in pkl_files:
        df = load_df_elo(f)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    df = load_df_elo(
        "/Users/salinasd/Documents/code/openeurollm/notebooks/chatbotarena/time-analysis/data/elo_results_20250320.pkl"
    )


    download_csv_from_lmarena()
    download_pickle_from_lmarena()

    print("\nLoading Leaderboard CSVs...")
    df = load_df()
    if not df.empty:
        print(f"Loaded Leaderboard DataFrame with {len(df)} rows.")
        print(df.head())

    print("\nLoading Elo Results Pickles...")
    df_elo = load_all_elo()
    if not df_elo.empty:
        print(f"Loaded Elo DataFrame with {len(df_elo)} rows.")
        print(df_elo.date.value_counts())
        #print(df_elo.head())
        #print(df_elo.info())


