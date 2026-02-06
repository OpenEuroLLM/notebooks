"""
Utility functions for quality annotation analysis.
"""

import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split


# Set seeds for reproducibility
SEED = 42


# Columns that contain lists (need expansion to binary indicators)
LIST_COLS = [
    "content_type",
    "business_sector",
    "technical_content",
    "regional_relevance",
    "country_relevance",
]

# Columns to exclude from features
EXCLUDE_COLS = ["id", "warc_record_id", "one_sentence_description", "text"]


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def expand_list_columns(df: pd.DataFrame, list_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Expand list columns into binary indicator columns.

    Args:
        df: DataFrame with list columns
        list_cols: List of column names containing lists. Defaults to LIST_COLS.

    Returns:
        DataFrame with list columns expanded to binary indicators
    """
    if list_cols is None:
        list_cols = [col for col in LIST_COLS if col in df.columns]

    df = df.copy()
    for col in list_cols:
        if col not in df.columns:
            continue

        # Parse string representation if needed
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x)
        )

        # Get all unique values across the column
        all_values = set()
        for values in df[col]:
            if values is not None:
                all_values.update(values)

        # Create binary columns for each unique value
        binary_cols = pd.DataFrame(
            {
                f"{col}_{val}": df[col].apply(lambda x, v=val: int(v in x) if x is not None else 0)
                for val in sorted(all_values)
            }
        )
        df = pd.concat([df, binary_cols], axis=1)

        # Drop original list column
        df = df.drop(columns=[col])

    return df


def prepare_features(
    df: pd.DataFrame,
    label_col: str,
    exclude_cols: list[str] | None = None,
    list_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare features for model training.

    Args:
        df: Input DataFrame
        label_col: Name of the target column
        exclude_cols: Columns to exclude from features. Defaults to EXCLUDE_COLS.
        list_cols: List columns to expand. Defaults to LIST_COLS.

    Returns:
        DataFrame with features prepared for training
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS

    df = df.copy()

    # Expand list columns
    df = expand_list_columns(df, list_cols)

    # Drop excluded columns (if they exist)
    cols_to_drop = [col for col in exclude_cols if col in df.columns and col != label_col]
    df = df.drop(columns=cols_to_drop)

    return df


def train_classifier(
    df: pd.DataFrame,
    label_col: str,
    model_path: str | Path,
    test_size: float = 0.2,
    time_limit: int = 120,
    presets: str = "medium_quality",
    force_retrain: bool = False,
) -> tuple[TabularPredictor, pd.DataFrame, pd.DataFrame]:
    """
    Train an AutoGluon classifier or load existing model.

    Args:
        df: DataFrame with features and label
        label_col: Name of the target column
        model_path: Path to save/load the model
        test_size: Fraction of data to use for testing
        time_limit: Training time limit in seconds
        presets: AutoGluon presets
        force_retrain: If True, retrain even if model exists

    Returns:
        Tuple of (predictor, train_df, test_df)
    """
    set_seed()

    model_path = Path(model_path)

    # Split data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=SEED)

    print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")
    print(f"Features: {[c for c in df_train.columns if c != label_col]}")

    # Load existing model or train new one
    if model_path.exists() and not force_retrain:
        print(f"Loading existing model from {model_path}")
        predictor = TabularPredictor.load(str(model_path))
    else:
        print(f"Training new model, will save to {model_path}")
        predictor = TabularPredictor(label=label_col, path=str(model_path))
        predictor.fit(df_train, time_limit=time_limit, presets=presets)

    return predictor, df_train, df_test
