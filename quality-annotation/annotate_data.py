from pathlib import Path

import ast
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

root = Path(__file__).parent / "data"

# Columns that contain lists (need expansion, special featurization)
LIST_COLS = [
    "content_type",
    "business_sector",
    "technical_content",
    "regional_relevance",
    "country_relevance",
]

# Columns to exclude from features
EXCLUDE_COLS = ["warc_record_id", "one_sentence_description", "text"]


def expand_list_columns(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    """Expand list columns into binary indicator columns."""
    df = df.copy()
    for col in list_cols:
        # Parse string representation if needed
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        # Get all unique values across the column
        all_values = set()
        for values in df[col]:
            all_values.update(values)
        binary_cols = pd.DataFrame(
            {
                f"{col}_{val}": df[col].apply(lambda x, v=val: int(v in x))
                for val in sorted(all_values)
            }
        )
        df = pd.concat([df, binary_cols], axis=1)
        # Drop original list column
        df = df.drop(columns=[col])
    return df


def main():
    print("Hello from quality-annotation!")

    # download propella annotations or load locally
    local_file = root / "annotations-nemotron-cc-10k-sample.parquet"
    if not local_file.exists():
        # TODO not sure why but this is downloading everything
        # from datasets import load_dataset
        # df = load_dataset("openeurollm/propella-annotations", split="nemotron_cc_10k_sample")
        url = "https://huggingface.co/datasets/openeurollm/propella-annotations/resolve/main/data/propella-1-4b/nemotron-cc-10k-sample/shard000000.parquet"
        pd.read_parquet(url).to_parquet(local_file)
    df_annotations = pd.read_parquet(local_file)

    # load local samples of nemotron-cc to get the quality labels
    df_sample = load_dataset(
        "spyysalo/nemotron-cc-10K-sample", columns=["warc_record_id", "label", "text"]
    )["train"].to_pandas()
    df_sample = df_sample.rename(columns={"label": "quality"})

    # rename columns of annotations "id" -> "warc_record_id", "label" -> "quality"
    df_annotations = df_annotations.rename(columns={"id": "warc_record_id"})
    df = pd.merge(df_annotations, df_sample, on="warc_record_id")

    """
    print(df_merged.to_string())
                             warc_record_id content_integrity     content_ratio content_length                                                                                  one_sentence_description                 content_type                    business_sector  technical_content information_density content_quality audience_level commercial_bias time_sensitivity content_safety educational_value reasoning_indicators  pii_presence    regional_relevance country_relevance  quality
0  5e730137-c560-4aeb-92e1-8d98aec5c34e          fragment  complete_content        minimal                                                                A list of timestamps with dates and times.            [structured_data]                            [other]    [non_technical]               dense            poor        general            none        evergreen           safe              none                 none        no_pii       [indeterminate]            [none]        2
1  629cb7e8-9d8b-4aaf-8fec-6cf7d7a8ab2c          complete  complete_content          brief                                   Promotional description of adult video actress Alexis Grace from Miami.              [transactional]              [media_entertainment]    [non_technical]                thin            poor        general  pure_marketing  slowly_changing           nsfw              none                 none        no_pii      [north_american]   [united_states]        1
2  3d38b8f3-dfe4-4522-b6b4-9142b70093e6          complete    mostly_content        minimal      A logical reasoning question asks for the underlying assumption in an argument about job applicants.  [qa_structured, analytical]                 [education_sector]    [non_technical]               dense            good        general            none        evergreen           safe          moderate           analytical        no_pii  [culturally_neutral]            [none]        2
3  0e19ab0e-9973-40ff-a33e-a6b3621f445a          complete  complete_content       moderate  Personal blog post answering a challenge to describe the author's personality traits using the alphabet.                   [creative]                 [general_interest]    [non_technical]            moderate        adequate        general            none        evergreen           safe              none                 none  contains_pii      [north_american]   [united_states]        1
4  6f98c1b4-a896-4da1-a395-921b02dc8812          complete  complete_content        minimal                             Product description for a strawberry, pineapple, and lemon flavored e-liquid.              [transactional]  [retail_commerce, consumer_goods]  [basic_technical]               dense            good        general  pure_marketing  slowly_changing           safe              none                 none        no_pii  [culturally_neutral]            [none]        0
    """
    # Column we want to predict
    label_col = "quality"

    # Save text column before dropping for later analysis
    text_series = df["text"].copy()

    # Expand list columns into binary indicators
    df = expand_list_columns(df, LIST_COLS)

    # Drop columns not useful for prediction
    df = df.drop(columns=EXCLUDE_COLS)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    y_test = df_test[label_col].values

    print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")
    print(f"Features: {[c for c in df_train.columns if c != label_col]}")

    # Load existing model or train new one
    model_path = root / "autogluon_models"
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        predictor = TabularPredictor.load(str(model_path))
    else:
        print(f"Training new model, will save to {model_path}")
        predictor = TabularPredictor(label=label_col, path=str(model_path))
        predictor.fit(df_train, time_limit=600, presets="best")

    # Evaluate
    y_pred = predictor.predict(df_test.drop(columns=[label_col]))
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # print("\nAutoGluon Leaderboard:")
    # print(predictor.leaderboard(df_test))

    # Plot confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(df[label_col].unique())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Quality Nemotron-cc ensemble")
    plt.ylabel("Predicted Quality from Propella features")
    plt.title("Confusion Matrix: Nemotron vs Predicted Quality")
    plt.tight_layout()
    plt.savefig(root / "confusion_matrix.png", dpi=150)
    # plt.show()
    print(f"\nConfusion matrix saved to {root / 'confusion_matrix.png'}")

    # print("\nFeature Importance:")
    # importance = predictor.feature_importance(df_test)
    # print(importance)

    # Save misclassified texts for analysis
    test_indices = df_test.index
    test_texts = text_series.loc[test_indices].values

    # Predicted 0 but actual 4 (under-predicted high quality)
    mask_pred0_actual4 = (y_pred == 0) & (y_test == 4)
    texts_pred0_actual4 = test_texts[mask_pred0_actual4]
    with open(root / "misclassified_pred0_actual4.txt", "w") as f:
        for i, text in enumerate(texts_pred0_actual4):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    print(f"\nSaved {len(texts_pred0_actual4)} texts with pred=0, actual=4 to {root / 'misclassified_pred0_actual4.txt'}")

    # Predicted 4 but actual 0 (over-predicted low quality)
    mask_pred4_actual0 = (y_pred == 4) & (y_test == 0)
    texts_pred4_actual0 = test_texts[mask_pred4_actual0]
    with open(root / "misclassified_pred4_actual0.txt", "w") as f:
        for i, text in enumerate(texts_pred4_actual0):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    print(f"Saved {len(texts_pred4_actual0)} texts with pred=4, actual=0 to {root / 'misclassified_pred4_actual0.txt'}")


# %%


if __name__ == "__main__":
    main()
