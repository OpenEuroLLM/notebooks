from pathlib import Path

import ast
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

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
EXCLUDE_COLS = ["warc_record_id", "one_sentence_description"]


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
        "spyysalo/nemotron-cc-10K-sample", columns=["warc_record_id", "label"]
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

    # Expand list columns into binary indicators
    df = expand_list_columns(df, LIST_COLS)

    # Drop columns not useful for prediction
    df = df.drop(columns=EXCLUDE_COLS)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_test = df_test[label_col].values

    print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")
    print(f"Features: {[c for c in df_train.columns if c != label_col]}")

    # Fit model with AutoGluon
    predictor = TabularPredictor(label=label_col, path=root / "autogluon_models")
    predictor.fit(df_train, time_limit=120, presets="medium_quality")

    # Evaluate
    y_pred = predictor.predict(df_test.drop(columns=[label_col]))
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAutoGluon Leaderboard:")
    print(predictor.leaderboard(df_test))

    # Feature importance analysis
    print("\nFeature Importance:")
    importance = predictor.feature_importance(df_test)
    print(importance)


# %%


if __name__ == "__main__":
    main()
