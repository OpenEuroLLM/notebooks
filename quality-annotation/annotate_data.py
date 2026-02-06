"""
Predict nemotron-cc quality labels from propella annotation features.

This script merges propella annotations with nemotron-cc quality labels
and trains a classifier to predict the quality from annotation features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

from utils import prepare_features, train_classifier, set_seed

# Configuration
DATA_ROOT = Path(__file__).parent / "data"
FIGURES_ROOT = Path(__file__).parent / "figures"
MODEL_PATH = DATA_ROOT / "autogluon_models"
LABEL_COL = "quality"

# Columns to exclude (in addition to defaults in utils.py)
EXCLUDE_COLS = ["warc_record_id", "one_sentence_description", "text"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge propella annotations with nemotron-cc quality labels."""
    # Download propella annotations or load locally
    local_file = DATA_ROOT / "annotations-nemotron-cc-10k-sample.parquet"
    if not local_file.exists():
        url = "https://huggingface.co/datasets/openeurollm/propella-annotations/resolve/main/data/propella-1-4b/nemotron-cc-10k-sample/shard000000.parquet"
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(url).to_parquet(local_file)
    df_annotations = pd.read_parquet(local_file)

    # Load nemotron-cc samples to get quality labels
    df_sample = load_dataset(
        "spyysalo/nemotron-cc-10K-sample", columns=["warc_record_id", "label", "text"]
    )["train"].to_pandas()
    df_sample = df_sample.rename(columns={"label": "quality"})

    # Merge annotations with quality labels
    df_annotations = df_annotations.rename(columns={"id": "warc_record_id"})
    df = pd.merge(df_annotations, df_sample, on="warc_record_id")

    # Save text column before dropping for later analysis
    text_series = df["text"].copy()

    return df, text_series


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    """Plot and save confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Quality Nemotron-cc ensemble")
    plt.ylabel("Predicted Quality from Propella features")
    plt.title("Confusion Matrix: Nemotron vs Predicted Quality")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def save_misclassified_texts(y_pred, y_test, test_texts, output_dir: Path):
    """Save misclassified texts for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predicted 0 but actual 4 (under-predicted high quality)
    mask_pred0_actual4 = (y_pred == 0) & (y_test == 4)
    texts_pred0_actual4 = test_texts[mask_pred0_actual4]
    with open(output_dir / "misclassified_pred0_actual4.txt", "w") as f:
        for i, text in enumerate(texts_pred0_actual4):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    print(f"Saved {len(texts_pred0_actual4)} texts with pred=0, actual=4")

    # Predicted 4 but actual 0 (over-predicted low quality)
    mask_pred4_actual0 = (y_pred == 4) & (y_test == 0)
    texts_pred4_actual0 = test_texts[mask_pred4_actual0]
    with open(output_dir / "misclassified_pred4_actual0.txt", "w") as f:
        for i, text in enumerate(texts_pred4_actual0):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    print(f"Saved {len(texts_pred4_actual0)} texts with pred=4, actual=0")


def main():
    set_seed()
    print("Predicting nemotron-cc quality from propella features")
    print("=" * 60)

    # Load data
    df, text_series = load_data()
    print(f"Loaded {len(df)} samples")

    # Prepare features
    df_prepared = prepare_features(df, label_col=LABEL_COL, exclude_cols=EXCLUDE_COLS)
    print(f"Prepared data shape: {df_prepared.shape}")

    # Train/load classifier
    predictor, df_train, df_test = train_classifier(
        df_prepared,
        label_col=LABEL_COL,
        model_path=MODEL_PATH,
        time_limit=120,
        presets="medium_quality",
    )

    # Evaluate
    y_test = df_test[LABEL_COL].values
    y_pred = predictor.predict(df_test.drop(columns=[LABEL_COL]))

    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    labels = sorted(df_prepared[LABEL_COL].unique())
    plot_confusion_matrix(
        y_test, y_pred,
        labels=labels,
        output_path=FIGURES_ROOT / "nemotron_quality_confusion_matrix.pdf"
    )

    # Save misclassified texts for analysis
    test_indices = df_test.index
    test_texts = text_series.loc[test_indices].values
    save_misclassified_texts(y_pred, y_test, test_texts, DATA_ROOT)


if __name__ == "__main__":
    main()
