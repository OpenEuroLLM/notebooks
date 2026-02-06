"""
Predict propella `content_quality` column given other annotation columns.

Goal: Check if we can predict quality from other features, which would:
- Allow getting quality distribution via importance sampling
- Help understand what the quality column "means"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import prepare_features, train_classifier, set_seed, EXCLUDE_COLS

# Configuration
DATA_ROOT = Path(__file__).parent / "data"
FIGURES_ROOT = Path(__file__).parent / "figures"
MODEL_PATH = DATA_ROOT / "autogluon_propella_quality"
LABEL_COL = "content_quality"

# Quality levels in order (for ordinal interpretation)
QUALITY_ORDER = ["unacceptable", "poor", "adequate", "good", "excellent"]


def load_propella_sample(max_rows: int = 10_000) -> pd.DataFrame:
    """Load a sample of propella annotations."""
    # Use the nemotron-cc sample that's already downloaded
    local_file = DATA_ROOT / "annotations-nemotron-cc-10k-sample.parquet"

    if local_file.exists():
        print(f"Loading local file: {local_file}")
        df = pd.read_parquet(local_file)
    else:
        print("Downloading nemotron-cc sample...")
        url = "https://huggingface.co/datasets/openeurollm/propella-annotations/resolve/main/data/propella-1-4b/nemotron-cc-10k-sample/shard000000.parquet"
        df = pd.read_parquet(url)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_file)

    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    print(f"Loaded {len(df)} samples")
    return df


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    """Plot and save confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Quality")
    plt.ylabel("Actual Quality")
    plt.title("Confusion Matrix: Predicting content_quality from other features")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_feature_importance(predictor, df_test, label_col: str, output_path: Path, top_n: int = 20):
    """Plot feature importance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = predictor.feature_importance(df_test, silent=True)
    importance = importance.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance)), importance["importance"].values)
    plt.yticks(range(len(importance)), importance.index)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importance for Predicting content_quality")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance saved to {output_path}")


def main():
    set_seed()

    print("=" * 60)
    print("Predicting propella content_quality from other features")
    print("=" * 60)

    # Load data
    df = load_propella_sample()

    # Show distribution of target
    print(f"\nTarget distribution ({LABEL_COL}):")
    print(df[LABEL_COL].value_counts().sort_index())

    # Prepare features (exclude the target from exclusions)
    exclude_cols = [col for col in EXCLUDE_COLS if col != LABEL_COL]
    df_prepared = prepare_features(df, label_col=LABEL_COL, exclude_cols=exclude_cols)

    print(f"\nPrepared data shape: {df_prepared.shape}")
    print(f"Columns: {list(df_prepared.columns)}")

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
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=QUALITY_ORDER, zero_division=0))

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        labels=QUALITY_ORDER,
        output_path=FIGURES_ROOT / "propella_quality_confusion_matrix.pdf"
    )

    # Plot feature importance
    try:
        plot_feature_importance(
            predictor,
            df_test,
            LABEL_COL,
            output_path=FIGURES_ROOT / "propella_quality_feature_importance.pdf"
        )
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

    # Show AutoGluon leaderboard
    print("\nAutoGluon Leaderboard:")
    print(predictor.leaderboard(df_test, silent=True))


if __name__ == "__main__":
    main()
