"""
Merge the processed Davidson dataset and the manual dataset.

Input:
    datasets/processed/processed_davidson_data.csv
    datasets/processed/manual_data.csv

Output:
    datasets/processed/merged_dataset.csv

Output format:
    text,class

Class labels:
    0 = Non-offensive
    1 = Offensive
    2 = Hate Speech
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# File Paths
# =============================================================================

DAVIDSON_FILE = Path("datasets/processed/processed_davidson_data.csv")
MANUAL_FILE = Path("datasets/processed/manual_data.csv")

OUTPUT_FILE = Path("datasets/processed/merged_dataset.csv")


# =============================================================================
# Functions
# =============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""

    df = pd.read_csv(path)

    # Ensure consistent column order
    df = df[["text", "class"]]

    return df


def merge_datasets(*datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Merge datasets, remove duplicate text entries,
    and shuffle the final dataset.
    """

    merged = pd.concat(datasets, ignore_index=True)

    # Remove duplicate sentences
    merged = merged.drop_duplicates(subset="text")

    # Shuffle dataset
    merged = merged.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return merged


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe."""

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


# =============================================================================
# Main
# =============================================================================

def main():

    davidson = load_dataset(DAVIDSON_FILE)
    manual = load_dataset(MANUAL_FILE)

    merged = merge_datasets(
        davidson,
        manual
    )

    save_dataset(
        merged,
        OUTPUT_FILE
    )

    print("=" * 60)
    print("Datasets merged successfully.\n")

    print(f"Davidson samples : {len(davidson):,}")
    print(f"Manual samples   : {len(manual):,}")
    print(f"Total samples    : {len(merged):,}")

    print("\nClass Distribution")
    print(merged["class"].value_counts().sort_index())

    print("=" * 60)


if __name__ == "__main__":
    main()