"""
Phase 3

Text cleaning
Dataset analysis
Train / Validation / Test split
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split


# =============================================================================
# File Paths
# =============================================================================

INPUT_FILE = Path("datasets/processed/merged_dataset.csv")

TRAIN_FILE = Path("datasets/splits/train.csv")
VALID_FILE = Path("datasets/splits/validation.csv")
TEST_FILE = Path("datasets/splits/test.csv")

ASSETS = Path("assets")


# =============================================================================
# Cleaning
# =============================================================================

def clean_text(text: str):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# Statistics
# =============================================================================

def print_statistics(df: pd.DataFrame):

    lengths = df["text"].str.split().apply(len)

    print("=" * 50)

    print(f"Samples : {len(df):,}")

    print()

    print("Class Distribution")

    print(df["class"].value_counts().sort_index())

    print()

    print(f"Average Length : {lengths.mean():.2f}")

    print(f"Maximum Length : {lengths.max()}")

    print(f"Minimum Length : {lengths.min()}")

    print("=" * 50)


# =============================================================================
# Plots
# =============================================================================

def create_plots(df: pd.DataFrame):

    ASSETS.mkdir(exist_ok=True)

    counts = df["class"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))

    counts.plot(kind="bar")

    plt.xticks(
        [0, 1, 2],
        [
            "Non-Offensive",
            "Offensive",
            "Hate Speech"
        ],
        rotation=0
    )

    plt.ylabel("Samples")

    plt.tight_layout()

    plt.savefig(
        ASSETS / "class_distribution.png",
        dpi=300
    )

    plt.close()

    lengths = df["text"].str.split().apply(len)

    plt.figure(figsize=(7, 4))

    plt.hist(lengths, bins=40)

    plt.xlabel("Sentence Length")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        ASSETS / "sentence_lengths.png",
        dpi=300
    )

    plt.close()


# =============================================================================
# Split Dataset
# =============================================================================

def split_dataset(df: pd.DataFrame):

    train, temp = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["class"]
    )

    valid, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        stratify=temp["class"]
    )

    return train, valid, test


# =============================================================================
# Save
# =============================================================================

def save_splits(train, valid, test):

    TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(TRAIN_FILE, index=False)

    valid.to_csv(VALID_FILE, index=False)

    test.to_csv(TEST_FILE, index=False)


# =============================================================================
# Main
# =============================================================================

def main():

    df = pd.read_csv(INPUT_FILE)

    # -------------------------------------------------------------------------
    # Preserve original row number in merged_dataset.csv
    # -------------------------------------------------------------------------
    df = df.reset_index(drop=True)
    df["dataset_row"] = df.index

    # -------------------------------------------------------------------------
    # Clean text
    # -------------------------------------------------------------------------
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Remove duplicate text while preserving dataset_row
    df = df.drop_duplicates(subset="text")

    # Remove empty text
    df = df[df["text"] != ""]

    print_statistics(df)

    create_plots(df)

    train, valid, test = split_dataset(df)

    save_splits(
        train,
        valid,
        test
    )

    print()

    print("Dataset Split")

    print(f"Train      : {len(train):,}")

    print(f"Validation : {len(valid):,}")

    print(f"Test       : {len(test):,}")

    print()

    print("Finished.")


if __name__ == "__main__":
    main()