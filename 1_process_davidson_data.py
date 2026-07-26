"""
Processes the Davidson et al. (2017) hate speech dataset.

Input:
    datasets/raw/davidson/davidson_data.csv

Output:
    datasets/processed/processed_davidson_data.csv

Output columns:
    text
    class

Label mapping:
    Original -> New
    0 (Hate Speech) -> 2
    1 (Offensive)   -> 1
    2 (Neither)     -> 0
"""

from pathlib import Path
import pandas as pd


# =============================================================================
# File Paths
# =============================================================================

INPUT_FILE = Path("datasets/raw/davidson/davidson_data.csv")
OUTPUT_FILE = Path("datasets/processed/processed_davidson_data.csv")


# =============================================================================
# Functions
# =============================================================================

def process_davidson_dataset(input_file: Path, output_file: Path) -> None:
    """
    Process the Davidson dataset into the project's standard format.

    Parameters
    ----------
    input_file : Path
        Path to the original Davidson CSV.

    output_file : Path
        Destination path for the processed CSV.
    """

    # Load dataset
    df = pd.read_csv(input_file)

    # Keep only required columns
    df = df[["tweet", "class"]]

    # Rename column
    df = df.rename(columns={"tweet": "text"})

    # Remap labels
    label_mapping = {
        0: 2,   # Hate Speech -> Hate Speech
        1: 1,   # Offensive -> Offensive
        2: 0    # Neither -> Non-offensive
    }

    df["class"] = df["class"].map(label_mapping)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(output_file, index=False)

    # Print summary
    print("=" * 50)
    print("Davidson dataset processed successfully.")
    print(f"Input file : {input_file}")
    print(f"Output file: {output_file}")
    print("\nClass distribution:")
    print(df["class"].value_counts().sort_index())
    print("=" * 50)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    process_davidson_dataset(INPUT_FILE, OUTPUT_FILE)