"""
Preprocess Email Dataset

Loads raw email text from data/Phishing_Email.csv, cleans text content,
encodes target labels (0 = Safe Email, 1 = Phishing Email), and saves Cleaned_Phishing_Email.csv.

Usage:
    python training/clean_emails.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.email_features import clean_text

RAW_PATH = Path("data/Phishing_Email.csv")
OUTPUT_PATH = Path("data/Cleaned_Phishing_Email.csv")


def preprocess_emails(raw_file, output_file):
    """Load, clean, and encode the raw email dataset."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_file}")

    print(f"Loading raw emails from {raw_file}...")
    df = pd.read_csv(raw_file)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Initial dataset shape: {df.shape}")

    # Deduplicate rows
    initial_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {initial_count - len(df)} duplicate rows. ({len(df):,} remaining)")

    # Map string labels to numeric values (0 = Safe, 1 = Phishing)
    label_map = {"Safe Email": 0, "Phishing Email": 1}
    df["Email Type"] = df["Email Type"].map(label_map).fillna(df["Email Type"])
    print("Class label counts:", df["Email Type"].value_counts().to_dict())

    # Pre-clean raw text using email_features.clean_text
    print("Cleaning email text content...")
    df["Clean_Text"] = df["Email Text"].apply(clean_text)

    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset to {output_file} (Shape: {df.shape})")


def main():
    preprocess_emails(RAW_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
