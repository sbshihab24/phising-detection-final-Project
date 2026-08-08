"""
Build URL Feature Dataset

Processes raw URLs from the PhiUSIIL dataset, extracts lexical features
using features/url_features.py, and saves the training CSV.

Usage:
    python training/build_url_data.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.url_features import FEATURE_COLS, get_url_features

SOURCE_PATH = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_PATH = Path("data/Live_Compatible_URL_Features.csv")


def load_raw_dataset(filepath):
    """Load and clean the raw URL dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath, usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"]).copy()
    
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]
    
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    
    return df[df["label"].isin([0, 1])].drop_duplicates().reset_index(drop=True)


def build_feature_dataset(df):
    """Extract lexical features for each URL in the dataset."""
    records = []
    failed = []

    urls = df["URL"].tolist()
    labels = df["label"].tolist()

    for idx, (url, label) in enumerate(zip(urls, labels)):
        try:
            feat_df = get_url_features(url)
            row_dict = feat_df.iloc[0].to_dict()
            
            row_dict["target"] = 1 - int(label)
            records.append(row_dict)
        except Exception as err:
            failed.append({"index": idx, "url": url, "error": str(err)})

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1:,} / {len(urls):,} URLs")

    output_df = pd.DataFrame(records, columns=FEATURE_COLS + ["target"])
    return output_df.dropna().drop_duplicates(), failed


def main():
    print(f"Loading raw dataset from {SOURCE_PATH}...")
    df = load_raw_dataset(SOURCE_PATH)
    print(f"Found {len(df):,} valid URL records.")

    print("Extracting URL features...")
    start_time = time.time()
    feature_df, errors = build_feature_dataset(df)
    elapsed_min = (time.time() - start_time) / 60

    feature_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeature extraction complete in {elapsed_min:.2f} minutes.")
    print(f"Saved dataset: {OUTPUT_PATH} (Shape: {feature_df.shape})")
    print("Target class distribution:")
    print(feature_df["target"].value_counts().sort_index())

    if errors:
        error_file = Path("results/failed_urls.csv")
        error_file.parent.mkdir(exist_ok=True)
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(f"Logged {len(errors)} failed URLs to {error_file}")


if __name__ == "__main__":
    main()
