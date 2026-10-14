"""
Shared helpers used by several scripts.

Keeping data loading in one place guarantees that baseline.py, ablation_tokenizer.py
and the KcELECTRA notebook all see exactly the same reviews in the same order, so
review_id values and the 80/20 split match across experiments.
"""

import numpy as np
import pandas as pd

SEED = 42  # one random seed for every split and sample in the project


def load_reviews(path):
    """Load the Naver Shopping file (rating<TAB>text), clean it and label it.

    - strips whitespace and drops empty texts
    - drops exact duplicate texts (keeps the first), so a review cannot be in train and test
    - label: 1 = positive (4-5 stars), 0 = negative (1-2 stars); the data has no 3-star reviews
    - review_id: row number after cleaning; used to link reviews across all result files
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["rating", "text"], quoting=3)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = (df["rating"] >= 4).astype(int)
    df["review_id"] = np.arange(len(df))
    return df
