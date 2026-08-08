"""
Email text cleaning for TF-IDF classification.
"""

import re


def clean_text(text):
    """
    Clean raw email text so it is ready for TF-IDF vectorization.
    Strips links, addresses, HTML, numbers and punctuation — keeps only words.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove links
    text = re.sub(r"\S+@\S+", " ", text)            # remove email addresses
    text = re.sub(r"<.*?>", " ", text)              # strip HTML tags
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)        # keep letters only
    return re.sub(r"\s+", " ", text).strip()
