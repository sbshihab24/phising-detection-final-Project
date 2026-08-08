"""
Prediction helper utilities for URL and email model classification.
"""

import math
import numpy as np


def sigmoid(x):
    """Map raw decision scores to probabilities between 0 and 1."""
    clipped = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + math.exp(-clipped))


def get_phishing_prob(model, sample):
    """
    Calculate the phishing class probability (class 1) for an input sample.
    Uses predict_proba if available, otherwise applies sigmoid to decision_function.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return float(probabilities[0][idx])

    if hasattr(model, "decision_function"):
        score = model.decision_function(sample)
        if isinstance(score, np.ndarray):
            score = score.ravel()[0]
        return sigmoid(score)

    return float(model.predict(sample)[0])


def risk_label(probability):
    """Categorize phishing probability into a human-readable risk level."""
    if probability >= 0.85:
        return "Critical Risk"
    if probability >= 0.65:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"
