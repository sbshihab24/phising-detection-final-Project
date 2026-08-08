"""
Train Email Phishing Classifiers

Extracts TF-IDF features from cleaned email body text and evaluates 5 machine
learning models (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost).
Saves the best model and vectorizer to models/.

Usage:
    python training/train_email_model.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

DATASET_PATH = Path("data/Cleaned_Phishing_Email.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RANDOM_SEED = 42


def load_dataset(filepath):
    """Load cleaned email text and target label series."""
    df = pd.read_csv(filepath)
    df = df[["Clean_Text", "Email Type"]].copy()

    df["Clean_Text"] = df["Clean_Text"].fillna("").astype(str).str.strip()
    df = df[df["Clean_Text"] != ""]

    df["Email Type"] = pd.to_numeric(df["Email Type"], errors="coerce")
    df = df.dropna(subset=["Email Type"])
    df["Email Type"] = df["Email Type"].astype(int)

    return df[df["Email Type"].isin([0, 1])].drop_duplicates(subset=["Clean_Text", "Email Type"]).reset_index(drop=True)


def get_model_suite(pos_weight):
    """Initialize dictionary of classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced", solver="liblinear"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=40, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_SEED, class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_split=3, min_samples_leaf=1,
            random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
        ),
        "Support Vector Machine": LinearSVC(
            C=1.0, random_state=RANDOM_SEED, class_weight="balanced", max_iter=5000
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, learning_rate=0.08, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
        ),
    }


def get_decision_scores(model, X):
    """Get probability or continuous score array for evaluation curves."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe Email", "Phishing Email"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, scores_map, output_path):
    """Save combined ROC curves plot."""
    plt.figure(figsize=(9, 7))
    for name, scores in scores_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Email Classifiers")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true, scores_map, output_path):
    """Save combined Precision-Recall curves plot."""
    plt.figure(figsize=(9, 7))
    for name, scores in scores_map.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.plot(rec, prec, label=f"{name} (AUC = {auc(rec, prec):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Email Classifiers")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_phishing_terms(lr_model, vectorizer, output_path):
    """Plot top TF-IDF phishing indicator terms from Logistic Regression weights."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]
    top_phish_idx = np.argsort(coefs)[-20:][::-1]
    top_safe_idx = np.argsort(coefs)[:20]

    terms_df = pd.DataFrame({
        "Phishing Term": feature_names[top_phish_idx],
        "Phishing Weight": coefs[top_phish_idx],
        "Safe Term": feature_names[top_safe_idx],
        "Safe Weight": coefs[top_safe_idx],
    })
    terms_df.to_csv(RESULTS_DIR / "email_important_terms.csv", index=False)

    top15 = terms_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top15["Phishing Term"][::-1], top15["Phishing Weight"][::-1])
    plt.xlabel("Logistic Regression Coefficient Weight")
    plt.ylabel("Term")
    plt.title("Top Terms Associated with Phishing Emails")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run python training/clean_emails.py first."
        )

    print(f"Loading cleaned dataset: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"Total valid email records: {len(df):,}")

    X_text = df["Clean_Text"]
    y = df["Email Type"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Fit TF-IDF Vectorizer on training split only
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Train matrix: {X_train.shape} | Test matrix: {X_test.shape}")

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    models = get_model_suite(pos_weight)
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    evaluation_results = []
    trained_models = {}
    score_history = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        cv_scores = cross_val_score(clone(model), X_train, y_train, cv=cv_splitter, scoring="f1", n_jobs=1)
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        decision_scores = get_decision_scores(model, X_test)

        trained_models[name] = model
        score_history[name] = decision_scores

        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, decision_scores)

        evaluation_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, test_preds),
            "Precision": precision_score(y_test, test_preds, zero_division=0),
            "Recall": recall_score(y_test, test_preds, zero_division=0),
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV Mean F1": cv_scores.mean(),
            "CV F1 Std": cv_scores.std(),
            "Training Time Seconds": time.time() - start_time,
        })
        print(f"  F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    comparison_df = (
        pd.DataFrame(evaluation_results)
        .sort_values(["F1 Score", "ROC AUC"], ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.to_csv(RESULTS_DIR / "email_model_comparison.csv", index=False)

    print("\nModel Comparison Table:")
    print(comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].round(4).to_string(index=False))

    best_name = str(comparison_df.iloc[0]["Model"])
    best_model = trained_models[best_name]

    print(f"\nBest Model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["Safe Email", "Phishing Email"]))

    # Save classification reports text
    with open(RESULTS_DIR / "email_classification_reports.txt", "w", encoding="utf-8") as f:
        for name, m in trained_models.items():
            f.write(f"{'=' * 60}\n{name}\n{'=' * 60}\n")
            f.write(classification_report(y_test, m.predict(X_test), target_names=["Safe Email", "Phishing Email"], zero_division=0))
            f.write("\n\n")

    # Generate and save benchmark charts
    plot_confusion_matrix(y_test, best_model.predict(X_test), best_name, RESULTS_DIR / "email_best_model_confusion_matrix.png")
    plot_roc_curves(y_test, score_history, RESULTS_DIR / "email_models_roc_curve.png")
    plot_pr_curves(y_test, score_history, RESULTS_DIR / "email_models_precision_recall_curve.png")
    plot_top_phishing_terms(trained_models["Logistic Regression"], vectorizer, RESULTS_DIR / "email_top_phishing_terms.png")

    # Save model binary and TF-IDF vectorizer
    joblib.dump(best_model, MODELS_DIR / "best_email_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "email_tfidf_vectorizer.pkl")

    best_metrics = comparison_df.iloc[0]
    metadata = {
        "project": "Phishing Detection Using Machine Learning",
        "module": "Email Phishing Classification",
        "best_model": best_name,
        "selection_metric": "F1 Score",
        "accuracy": float(best_metrics["Accuracy"]),
        "precision": float(best_metrics["Precision"]),
        "recall": float(best_metrics["Recall"]),
        "f1_score": float(best_metrics["F1 Score"]),
        "roc_auc": float(best_metrics["ROC AUC"]),
        "cross_validation_mean_f1": float(best_metrics["CV Mean F1"]),
        "cross_validation_f1_std": float(best_metrics["CV F1 Std"]),
        "training_records": int(X_train.shape[0]),
        "testing_records": int(X_test.shape[0]),
        "tfidf_features": int(X_train.shape[1]),
        "safe_email_label": 0,
        "phishing_email_label": 1,
    }
    with open(MODELS_DIR / "email_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nTrained artifacts saved to {MODELS_DIR}/ and evaluation results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
