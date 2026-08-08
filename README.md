# Phishing Detection System Using Machine Learning

An end-to-end machine learning system for real-time classification of phishing URLs and email content. Built as part of a Master's thesis project, this application provides an interactive web console for threat analysis, model benchmarking, and security evaluation.

---

## 📌 Project Overview

Phishing remains one of the primary attack vectors in cyber crime. This system provides dual-layer phishing protection:

1. **URL Structural Threat Analyzer**: Analyzes 18 lexical and structural features extracted directly from URL strings (e.g., URL length, subdomain depth, percent-encoded obfuscation, special character ratios, HTTPS scheme). Operates completely offline without visiting external websites.
2. **Email Content NLP Analyzer**: Pre-processes raw email body text, extracts 10,000 TF-IDF n-gram features (1–2 word grams), and classifies social engineering language patterns using a linear Support Vector Machine (LinearSVM).

---

## 📂 Datasets & Public Sources

Anyone who clones this repository can run the web app immediately using the pre-trained model binaries in `models/` without downloading large raw datasets.

If you wish to re-train the models from scratch, download the raw datasets from the public benchmark sources below and place them in the `data/` directory:

1. **PhiUSIIL Phishing URL Dataset** (~235,000 URLs):
   - File Path: `data/PhiUSIIL_Phishing_URL_Dataset.csv`
   - Public Source: [Kaggle / IEEE DataPort - PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets)
2. **Phishing Email Dataset** (~18,600 Emails):
   - File Path: `data/Phishing_Email.csv`
   - Public Source: [Kaggle - Phishing Email Dataset](https://www.kaggle.com/datasets)

---

## 🏗️ Project Architecture & Directory Structure

```
phising detection final/
├── app.py                  # Streamlit application entry point & tab orchestrator
├── layout.py               # Theme layout & Control Center sidebar status component
├── README.md               # Complete project documentation
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git version control ignore rules
│
├── config/
│   └── paths.py            # Centralized system file & artifact paths
│
├── features/
│   ├── url_features.py     # 18 lexical URL feature extraction & trusted TLD validation
│   └── email_features.py   # Raw email text pre-cleaning & normalization
│
├── frontend/
│   ├── styles.css          # Theme tokens, fonts (Sora, Inter), and component styles
│   └── templates.py        # Reusable HTML cards, hero header, and risk banners
│
├── utils/
│   └── prediction.py       # Probability scoring, decision function mapping & risk levels
│
├── views/
│   ├── home.py             # System Overview & architecture workflow
│   ├── url_checker.py      # Interactive URL Structural Threat Analyzer
│   ├── email_checker.py    # Interactive Email Body NLP Content Analyzer
│   └── results_viewer.py   # Empirical Model Analytics & Benchmark Dashboard
│
├── training/
│   ├── clean_emails.py     # Step 1: Preprocess raw email CSV dataset
│   ├── build_url_data.py   # Step 2: Extract lexical features from raw URL dataset
│   ├── train_email_model.py # Step 3: Train & compare 5 email NLP classifiers
│   └── train_url_model.py  # Step 4: Train & compare 5 URL XGBoost classifiers
│
├── data/                   # Raw and processed datasets (CSV)
├── models/                 # Serialized model binaries (.pkl) & metadata (.json)
└── results/                # Model evaluation charts, ROC curves & benchmark reports
```

---

## 📊 Machine Learning Model Benchmarks

Five classification algorithms were trained, cross-validated (5-fold Stratified K-Fold), and benchmarked for both modules:

### 1. URL Phishing Classification (235,000+ URLs)
- **Selected Model**: **XGBoost Classifier**
- **Selected Features**: 18 Lexical URL Features
- **Balancing Technique**: Training set downsampling (Stratified split)

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-------------------|-----------|--------|----------|---------|
| **XGBoost** | **97.10%** | **97.34%** | **96.85%** | **97.75%** | **97.30%** | **98.94%** |
| Random Forest | 96.85% | 97.02% | 96.42% | 97.58% | 97.00% | 98.71% |
| Decision Tree | 94.20% | 94.50% | 93.80% | 94.90% | 94.35% | 94.50% |
| Logistic Regression | 92.10% | 92.40% | 91.50% | 93.10% | 92.29% | 97.12% |
| Support Vector Machine | 91.80% | 92.05% | 91.10% | 92.90% | 91.99% | 96.85% |

### 2. Email Phishing NLP Classification (18,000+ Emails)
- **Selected Model**: **Linear Support Vector Machine (LinearSVM)**
- **Feature Vectorizer**: TF-IDF (10,000 features, 1–2 word n-grams, English stop-words)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **LinearSVM** | **97.85%** | **97.90%** | **97.94%** | **97.92%** | **99.82%** |
| Logistic Regression | 97.60% | 97.45% | 97.80% | 97.62% | 99.75% |
| XGBoost | 96.90% | 96.80% | 97.10% | 96.95% | 99.45% |
| Random Forest | 96.75% | 96.50% | 97.05% | 96.77% | 99.30% |
| Decision Tree | 92.40% | 92.10% | 92.80% | 92.45% | 92.40% |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 – 3.13
- Virtual environment recommended (`python -m venv .venv`)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/sbshihab24/phising-detection-final.git
cd "phising detection final"
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔁 Model Retraining Pipeline (Optional)

If you wish to re-train the models from raw dataset files:

```bash
# Step 1: Preprocess raw email dataset
python training/clean_emails.py

# Step 2: Extract URL lexical features from raw dataset
python training/build_url_data.py

# Step 3: Train & benchmark email NLP models
python training/train_email_model.py

# Step 4: Train & benchmark URL XGBoost models
python training/train_url_model.py
```

All trained artifacts, metadata JSON files, and benchmark PNG plots will be automatically updated in `models/` and `results/`.

---

## 🔒 Security & Privacy Notice

- **Non-Invasive Analysis**: The URL analyzer performs offline string parsing and feature calculation. No network connections or HTTP requests are made to target websites.
- **Session Local Processing**: Email text submitted through the UI is processed strictly in local application memory and is never logged or stored.
