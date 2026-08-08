"""
Phishing Detection System — Streamlit App

Run with: streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from config import paths
from frontend.templates import hero_banner
from layout import inject_styles, render_sidebar
from views import email_checker, home, results_viewer, url_checker


st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()


# --- cached resource loading ---

@st.cache_resource
def load_url_model():
    if not paths.URL_MODEL.exists():
        raise FileNotFoundError(f"URL model not found: {paths.URL_MODEL}")
    if not paths.URL_FEAT_COLS.exists():
        raise FileNotFoundError("URL feature columns file not found.")
    return joblib.load(paths.URL_MODEL), joblib.load(paths.URL_FEAT_COLS)


@st.cache_resource
def load_email_model():
    if not paths.EMAIL_MODEL.exists():
        raise FileNotFoundError(f"Email model not found: {paths.EMAIL_MODEL}")
    if not paths.EMAIL_VECTORIZER.exists():
        raise FileNotFoundError("Email TF-IDF vectorizer not found.")
    return joblib.load(paths.EMAIL_MODEL), joblib.load(paths.EMAIL_VECTORIZER)


@st.cache_data
def load_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# --- load resources ---

load_error = None
try:
    url_model, url_feat_cols = load_url_model()
    email_model, email_vec = load_email_model()
except Exception as e:
    load_error = str(e)

url_meta = load_json(paths.URL_META)
email_meta = load_json(paths.EMAIL_META)
url_comp = load_csv(paths.URL_COMPARISON)
email_comp = load_csv(paths.EMAIL_COMPARISON)


# --- hero banner & sidebar ---

st.markdown(
    hero_banner(
        "Phishing Detection System",
        "Real-time URL and email phishing classification with actionable guidance.",
    ),
    unsafe_allow_html=True,
)

render_sidebar(url_meta, email_meta)

if load_error:
    st.error("Error loading model artifacts.")
    st.code(load_error)
    st.stop()


# --- main navigation tabs ---

tab_home, tab_url, tab_email, tab_results = st.tabs([
    "🏠 System Overview",
    "🌐 URL Threat Analyzer",
    "📧 Email Body Analyzer",
    "📊 Model Analytics & Benchmark",
])

with tab_home:
    home.render(url_meta, email_meta)

with tab_url:
    url_checker.render(url_model, url_feat_cols)

with tab_email:
    email_checker.render(email_model, email_vec)

with tab_results:
    results_viewer.render(url_comp, email_comp)


st.markdown(
    '<div class="app-footer">'
    'Academic Prototype — Master\'s Thesis Project | Designed for Offline Cybersecurity Analysis'
    '</div>',
    unsafe_allow_html=True,
)