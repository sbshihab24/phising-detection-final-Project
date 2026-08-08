import numpy as np
import streamlit as st

from config import paths


def render(url_comp, email_comp):
    """Render the Model Performance & Analytics Dashboard."""
    st.subheader("📊 Empirical Model Benchmark Analytics")

    url_tab, email_tab = st.tabs(["🌐 URL Model Suite (5 Models)", "📧 Email Model Suite (5 Models)"])

    with url_tab:
        st.markdown("#### URL Classifier Cross-Validation Benchmark")

        if not url_comp.empty:
            df = url_comp.copy()
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].round(4)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("URL model comparison metrics unavailable.")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if paths.URL_CONF_MATRIX.exists():
                st.image(str(paths.URL_CONF_MATRIX), caption="XGBoost Confusion Matrix", width="stretch")
        with c2:
            if paths.URL_FEAT_IMP.exists():
                st.image(str(paths.URL_FEAT_IMP), caption="Random Forest Feature Importance Weights", width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            if paths.URL_ROC.exists():
                st.image(str(paths.URL_ROC), caption="ROC Curves — All 5 URL Classifiers", width="stretch")
        with c4:
            if paths.URL_PR.exists():
                st.image(str(paths.URL_PR), caption="Precision-Recall Curves — All 5 URL Classifiers", width="stretch")

    with email_tab:
        st.markdown("#### Email NLP Classifier Benchmark")

        if not email_comp.empty:
            df = email_comp.copy()
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].round(4)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("Email model comparison metrics unavailable.")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if paths.EMAIL_CONF_MATRIX.exists():
                st.image(str(paths.EMAIL_CONF_MATRIX), caption="LinearSVM Confusion Matrix", width="stretch")
        with c2:
            if paths.EMAIL_TOP_TERMS.exists():
                st.image(str(paths.EMAIL_TOP_TERMS), caption="Top TF-IDF Phishing Discriminator Words", width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            if paths.EMAIL_ROC.exists():
                st.image(str(paths.EMAIL_ROC), caption="ROC Curves — All 5 Email Classifiers", width="stretch")
        with c4:
            if paths.EMAIL_PR.exists():
                st.image(str(paths.EMAIL_PR), caption="Precision-Recall Curves — All 5 Email Classifiers", width="stretch")
