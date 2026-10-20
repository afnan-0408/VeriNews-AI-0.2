"""VeriNews-AI - Home page (Streamlit entry point). Run: streamlit run app.py"""
import streamlit as st

from src import config
from src.ui import card, hero, setup_page, status_banner

setup_page("Home")
hero("VeriNews-AI", "A transparent fake-news classifier. TF-IDF features and Logistic Regression, trained by you on "
                    "a real dataset, evaluated on held-out data, with no hidden shortcuts and no invented numbers.")
st.write("")
status = status_banner()

c1, c2 = st.columns([3, 1])
with c2:
    st.page_link("pages/1_News_Analyzer.py", label="Start Analyzing", icon="🔎", use_container_width=True)

st.subheader("Model status")
meta, metrics = status["metadata"], status["metrics"]
if meta and metrics:
    t = metrics["test"]
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{t['accuracy']:.2%}")
    b.metric("Macro F1", f"{t['macro_f1']:.3f}")
    c.metric("Train / test rows", f"{metrics['sizes']['train']:,} / {metrics['sizes']['test']:,}")
    d.metric("Model version", meta["model_version"])
    st.caption(f"Trained {meta['trained_at']} UTC · {metrics['evaluation_method']}")
else:
    st.info("No statistics yet. They appear here after you train a model.")

st.subheader("What you can do")
cols = st.columns(3)
items = [("News Analyzer", "Classify a single article and see which terms drove the score."),
         ("Batch Analysis", "Upload a CSV, score every row and download the results."),
         ("Model Performance", "Inspect real held-out metrics, the confusion matrix and known limitations."),
         ("Prediction History", "Privacy-conscious local history. Full text is never saved unless you opt in."),
         ("Dataset & Training", "Import ISOT or your own CSV, map columns and train a model."),
         ("Methodology", "How it works, and why classification is not fact-checking.")]
for i, (t_, b_) in enumerate(items):
    with cols[i % 3]:
        card(t_, b_)
        st.write("")
st.caption(f"{config.APP_NAME} v{config.APP_VERSION} · Educational ML project. Not a substitute for professional fact-checking.")
