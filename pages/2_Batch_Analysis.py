import pandas as pd
import streamlit as st

from src import config, database
from src.data_processing import DatasetError, read_csv_safe
from src.prediction import ModelNotAvailable, load_bundle, predict_batch
from src.ui import disclaimer, distribution_figure, hero, setup_page, status_banner

setup_page("Batch Analysis", "📑")
hero("Batch Analysis", "Upload a CSV, choose the text column, and score every row.")
status = status_banner()

up = st.file_uploader(f"CSV file (max {config.MAX_UPLOAD_MB} MB)", type=["csv", "tsv", "txt"])
if up is None:
    st.info("Upload a file to begin. Rows with missing or very short text are skipped and reported, not guessed.")
    st.stop()
try:
    df = read_csv_safe(up, max_rows=config.MAX_BATCH_ROWS + 1)
except DatasetError as exc:
    st.error(str(exc)); st.stop()
if len(df) > config.MAX_BATCH_ROWS:
    st.warning(f"Only the first {config.MAX_BATCH_ROWS:,} rows will be analysed (limit set by VERINEWS_MAX_BATCH_ROWS).")
    df = df.iloc[:config.MAX_BATCH_ROWS]
st.write(f"**{len(df):,} rows**, {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

guess = next((c for c in df.columns if c.lower() in ("text", "content", "article", "body")), df.columns[0])
col = st.selectbox("News text column", list(df.columns), index=list(df.columns).index(guess))
save = st.checkbox("Save metadata of these predictions to history (no text stored)", value=False)

if st.button("Run batch analysis", type="primary", disabled=status["state"] in ("missing", "error")):
    try:
        bundle = load_bundle()
    except ModelNotAvailable as exc:
        st.warning(str(exc)); st.stop()
    bar = st.progress(0.0, text="Scoring…")
    results = predict_batch(bundle, df[col].tolist(), progress=lambda f: bar.progress(f, text="Scoring…"))
    bar.empty()
    out = df.copy()
    out["predicted_label"] = [r["label"] for r in results]
    out["score_real"] = [r["score_real"] for r in results]
    out["score_fake"] = [r["score_fake"] for r in results]
    out["status"] = [r["status"] for r in results]
    ok = out[out["status"] == "ok"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Scored", f"{len(ok):,}"); c2.metric("Skipped", f"{len(out) - len(ok):,}")
    c3.metric("Model", bundle.version)
    if bundle.is_demo:
        st.warning("Scored with the synthetic demo model; do not interpret these results.")
    if len(ok):
        st.plotly_chart(distribution_figure({k: int(v) for k, v in ok["predicted_label"].value_counts().items()},
                                            "Predicted class distribution"), use_container_width=True)
    st.dataframe(out.head(200), use_container_width=True)
    st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"), "verinews_results.csv", "text/csv")
    if save and len(ok):
        try:
            database.add_many(({"label": r["label"], "score_real": r["score_real"], "text": str(t),
                                "model_version": bundle.version, "source": "batch upload"}
                               for r, t in zip(results, df[col]) if r["status"] == "ok"), store_text=False)
        except database.DatabaseError:
            st.warning("Could not save to history.")
    disclaimer()
