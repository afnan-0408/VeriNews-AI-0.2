import streamlit as st

from src import config
from src.data_processing import (DEFAULT_LABEL_MAP, DatasetError, load_isot_pair, prepare_dataset, read_csv_safe)
from src.demo_data import make_demo_dataframe
from src.training import TrainConfig, TrainingError, train_model
from src.ui import hero, setup_page

setup_page("Dataset & Training", "🧪")
hero("Dataset & Training", "Import a real dataset, map its columns, review the cleaning summary, then train.")
st.caption("Uploaded files are parsed as data only (never executed) and are held in memory, not saved to disk. "
           "Only the trained model and metrics are written to the artifacts folder.")

SRC_ISOT, SRC_CSV, SRC_DEMO = "ISOT (Fake.csv + True.csv)", "Single CSV with a label column", "Synthetic demo (testing only)"
source = st.radio("Dataset source", [SRC_ISOT, SRC_CSV, SRC_DEMO], horizontal=True)
ss = st.session_state
is_demo = source == SRC_DEMO
raw, text_col, title_col, label_col = None, "text", None, "label"

try:
    if source == SRC_ISOT:
        d = config.data_dir()
        local = (d / "Fake.csv").is_file() and (d / "True.csv").is_file()
        use_local = st.checkbox(f"Use Fake.csv / True.csv found in {d.name}/", value=local, disabled=not local)
        if use_local:
            fake_src, true_src = d / "Fake.csv", d / "True.csv"
        else:
            u1, u2 = st.columns(2)
            fake_src = u1.file_uploader("Fake.csv", type=["csv"]); true_src = u2.file_uploader("True.csv", type=["csv"])
        if fake_src is not None and true_src is not None:
            raw = load_isot_pair(fake_src, true_src)
            title_col = "title" if "title" in raw.columns else None
        else:
            st.info("The ISOT Fake News Dataset must be obtained from its official source "
                    "(see About). This app does not download datasets.")
    elif source == SRC_CSV:
        up = st.file_uploader(f"CSV (max {config.MAX_UPLOAD_MB} MB)", type=["csv", "tsv", "txt"])
        if up is not None:
            raw = read_csv_safe(up)
            cols = list(raw.columns)
            m1, m2, m3 = st.columns(3)
            text_col = m1.selectbox("Text column", cols, index=cols.index("text") if "text" in cols else 0)
            tsel = m2.selectbox("Title column (optional)", ["(none)"] + cols)
            title_col = None if tsel == "(none)" else tsel
            label_col = m3.selectbox("Label column", cols, index=cols.index("label") if "label" in cols else min(1, len(cols) - 1))
    else:
        raw, title_col = make_demo_dataframe(), "title"
        st.warning("Synthetic template-generated data. A model trained on it only learns the templates.")
except DatasetError as exc:
    st.error(str(exc)); raw = None

if raw is None:
    st.stop()

st.write(f"**Loaded:** {len(raw):,} rows × {len(raw.columns)} columns")
st.dataframe(raw.head(8), use_container_width=True)

# Label mapping: nothing is assumed silently; unmapped values are dropped and counted.
label_map = None
if source == SRC_CSV:
    vals = sorted({str(v).strip() for v in raw[label_col].dropna().unique()})[:50]
    gf = [v for v in vals if DEFAULT_LABEL_MAP.get(v.lower()) == "FAKE"]
    gr = [v for v in vals if DEFAULT_LABEL_MAP.get(v.lower()) == "REAL"]
    l1, l2 = st.columns(2)
    fv = l1.multiselect("Label values meaning FAKE", vals, default=gf)
    rv = l2.multiselect("Label values meaning REAL", vals, default=gr)
    if set(fv) & set(rv):
        st.error("A value cannot be both FAKE and REAL."); st.stop()
    label_map = {**{v: "FAKE" for v in fv}, **{v: "REAL" for v in rv}}
    st.caption("Numeric labels (0/1) are never auto-mapped — choose what they mean. Unmapped values are dropped and reported.")

o1, o2 = st.columns(2)
min_chars = o1.number_input("Minimum article length (characters)", 1, 1000, 20)
strip_prefix = o2.checkbox("Strip leading wire-agency tags like “CITY (Reuters) -”", value=False,
                           help="ISOT's real articles often start with such tags, which a model can use as a shortcut.")
extra = st.columns(2)
group_col = extra[0].selectbox("Group column (optional, for grouped split)", ["(none)"] + list(raw.columns))
date_col = extra[1].selectbox("Date column (optional, for time-based split)", ["(none)"] + list(raw.columns))
keep = [c for c in (group_col, date_col) if c != "(none)"]

try:
    clean, report = prepare_dataset(raw, text_col, label_col, title_col, label_map, int(min_chars), strip_prefix, keep)
except DatasetError as exc:
    st.error(str(exc)); st.stop()

st.subheader("Cleaning summary")
r = report
a, b, c = st.columns(3)
a.metric("Rows before → after", f"{r.rows_before:,} → {r.rows_after:,}")
b.metric("FAKE / REAL", f"{r.class_counts.get('FAKE', 0):,} / {r.class_counts.get('REAL', 0):,}")
c.metric("Duplicates removed", f"{r.dropped_duplicates:,}")
st.caption(f"Missing text {r.dropped_missing_text:,} · too short {r.dropped_too_short:,} · unknown/missing label "
           f"{r.dropped_unknown_label:,} · label conflicts {r.dropped_label_conflicts:,}"
           + (f" · unmapped values: {', '.join(r.unknown_label_values[:8])}" if r.unknown_label_values else ""))

st.subheader("Training configuration")

with st.form("train_form"):
    t1, t2, t3 = st.columns(3)

    test_size = t1.slider(
        "Test fraction", 0.1, 0.4, 0.2, 0.05
    )

    max_rows = t2.number_input(
        "Max training rows",
        min_value=100,
        max_value=500_000,
        value=40_000,
        step=1000,
        help="Limits the number of usable articles for training."
    )

    cv_folds = t3.number_input(
        "CV folds (0 = off)",
        min_value=0,
        max_value=10,
        value=0
    )

    u1, u2, u3 = st.columns(3)

    max_features = u1.number_input(
        "Max TF-IDF features",
        min_value=1000,
        max_value=200_000,
        value=20_000,
        step=1000
    )

    ngram_hi = u2.selectbox(
        "Max n-gram", [1, 2, 3], index=1
    )

    C = u3.number_input(
        "Regularisation C",
        min_value=0.01,
        max_value=100.0,
        value=1.0
    )

    min_df = st.number_input(
        "Min document frequency",
        min_value=1,
        max_value=50,
        value=1 if is_demo else 2
    )

    options = ["random"] + (["group"] if group_col != "(none)" else []) + (["time"] if date_col != "(none)" else [])
    strategy = st.selectbox("Evaluation split", options,
                            help="Random splits can overestimate performance when related articles are in both sets.")
    go = st.form_submit_button("Train model", type="primary")

if go:
    cfg = TrainConfig(test_size=test_size, max_rows=int(max_rows), cv_folds=int(cv_folds), max_features=int(max_features),
                      ngram_range=(1, int(ngram_hi)), C=float(C), min_df=int(min_df), split_strategy=strategy,
                      group_col=None if group_col == "(none)" else group_col, date_col=None if date_col == "(none)" else date_col)
    bar = st.progress(0.0, text="Starting…")
    try:
        res = train_model(clean, cfg, dataset_info={"name": source, "uses_title": bool(title_col)},
                          cleaning_report=report.to_dict(), is_demo=is_demo,
                          progress=lambda f, msg: bar.progress(f, text=msg))
    except (TrainingError, DatasetError) as exc:
        bar.empty(); st.error(f"Training failed: {exc}")
    except MemoryError:
        bar.empty(); st.error("Out of memory. Lower 'Max training rows' or 'Max TF-IDF features'.")
    except Exception:  # noqa: BLE001
        bar.empty(); st.error("Training failed unexpectedly. Check the dataset and settings, then try again.")
    else:
        bar.empty()
        t = res.metrics["test"]
        st.success(f"Model saved: {res.metadata['model_version']}")
        x, y, z = st.columns(3)
        x.metric("Test accuracy", f"{t['accuracy']:.2%}"); y.metric("Macro F1", f"{t['macro_f1']:.3f}")
        z.metric("Train / test", f"{res.metrics['sizes']['train']:,} / {res.metrics['sizes']['test']:,}")
        st.caption(res.metrics["evaluation_method"] + " See Model Performance for the full report.")
        if is_demo:
            st.warning("Trained on synthetic demo data — these numbers do not reflect real-world performance.")
