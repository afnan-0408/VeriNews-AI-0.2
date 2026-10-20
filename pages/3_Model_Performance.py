import json

import pandas as pd
import streamlit as st

from src.ui import confusion_figure, distribution_figure, hero, setup_page, status_banner

setup_page("Model Performance", "📊")
hero("Model Performance", "Real metrics from the saved model, computed on a held-out test set.")
status = status_banner()
meta, m = status["metadata"], status["metrics"]
if not (meta and m):
    st.info("No evaluation data yet. Train a model on the Dataset & Training page."); st.stop()

t = m["test"]
a, b, c, d = st.columns(4)
a.metric("Accuracy", f"{t['accuracy']:.2%}"); b.metric("Macro F1", f"{t['macro_f1']:.3f}")
c.metric("Weighted F1", f"{t['weighted_f1']:.3f}")
d.metric("ROC-AUC", "n/a" if t["roc_auc"] is None else f"{t['roc_auc']:.3f}")
st.caption(f"Model **{meta['model_version']}** · trained {meta['trained_at']} UTC · {meta['algorithm']}")
st.info(f"**Evaluation method:** {m['evaluation_method']}")

left, right = st.columns(2)
left.plotly_chart(confusion_figure(t["confusion_matrix"], t["confusion_labels"]), use_container_width=True)
per = pd.DataFrame(t["per_class"]).T.rename_axis("class").reset_index()
right.subheader("Per-class metrics"); right.dataframe(per, hide_index=True, use_container_width=True)
st.subheader("Classification report"); st.code(t["classification_report"])

st.subheader("Dataset and splits")
s = m["sizes"]; ds = meta["dataset"]
x, y_ = st.columns(2)
x.write(f"**Dataset:** {ds.get('name', 'unknown')}  \n**Rows used:** {ds['rows_used']:,}  \n"
        f"**Train / test:** {s['train']:,} / {s['test']:,}")
if ds.get("subsampled_from"):
    x.caption(f"Subsampled from {ds['subsampled_from']:,} rows to respect the training-size limit.")
if ds.get("cleaning"):
    cl = ds["cleaning"]
    y_.write(f"**Cleaning:** {cl['rows_before']:,} → {cl['rows_after']:,} rows  \n"
             f"Duplicates {cl['dropped_duplicates']:,} · label conflicts {cl['dropped_label_conflicts']:,} · "
             f"unknown labels {cl['dropped_unknown_label']:,} · missing/short {cl['dropped_missing_text'] + cl['dropped_too_short']:,}")
bal = m["class_balance"]
g1, g2 = st.columns(2)
g1.plotly_chart(distribution_figure(bal["train"]["counts"], "Training class balance"), use_container_width=True)
g2.plotly_chart(distribution_figure(bal["test"]["counts"], "Test class balance"), use_container_width=True)

st.subheader("Training vs test (kept separate)")
st.write(f"Training accuracy {m['train']['accuracy']:.2%} · macro F1 {m['train']['macro_f1']:.3f}. {m['train']['note']}")
cv = m.get("cross_validation")
if cv and "error" not in cv:
    st.write(f"**Cross-validation** ({cv['folds']} folds, {cv['on']}): accuracy {cv['accuracy_mean']:.2%} ± {cv['accuracy_std']:.2%}, "
             f"macro F1 {cv['f1_macro_mean']:.3f} ± {cv['f1_macro_std']:.3f}")
elif cv:
    st.caption(cv["error"])
else:
    st.caption("Cross-validation was not run (disabled or too few examples).")
lk = m["leakage"]
st.write(f"**Leakage check:** {lk['exact_duplicates_of_train']} exact and {lk['exact_or_near_duplicates_of_train']} exact-or-near "
         f"duplicate test rows found (cosine ≥ {lk['near_duplicate_threshold']}); {lk['removed_from_test']} removed from the test set.")

st.subheader("Limitations")
for lim in meta["limitations"]:
    st.markdown(f"- {lim}")
st.markdown("- Very high scores on ISOT are common and usually reflect dataset shortcuts (e.g. publisher boilerplate such as "
            "agency tags), not real-world robustness. Consider the grouped/time-based evaluation and external test data.")
with st.expander("Raw metrics JSON"):
    st.code(json.dumps(m, indent=2), language="json")
