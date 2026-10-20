from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import database
from src.ui import hero, setup_page

setup_page("Prediction History", "🕑")
hero("Prediction History", "Stored locally in SQLite. Full article text is saved only when you explicitly opt in.")

try:
    total = database.count()
except database.DatabaseError as exc:
    st.error(str(exc)); st.stop()

f1, f2, f3 = st.columns([2, 2, 3])
rng = f1.date_input("Date range", value=(date.today() - timedelta(days=30), date.today()))
labels = f2.multiselect("Prediction", ["FAKE", "REAL"], default=["FAKE", "REAL"])
search = f3.text_input("Search source, headline or stored text")
d_from = d_to = None
if isinstance(rng, tuple) and len(rng) == 2:
    d_from, d_to = rng[0].isoformat(), rng[1].isoformat()
try:
    rows = database.query_history(d_from, d_to, labels or None, search, limit=1000)
except database.DatabaseError as exc:
    st.error(str(exc)); st.stop()

st.caption(f"{len(rows):,} matching of {total:,} stored predictions")
if not rows:
    st.info("No records match. Predictions made on the Analyzer page appear here.")
else:
    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("— not stored —")
    st.dataframe(df.drop(columns=["text_hash", "text_stored"]), hide_index=True, use_container_width=True)
    st.download_button("Download filtered history (CSV)", df.drop(columns=["text_hash"]).to_csv(index=False).encode("utf-8"),
                       "verinews_history.csv", "text/csv")
    with st.expander("Delete records"):
        pick = st.multiselect("Delete selected record IDs", [r["id"] for r in rows])
        if st.button("Delete selected", disabled=not pick):
            n = database.delete_ids(pick); st.success(f"Deleted {n} record(s)."); st.rerun()
        st.divider()
        sure = st.checkbox("I understand this permanently deletes ALL history")
        if st.button("Delete all history", type="primary", disabled=not sure):
            n = database.delete_all(); st.success(f"Deleted {n} record(s)."); st.rerun()
