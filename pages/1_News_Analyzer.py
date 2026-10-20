import streamlit as st

from src import config, database
from src.prediction import InvalidInput, ModelNotAvailable, load_bundle, predict
from src.ui import disclaimer, hero, score_figure, setup_page, status_banner, terms_figure

setup_page("News Analyzer", "🔎")
hero("News Analyzer", "Paste an article. The model returns FAKE or REAL with the terms that drove the score.")

SAMPLES = {
    "Sample A (sober wire style)": ("Regional council reviews budget",
        "The city council said on Tuesday that officials expect the budget review to conclude next quarter, according to the "
        "published schedule. Spokespeople declined to comment further until the formal documents are released, and a "
        "committee will publish its findings after consultation with independent experts."),
    "Sample B (sensational style)": ("SHOCKING: they are hiding the truth",
        "BREAKING!!! Secret insiders reveal that the government is hiding the truth, share before it is deleted! "
        "This outrageous conspiracy proves everything you were told is a lie, so spread the word immediately before they "
        "silence everyone!"),
}
DEFAULTS = {"an_headline": "", "an_source": "", "an_text": ""}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _load_sample(name):
    st.session_state.an_headline, st.session_state.an_text = SAMPLES[name]
    st.session_state.an_source = "sample text"
    st.session_state.pop("an_result", None)


def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("an_result", None)


status = status_banner()
st.caption("The sample texts are invented for interface testing. A result on them is not evidence of anything.")
sc = st.columns(len(SAMPLES) + 1)
for i, name in enumerate(SAMPLES):
    sc[i].button(name, on_click=_load_sample, args=(name,), use_container_width=True)
sc[-1].button("Reset", on_click=_reset, use_container_width=True)

st.text_input("Headline (optional)", key="an_headline", max_chars=500)
st.text_input("Source (optional, stored in history only — not used by the model)", key="an_source", max_chars=200)
st.text_area("News text", key="an_text", height=260, max_chars=config.MAX_TEXT_CHARS,
             placeholder=f"Paste at least {config.MIN_WORDS} words of article text…")
o1, o2 = st.columns(2)
save = o1.checkbox("Save this prediction to history (metadata only)", value=True)
store_text = o2.checkbox("Also store the full text in history", value=False,
                         help="Off by default for privacy. Stored locally in SQLite.", disabled=not save)

if st.button("Analyze", type="primary", disabled=status["state"] in ("missing", "error")):
    try:
        bundle = load_bundle()
        res = predict(bundle, st.session_state.an_text, st.session_state.an_headline)
    except ModelNotAvailable as exc:
        st.warning(str(exc))
    except InvalidInput as exc:
        st.warning(str(exc))
    except Exception:  # noqa: BLE001 - never leak a stack trace
        st.error("Something went wrong while analysing this text. Please try different text.")
    else:
        st.session_state.an_result = res
        if save:
            try:
                database.add_prediction(res["label"], res["score_real"], st.session_state.an_text.strip(),
                                        st.session_state.an_headline, st.session_state.an_source,
                                        res["model_version"], res["latency_ms"], "single", store_text and save)
            except database.DatabaseError:
                st.warning("The prediction could not be saved to history.")

res = st.session_state.get("an_result")
if res:
    cls = "vn-fake" if res["label"] == "FAKE" else "vn-real"
    st.markdown(f'<div class="vn-result"><div class="vn-muted">Model classification</div>'
                f'<div class="lab {cls}">{res["label"]}</div>'
                f'<div class="vn-muted">Latency {res["latency_ms"]:.1f} ms · model {res["model_version"]}</div></div>',
                unsafe_allow_html=True)
    if res["is_demo"]:
        st.warning("Produced by the synthetic demo model. Do not interpret this result.")
    left, right = st.columns(2)
    left.plotly_chart(score_figure(res["score_fake"], res["score_real"]), use_container_width=True)
    ex = res["explanation"]
    if (ex["toward_real"] if res["label"] == "REAL" else ex["toward_fake"]):
        right.plotly_chart(terms_figure(ex, res["label"]), use_container_width=True)
    else:
        right.info("No single term pushed strongly toward this class.")
    st.caption(f"{ex['known_terms']} distinct terms matched the model vocabulary.")
    disclaimer()
