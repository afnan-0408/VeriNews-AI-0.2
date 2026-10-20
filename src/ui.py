"""Shared Streamlit UI helpers: styling, status banners and Plotly figures. No external assets are loaded."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from . import config
from .prediction import model_status

VIOLET, BLUE, CYAN, RED = "#8b5cf6", "#3b82f6", "#22d3ee", "#f43f5e"

_CSS = """
<style>
:root { --vn-violet:#8b5cf6; --vn-blue:#3b82f6; --vn-cyan:#22d3ee; --vn-card:rgba(20,28,64,.62); --vn-line:rgba(139,92,246,.28); }
.block-container { padding-top: 2rem; max-width: 1150px; }
.vn-hero { position:relative; overflow:hidden; border:1px solid var(--vn-line); border-radius:20px; padding:2.6rem 2rem;
  background: radial-gradient(900px 300px at 15% -10%, rgba(139,92,246,.35), transparent 60%),
              radial-gradient(700px 300px at 100% 110%, rgba(59,130,246,.30), transparent 60%), #0a1030; }
.vn-hero h1 { font-size: clamp(2rem, 5vw, 3.2rem); margin:0 0 .4rem; letter-spacing:-.02em;
  background: linear-gradient(90deg,#c4b5fd,#60a5fa,#22d3ee,#c4b5fd); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: vn-shift 12s linear infinite; }
.vn-hero p { color:#b9c2ec; max-width:46rem; margin:0; font-size:1.05rem; line-height:1.6; }
.vn-scan { position:absolute; left:0; right:0; height:2px; top:0; background:linear-gradient(90deg,transparent,var(--vn-cyan),transparent);
  opacity:.5; animation: vn-scan 6s ease-in-out infinite; }
@keyframes vn-shift { to { background-position: 300% 0; } }
@keyframes vn-scan { 0%,100% { top:0; } 50% { top:100%; } }
.vn-card { background:var(--vn-card); border:1px solid var(--vn-line); border-radius:14px; padding:1.1rem 1.2rem; height:100%;
  transition: transform .2s ease, border-color .2s ease; }
.vn-card:hover { transform: translateY(-2px); border-color: var(--vn-cyan); }
.vn-card h4 { margin:0 0 .35rem; color:#e7eafc; } .vn-card p { margin:0; color:#aab4e0; font-size:.93rem; line-height:1.5; }
.vn-result { border-radius:16px; padding:1.3rem 1.5rem; border:1px solid var(--vn-line); background:var(--vn-card); }
.vn-result .lab { font-size:2.4rem; font-weight:800; letter-spacing:.02em; }
.vn-fake { color:#fb7185; } .vn-real { color:#34d399; }
.vn-muted { color:#9aa5d6; font-size:.9rem; }
.stButton > button, .stDownloadButton > button { transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow:0 6px 18px rgba(139,92,246,.35); }
@media (max-width: 640px) { .vn-hero { padding:1.6rem 1.1rem; } .vn-result .lab { font-size:1.9rem; } }
@media (prefers-reduced-motion: reduce) { .vn-hero h1, .vn-scan { animation:none; } .vn-card, .stButton > button { transition:none; } }
</style>
"""


def setup_page(title: str, icon: str = "🛡️") -> None:
    st.set_page_config(page_title=f"{title} · {config.APP_NAME}", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="vn-hero"><div class="vn-scan"></div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="vn-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>', unsafe_allow_html=True)


def status_banner(status: dict | None = None) -> dict:
    """Show the real model status; returns it so callers can branch."""
    status = status or model_status()
    state = status["state"]
    if state == "missing":
        st.warning("**No trained model yet.** Open **Dataset & Training** to train one on a real dataset "
                   "(for example ISOT). Predictions are disabled until then.")
    elif state == "error":
        st.error(f"The saved model could not be used: {status['message']}")
    elif state == "demo":
        st.warning("**Demo model.** Trained on a small SYNTHETIC dataset for interface testing only. "
                   "Its results and metrics do not reflect real-world fake-news detection.")
    return status


def disclaimer() -> None:
    st.caption("ℹ️ Results are model estimates of writing-style similarity to the training data — "
               "not verified proof that a story is true or false. Scores are uncalibrated.")


def _layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#e7eafc"))
    return fig


def score_figure(score_fake: float, score_real: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=[score_fake, score_real], y=["FAKE score", "REAL score"], orientation="h",
                           marker_color=[RED, "#34d399"], text=[f"{score_fake:.1%}", f"{score_real:.1%}"], textposition="auto"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title="Model scores (uncalibrated)")
    return _layout(fig, 220)


def terms_figure(explanation: dict, predicted: str) -> go.Figure:
    side = explanation["toward_real"] if predicted == "REAL" else explanation["toward_fake"]
    side = list(reversed(side))
    fig = go.Figure(go.Bar(x=[abs(t["contribution"]) for t in side], y=[t["term"] for t in side], orientation="h",
                           marker_color=VIOLET))
    fig.update_layout(title=f"Terms pushing toward {predicted} (tf-idf × coefficient)")
    return _layout(fig, 340)


def confusion_figure(cm, labels) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                               colorscale=[[0, "#0d1430"], [1, VIOLET]], text=cm, texttemplate="%{text}", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Confusion matrix (held-out test set)")
    return _layout(fig, 360)


def distribution_figure(counts: dict, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()),
                           marker_color=[RED if k == "FAKE" else "#34d399" for k in counts]))
    fig.update_layout(title=title)
    return _layout(fig, 300)
