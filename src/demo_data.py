"""SYNTHETIC DEMO DATASET - for testing the interface and test-suite ONLY.

The texts are machine-generated from templates. A model trained on them learns
template wording, NOT how to detect fake news. Its metrics are meaningless as a
statement about real-world performance. Models trained from this data are
flagged ``is_demo`` in their metadata and the UI shows a warning.

CLI:  python -m src.demo_data --out data/demo_dataset.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

_SUBJECTS = ["The city council", "A regional health agency", "The finance ministry", "Local transport authorities",
             "A university research team", "The national statistics office", "The electricity regulator"]
_PLACES = ["in the northern district", "across three provinces", "at the capital's main terminal",
           "in the coastal region", "near the river basin", "at a press conference"]
_REAL_VERBS = ["said on Tuesday that", "reported in a statement that", "confirmed in a briefing that",
               "announced following a review that", "told reporters that"]
_REAL_BODY = ["officials expect the budget review to conclude next quarter, according to the published schedule",
              "the survey of {n} households found modest changes compared with last year's figures",
              "the committee will publish its full findings after consultation with independent experts",
              "spokespeople declined to comment further until the formal documents are released",
              "data released by the agency showed a {n} percent change over the previous period"]
_FAKE_OPEN = ["SHOCKING:", "BREAKING!!!", "You won't believe this:", "EXPOSED:", "THEY don't want you to know:"]
_FAKE_BODY = ["secret insiders reveal that the government is hiding the truth about {n} hidden files, share before it is deleted",
              "a miracle trick that doctors hate was just leaked and the mainstream media is silent, wake up people",
              "anonymous sources claim a massive cover-up involving {n} elites, this is what they are not telling you",
              "totally unbelievable scandal destroys everything, patriots are furious and demand answers right now",
              "this outrageous conspiracy proves everything you were told is a lie, spread the word immediately"]


def make_demo_dataframe(n_per_class: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, seen = [], set()
    for label, builder in (("REAL", _real), ("FAKE", _fake)):
        count, attempts = 0, 0
        while count < n_per_class and attempts < n_per_class * 50:
            attempts += 1
            title, text = builder(rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"title": title, "text": text, "label": label, "source": "SYNTHETIC-DEMO"})
            count += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _real(rng: random.Random):
    s, p = rng.choice(_SUBJECTS), rng.choice(_PLACES)
    body = " ".join(rng.choice(_REAL_BODY).format(n=rng.randint(2, 900)) + "." for _ in range(2))
    return f"{s} {p}", f"{s} {p} {rng.choice(_REAL_VERBS)} {body} Further updates are expected later this week."


def _fake(rng: random.Random):
    o = rng.choice(_FAKE_OPEN)
    body = " ".join(rng.choice(_FAKE_BODY).format(n=rng.randint(2, 900)) + "!" for _ in range(2))
    return f"{o} {rng.choice(_SUBJECTS)} lies again", f"{o} {body} Share this with everyone you know before it disappears!"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_dataset.csv")
    ap.add_argument("--n-per-class", type=int, default=60)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make_demo_dataframe(a.n_per_class).to_csv(a.out, index=False)
    print(f"Wrote SYNTHETIC demo dataset to {a.out}")
