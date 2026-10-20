#!/usr/bin/env python
"""Train from the command line.

Examples
  python scripts/train_model.py --isot-dir data            # data/Fake.csv + data/True.csv (ISOT)
  python scripts/train_model.py --csv data/news.csv --text-col content --label-col verdict
  python scripts/train_model.py --demo                     # SYNTHETIC data, interface testing only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_processing import DatasetError, load_isot_pair, prepare_dataset, read_csv_safe  # noqa: E402
from src.demo_data import make_demo_dataframe  # noqa: E402
from src.training import TrainConfig, TrainingError, train_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--isot-dir", help="Folder containing Fake.csv and True.csv")
    src.add_argument("--csv", help="Single CSV with text and label columns")
    src.add_argument("--demo", action="store_true", help="Use the SYNTHETIC demo dataset (testing only)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--fake-values", nargs="*", help="Raw label values meaning FAKE (default: fake/false/hoax...)")
    ap.add_argument("--real-values", nargs="*", help="Raw label values meaning REAL (default: real/true/genuine...)")
    ap.add_argument("--strip-agency-prefix", action="store_true", help="Remove leading 'CITY (Reuters) -' tags")
    ap.add_argument("--split", choices=["random", "group", "time"], default="random")
    ap.add_argument("--group-col"); ap.add_argument("--date-col")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=config.MAX_TRAIN_ROWS)
    ap.add_argument("--min-df", type=int, default=None)
    ap.add_argument("--artifact-dir", default=None)
    a = ap.parse_args()

    try:
        keep = [c for c in (a.group_col, a.date_col) if c]
        label_map = None
        if a.fake_values or a.real_values:
            label_map = {**{v: "FAKE" for v in a.fake_values or []}, **{v: "REAL" for v in a.real_values or []}}
        if a.demo:
            raw, name, is_demo = make_demo_dataframe(), "SYNTHETIC demo dataset", True
            text_col, title_col, label_col = "text", "title", "label"
            if a.min_df is None:
                a.min_df = 1
        elif a.isot_dir:
            d = Path(a.isot_dir)
            raw, name, is_demo = load_isot_pair(d / "Fake.csv", d / "True.csv"), "ISOT-style Fake.csv + True.csv", False
            text_col, title_col, label_col = "text", a.title_col or "title", "label"
        else:
            raw, name, is_demo = read_csv_safe(a.csv), Path(a.csv).name, False
            text_col, title_col, label_col = a.text_col, a.title_col, a.label_col
        clean, report = prepare_dataset(raw, text_col, label_col, title_col, label_map,
                                        strip_agency_prefix=a.strip_agency_prefix, keep_cols=keep)
        cfg = TrainConfig(test_size=a.test_size, max_rows=a.max_rows, split_strategy=a.split,
                          group_col=a.group_col, date_col=a.date_col, **({"min_df": a.min_df} if a.min_df else {}))
        res = train_model(clean, cfg, a.artifact_dir, {"name": name, "uses_title": bool(title_col)}, report.to_dict(),
                          is_demo, progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    except (DatasetError, TrainingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    t = res.metrics["test"]
    print(json.dumps({"model_version": res.metadata["model_version"], "evaluation": res.metrics["evaluation_method"],
                      "n_train": res.metrics["sizes"]["train"], "n_test": res.metrics["sizes"]["test"],
                      "accuracy": round(t["accuracy"], 4), "macro_f1": round(t["macro_f1"], 4),
                      "roc_auc": t["roc_auc"], "artifacts": str(res.artifact_dir)}, indent=2))
    if is_demo:
        print("NOTE: trained on SYNTHETIC demo data - metrics say nothing about real-world performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
