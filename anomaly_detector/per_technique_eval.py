#!/usr/bin/env python3
"""

Trains on one interleaved session and, on the held-out session, reports the
detection rate (recall) for each of the 14 attack techniques separately. A
window's "technique" is the majority technique among its attack events (the
labeler tagged each attack event via the wall-clock timeline).


"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import make_windows, apply_tfidf, _undersample_normal  # noqa: E402

BUILD_TOOLCHAIN = {"cc1", "gcc", "cc", "as", "ld", "collect2", "make", "cmake", "ninja"}


def load(path):
    df = pd.read_csv(path)
    df = df[~df["comm"].isin(BUILD_TOOLCHAIN)].reset_index(drop=True)
    # global-mode windows are built in timestamp order; pre-sort once so our
    # per-window technique extraction lines up exactly with make_windows.
    return df.sort_values("timestamp_ns").reset_index(drop=True)


def window_dominant_technique(df, W, stride, n_windows):
    
    tech = df["technique"].fillna("").to_numpy()
    lbl = df["label"].to_numpy()
    out = []
    for k in range(n_windows):
        s = k * stride
        sl_t = tech[s:s + W]
        sl_l = lbl[s:s + W]
        atk = sl_t[sl_l == 1]
        atk = [t for t in atk if t]           # drop empty
        out.append(Counter(atk).most_common(1)[0][0] if atk else "unattributed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--min-attack-frac", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    W, S = args.window, args.stride

    print("=" * 60)
    print("  Per-technique detection (train -> test, cross-session)")
    print(f"  W={W} stride={S} ngram={args.ngram} "
          f"frac={args.min_attack_frac} thr={args.threshold}")
    print("=" * 60)

    df_tr, df_te = load(args.train), load(args.test)

    Xtr, ytr = make_windows(df_tr, W, S, per_pid=False, ngram=args.ngram,
                            min_attack=1, min_attack_frac=args.min_attack_frac)
    Xte, yte = make_windows(df_te, W, S, per_pid=False, ngram=args.ngram,
                            min_attack=1, min_attack_frac=args.min_attack_frac)

    dom = window_dominant_technique(df_te, W, S, len(yte))
    assert len(dom) == len(yte), (len(dom), len(yte))

    Xtr, Xte, _ = apply_tfidf(Xtr, Xte)
    Xb, yb = _undersample_normal(Xtr, ytr, ratio=2.0)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(Xb, yb)
    pred = (rf.predict_proba(Xte)[:, 1] >= args.threshold).astype(int)

    dom = np.array(dom)
    atk = yte == 1
    print(f"\n  Attack windows in test: {int(atk.sum()):,}  "
          f"| overall recall {pred[atk].mean():.3f}\n")
    print(f"  {'technique':<22} {'windows':>8} {'detected':>9} {'recall':>7}")
    print("  " + "─" * 48)
    rows = []
    for t in sorted(set(dom[atk])):
        mask = atk & (dom == t)
        n = int(mask.sum())
        det = int(pred[mask].sum())
        rows.append((t, n, det, det / n if n else 0.0))
    for t, n, det, rec in sorted(rows, key=lambda r: r[3]):
        print(f"  {t:<22} {n:>8,} {det:>9,} {rec:>7.3f}")


if __name__ == "__main__":
    main()
