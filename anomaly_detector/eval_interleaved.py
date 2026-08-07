#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import (KNOWN_SYSCALLS, N_BIGRAMS, apply_tfidf,  # noqa: E402
                        make_windows, _undersample_normal)

# Only the build toolchain is dropped (the simulator compiles its payloads at
# startup; those compiler syscalls are collection overhead, not escalation).
# cron is intentionally NOT excluded here — in interleaved data the cron daemon
# is benign activity, correctly labeled normal by cgroup.
BUILD_TOOLCHAIN = {"cc1", "gcc", "cc", "as", "ld", "collect2", "make", "cmake", "ninja"}


def load(path, exclude):
    df = pd.read_csv(path)
    if exclude:
        df = df[~df["comm"].isin(exclude)].reset_index(drop=True)
    return df


def build(df, W, stride, ngram, min_attack, min_frac):
    return make_windows(df, W, stride, per_pid=False, ngram=ngram,
                        min_attack=min_attack, min_attack_frac=min_frac)


def fit_eval(Xtr, ytr, Xte, yte, tfidf, balance_ratio):
    """Fit TF-IDF (train only), balance train, train RF, score test."""
    if tfidf:
        Xtr, Xte, _ = apply_tfidf(Xtr, Xte)
    Xtr_b, ytr_b = _undersample_normal(Xtr, ytr, ratio=balance_ratio)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr_b, ytr_b)
    proba = rf.predict_proba(Xte)[:, 1]
    return rf, proba


def report(yte, proba, thr):
    pred = (proba >= thr).astype(int)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred, zero_division=0)
    f = f1_score(yte, pred, zero_division=0)
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(precision=p, recall=r, f1=f, auc=auc, fpr=fpr,
                tn=tn, fp=fp, fn=fn, tp=tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[50, 100, 150, 200])
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--no-tfidf", action="store_true")
    ap.add_argument("--balance-ratio", type=float, default=2.0)
    ap.add_argument("--min-attack", type=int, default=1)
    ap.add_argument("--min-attack-frac", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    tfidf = not args.no_tfidf

    print("=" * 68)
    print("  Cross-session interleaved evaluation")
    print(f"  train : {os.path.basename(args.train)}")
    print(f"  test  : {os.path.basename(args.test)}")
    print(f"  ngram={args.ngram}  tfidf={tfidf}  balance={args.balance_ratio}:1")
    print(f"  window-label rule: min_attack={args.min_attack}, "
          f"min_frac={args.min_attack_frac}")
    print("=" * 68)

    df_tr = load(args.train, BUILD_TOOLCHAIN)
    df_te = load(args.test, BUILD_TOOLCHAIN)
    print(f"  train rows {len(df_tr):,} | test rows {len(df_te):,} "
          f"(build-toolchain dropped)")

    print(f"\n  {'W':>5} {'stride':>7} {'trainWin':>9} {'testWin':>8} "
          f"{'atk%':>6} {'F1':>7} {'AUC':>7} {'FPR':>7} {'Prec':>7} {'Rec':>7}")
    print("  " + "─" * 74)

    best = None
    for W in args.windows:
        stride = max(1, W // 4)
        Xtr, ytr = build(df_tr, W, stride, args.ngram, args.min_attack, args.min_attack_frac)
        Xte, yte = build(df_te, W, stride, args.ngram, args.min_attack, args.min_attack_frac)
        if len(Xtr) < 20 or len(Xte) < 20 or ytr.sum() == 0 or yte.sum() == 0:
            print(f"  {W:>5} {stride:>7}   (too few windows / one class missing)")
            continue
        _, proba = fit_eval(Xtr, ytr, Xte, yte, tfidf, args.balance_ratio)
        m = report(yte, proba, args.threshold)
        atk_pct = yte.mean() * 100
        print(f"  {W:>5} {stride:>7} {len(Xtr):>9,} {len(Xte):>8,} "
              f"{atk_pct:>5.1f}% {m['f1']:>7.3f} {m['auc']:>7.3f} "
              f"{m['fpr']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f}")
        if best is None or m["f1"] > best[1]["f1"]:
            best = (W, m, stride)

    if best is None:
        print("\n no usable window configuration.")
        return

    W, m, stride = best
    print("\n" + "=" * 68)
    print(f"  BEST (by test F1): W={W}, stride={stride}")
    print("=" * 68)
    print(f"  Attack F1        : {m['f1']:.4f}")
    print(f"  ROC-AUC          : {m['auc']:.4f}")
    print(f"  Precision (atk)  : {m['precision']:.4f}")
    print(f"  Recall (atk)     : {m['recall']:.4f}")
    print(f"  False-pos rate   : {m['fpr']:.4f}  "
          f"({m['fp']:,} of {m['fp']+m['tn']:,} benign windows)")
    print(f"  Confusion (test) : TN={m['tn']:,} FP={m['fp']:,} "
          f"FN={m['fn']:,} TP={m['tp']:,}")


if __name__ == "__main__":
    main()
