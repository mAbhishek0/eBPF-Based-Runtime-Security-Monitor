#!/usr/bin/env python3
"""
revision_eval.py — feature ablation and parameter-selection analyses.

(A) Feature ablation: cross-session attack-class F1/FPR for unigram only,
    unigram+bigram, and unigram+bigram+TF-IDF (the shipped model).

(B) Window-size CV: window-size sweep scored by StratifiedKFold within each
    session separately, so W=200 is justified on the training session without
    consulting the held-out test session.

Reuses the global windowing from loto_eval and the TF-IDF transform from
bosc_train, so numbers match the main pipeline (W=200, stride=50, frac=0.10).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loto_eval import build_windows, _USED
from bosc_train import apply_tfidf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

S1 = "data_collector/interleaved/labeled_session1.csv"
S2 = "data_collector/interleaved/labeled_session2.csv"
W, STRIDE, FRAC, THR = 200, 50, 0.10, 0.5


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def fpr(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    benign = y == 0
    return float((pred[benign] == 1).sum() / max(benign.sum(), 1))


def metrics(ytr, proba, yte):
    pred = proba > THR
    return (precision_score(yte, pred, zero_division=0),
            recall_score(yte, pred, zero_division=0),
            f1_score(yte, pred, zero_division=0),
            fpr(yte, pred))


def undersample(X, y, ratio=2, seed=42):
    na = int(y.sum())
    nb = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    keep_b = rng.choice(nb, min(len(nb), ratio * na), replace=False)
    keep = np.sort(np.concatenate([keep_b, np.where(y == 1)[0]]))
    return X[keep], y[keep]


def ablation(df1, df2):
    print("\n(A) Feature ablation  (train S1 -> test S2, attack class, thr 0.5,"
          " benign undersampled 2:1 in training)")
    print(f"{'representation':<22}{'dim':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}{'FPR':>8}")
    print("-" * 59)
    for name, ngram, use_tfidf in [("unigram only", 1, False),
                                   ("unigram+bigram", 2, False),
                                   ("unigram+bigram+TFIDF", 2, True)]:
        X1, f1, _ = build_windows(df1, W, STRIDE, ngram)
        X2, f2, _ = build_windows(df2, W, STRIDE, ngram)
        y1 = (f1 >= FRAC).astype(int)
        y2 = (f2 >= FRAC).astype(int)
        X1, y1 = undersample(X1, y1)                 # match the shipped pipeline
        if use_tfidf:
            X1t, X2t, _ = apply_tfidf(X1, X2)
        else:
            X1t, X2t = X1, X2
        m = rf()
        m.fit(X1t, y1)
        proba = m.predict_proba(X2t)[:, 1]
        P, R, F, fp = metrics(y1, proba, y2)
        print(f"{name:<22}{X1.shape[1]:>5}{P:>8.3f}{R:>8.3f}{F:>8.3f}{fp:>8.3%}")


def param_cv(df, label):
    print(f"\n(B) Window-size CV on {label} only  (5-fold StratifiedKFold, F1)")
    print(f"{'W':>5}{'stride':>8}{'n_win':>9}{'mean F1':>10}{'std':>8}")
    print("-" * 40)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for w in (50, 100, 150, 200):
        stride = int(w / 4)
        X, f, _ = build_windows(df, w, stride, ngram=2)
        y = (f >= FRAC).astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        # balance benign:attack to 2:1 so CV mirrors training and stays fast
        na = int(y.sum())
        nb = np.where(y == 0)[0]
        rng = np.random.default_rng(42)
        keep_b = rng.choice(nb, min(len(nb), 2 * na), replace=False)
        keep = np.sort(np.concatenate([keep_b, np.where(y == 1)[0]]))
        Xb, yb = X[keep], y[keep]
        s = cross_val_score(rf(), Xb, yb, cv=skf, scoring="f1", n_jobs=1)
        print(f"{w:>5}{stride:>8}{len(Xb):>9,}{s.mean():>10.3f}{s.std():>8.3f}")


def main():
    print("Loading sessions ...")
    df1 = pd.read_csv(S1, usecols=_USED)
    df2 = pd.read_csv(S2, usecols=_USED)
    ablation(df1, df2)
    param_cv(df1, "S1 (train)")
    param_cv(df2, "S2")


if __name__ == "__main__":
    main()
