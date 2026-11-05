#!/usr/bin/env python3
"""
loto_eval.py — Leave-One-Technique-Out (LOTO) evaluation.

Tests whether the detector generalises to an attack technique it never saw in
training, rather than memorising per-technique fingerprints. The cross-session
result varies only the benign side (the simulated attack is deterministic,
identical events in S1 and S2), so a separate test is needed for whether the
model learns the attack boundary or the signatures.

For each technique T:
  * IN-DIST  : train on one full session, test recall on the other session's
               windows dominated by T. T *is* in training — the baseline.
  * LOTO     : same, but drop every training window containing any event of T,
               so the model has never seen T. Test recall on the held-out T.
Both directions (S1->S2 and S2->S1) are averaged. Because benign traffic also
differs across sessions, LOTO is a leave-one-technique-out AND cross-session
test at once. A large IN-DIST -> LOTO recall drop means memorisation; a small
one means the boundary generalises to unseen attacks.

Usage
    python3 anomaly_detector/loto_eval.py \\
        --s1 data_collector/interleaved/labeled_session1.csv \\
        --s2 data_collector/interleaved/labeled_session2.csv
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import KNOWN_SYSCALLS, BIGRAM_INDEX, N_BIGRAMS, apply_tfidf
from sklearn.ensemble import RandomForestClassifier

SYS_IDX = {s: i for i, s in enumerate(KNOWN_SYSCALLS)}
N_UNI = len(KNOWN_SYSCALLS)

_USED = ["timestamp_ns", "syscall_name", "label", "technique"]

# cycle_burst is a burst-coverage re-run of the base setuid-loop pattern, not a
# distinct technique (the paper reports fourteen). Its windows stay in training
# as attack positives; we just do not hold it out or test on it.
NOT_A_TECHNIQUE = {"cycle_burst"}


def build_windows(df, W, stride, ngram):
    """Global sliding windows → (X, attack_frac, [technique Counter per window])."""
    df = df.sort_values("timestamp_ns")
    names = df["syscall_name"].to_numpy()
    labels = df["label"].to_numpy().astype(np.int8)
    techs = df["technique"].fillna("").to_numpy()
    uidx = np.array([SYS_IDX.get(x, -1) for x in names], dtype=np.int32)
    n_feat = N_UNI + (N_BIGRAMS if ngram >= 2 else 0)

    X, atk_frac, tech_comp = [], [], []
    for start in range(0, len(names) - W + 1, stride):
        sl = slice(start, start + W)
        vec = np.zeros(n_feat, dtype=np.float32)
        u = uidx[sl]
        for i in u[u >= 0]:
            vec[i] += 1
        if ngram >= 2:
            wn = names[sl]
            for k in range(W - 1):
                bi = BIGRAM_INDEX.get((wn[k], wn[k + 1]), -1)
                if bi >= 0:
                    vec[N_UNI + bi] += 1
        lab = labels[sl]
        na = int(lab.sum())
        X.append(vec)
        atk_frac.append(na / W)
        tech_comp.append(Counter(techs[sl][lab == 1]) if na else Counter())
    return np.asarray(X, dtype=np.float32), np.asarray(atk_frac), tech_comp


def dominant_mask(tech_comp, atk_frac, frac, T):
    """Attack windows (>= frac attack events) whose attack events are majority T."""
    out = np.zeros(len(tech_comp), dtype=bool)
    for i, (tc, f) in enumerate(zip(tech_comp, atk_frac)):
        tot = sum(tc.values())
        if f >= frac and tot and tc.get(T, 0) > 0.5 * tot:
            out[i] = True
    return out


def contains_mask(tech_comp, T):
    return np.array([tc.get(T, 0) > 0 for tc in tech_comp], dtype=bool)


def pure_mask(tech_comp, atk_frac, frac, T):
    """Attack windows whose attack events are *exclusively* technique T."""
    out = np.zeros(len(tech_comp), dtype=bool)
    for i, (tc, f) in enumerate(zip(tech_comp, atk_frac)):
        if f >= frac and sum(tc.values()) and set(tc) == {T}:
            out[i] = True
    return out


def _fit_predict(Xtr, ytr, Xte, seed=42):
    Xtr_t, Xte_t, _ = apply_tfidf(Xtr, Xte)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(Xtr_t, ytr)
    return rf.predict_proba(Xte_t)[:, 1]


def main():
    ap = argparse.ArgumentParser(description="Leave-one-technique-out evaluation")
    ap.add_argument("--s1", default="data_collector/interleaved/labeled_session1.csv")
    ap.add_argument("--s2", default="data_collector/interleaved/labeled_session2.csv")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.10,
                    help="min attack fraction for a window to count as attack")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("Loading and windowing both sessions ...")
    W1 = build_windows(pd.read_csv(args.s1, usecols=_USED),
                       args.window, args.stride, args.ngram)
    W2 = build_windows(pd.read_csv(args.s2, usecols=_USED),
                       args.window, args.stride, args.ngram)
    print(f"  S1 windows: {len(W1[0]):,}   S2 windows: {len(W2[0]):,}")

    # named techniques present in both sessions (exclude benign "" and unattributed)
    techs = set()
    for _, _, comp in (W1, W2):
        for tc in comp:
            techs.update(tc)
    techniques = sorted(t for t in techs
                        if t and t != "unattributed" and t not in NOT_A_TECHNIQUE)
    print(f"  Techniques: {len(techniques)}\n")

    directions = [("S1->S2", W1, W2), ("S2->S1", W2, W1)]
    thr, frac = args.threshold, args.frac

    rows = []
    tot_dom = tot_pure = 0
    for T in techniques:
        indist, loto, loto_pure, ndom, npure = [], [], [], [], []
        for _, train, test in directions:
            Xtr, ftr, ttr = train
            Xte, fte, tte = test
            dom = dominant_mask(tte, fte, frac, T)
            pure = pure_mask(tte, fte, frac, T)
            if dom.sum() == 0:
                continue
            ndom.append(int(dom.sum()))
            npure.append(int(pure.sum()))
            y_full = (ftr >= frac).astype(int)

            # IN-DIST: T present in training
            proba = _fit_predict(Xtr, y_full, Xte)
            indist.append((proba[dom] > thr).mean())

            # LOTO: drop every training window containing any T event
            keep = ~contains_mask(ttr, T)
            proba = _fit_predict(Xtr[keep], y_full[keep], Xte)
            loto.append((proba[dom] > thr).mean())
            if pure.sum():
                loto_pure.append((proba[pure] > thr).mean())

        if not ndom:
            continue
        tot_dom += int(np.sum(ndom))
        tot_pure += int(np.sum(npure))
        rows.append((T, int(np.mean(ndom)), int(np.mean(npure)),
                     float(np.mean(indist)), float(np.mean(loto)),
                     float(np.mean(loto_pure)) if loto_pure else float("nan")))

    rows.sort(key=lambda r: r[4])  # worst LOTO recall (dominant) first
    print(f"{'technique':<22}{'dom':>6}{'pure':>6}{'in-dist':>9}"
          f"{'LOTO':>7}{'LOTO-pure':>11}")
    print("-" * 61)
    for T, nd, npu, ind, lo, lp in rows:
        lp_s = f"{lp:>11.3f}" if lp == lp else f"{'--':>11}"
        print(f"{T:<22}{nd:>6}{npu:>6}{ind:>9.3f}{lo:>7.3f}{lp_s}")
    ind_m = np.mean([r[3] for r in rows])
    lo_m = np.mean([r[4] for r in rows])
    lp_m = np.nanmean([r[5] for r in rows])
    print("-" * 61)
    print(f"{'MEAN':<22}{'':>6}{'':>6}{ind_m:>9.3f}{lo_m:>7.3f}{lp_m:>11.3f}")
    caught = sum(1 for r in rows if r[4] >= 0.5)
    print(f"\nHeld-out techniques with LOTO recall >= 0.5: {caught}/{len(rows)}")
    print(f"Test windows that are pure single-technique: "
          f"{tot_pure}/{tot_dom} ({100*tot_pure/max(tot_dom,1):.1f}%)")


if __name__ == "__main__":
    main()
