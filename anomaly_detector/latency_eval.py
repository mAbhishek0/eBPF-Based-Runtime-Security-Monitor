#!/usr/bin/env python3
"""
We define detection latency as the time from the
start of an attack burst (the first escalation syscall after a benign gap) to
the close of the first window that the model flags as attack.

Latency is measured in kernel-monotonic time (timestamp_ns), so it is immune to
wall-clock adjustments. Bursts that never produce a flagged window are counted
as undetected (they contribute to the miss rate, not the latency distribution).

"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import make_windows, apply_tfidf, _undersample_normal  # noqa: E402

BUILD_TOOLCHAIN = {"cc1", "gcc", "cc", "as", "ld", "collect2", "make", "cmake", "ninja"}


def load(path):
    df = pd.read_csv(path)
    df = df[~df["comm"].isin(BUILD_TOOLCHAIN)].reset_index(drop=True)
    return df.sort_values("timestamp_ns").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--min-attack-frac", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gap-s", type=float, default=1.0,
                    help="A new attack burst starts after this many seconds "
                         "with no attack syscalls (default 1.0)")
    args = ap.parse_args()
    W, S = args.window, args.stride

    print("=" * 60)
    print("  Detection latency (cross-session, interleaved)")
    print(f"  W={W} stride={S} thr={args.threshold} burst-gap={args.gap_s}s")
    print("=" * 60)

    df_tr, df_te = load(args.train), load(args.test)
    Xtr, ytr = make_windows(df_tr, W, S, per_pid=False, ngram=args.ngram,
                            min_attack=1, min_attack_frac=args.min_attack_frac)
    Xte, yte = make_windows(df_te, W, S, per_pid=False, ngram=args.ngram,
                            min_attack=1, min_attack_frac=args.min_attack_frac)
    Xtr, Xte, _ = apply_tfidf(Xtr, Xte)
    Xb, yb = _undersample_normal(Xtr, ytr, ratio=2.0)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(Xb, yb)
    pred = (rf.predict_proba(Xte)[:, 1] >= args.threshold).astype(int)

    ts = df_te["timestamp_ns"].to_numpy()
    lbl = df_te["label"].to_numpy()
    tech = df_te["technique"].fillna("").to_numpy()
    n = len(ts)

    # close time of each flagged window (window k closes at its last event)
    alarm_close = np.array(sorted(
        ts[min(k * S + W - 1, n - 1)] for k in range(len(pred)) if pred[k] == 1
    ))

    # attack-burst onsets: first attack event after >gap seconds of no attack
    gap_ns = args.gap_s * 1e9
    atk_idx = np.where(lbl == 1)[0]
    onsets = []            # (onset_ts, technique)
    prev_t = None
    for i in atk_idx:
        if prev_t is None or ts[i] - prev_t > gap_ns:
            onsets.append((ts[i], tech[i]))
        prev_t = ts[i]

    # for each onset, latency = first flagged-window close at/after onset
    lat_ms, per_tech = [], {}
    misses = 0
    for t0, tname in onsets:
        j = np.searchsorted(alarm_close, t0, side="left")
        if j < len(alarm_close):
            lat = (alarm_close[j] - t0) / 1e6      # ns -> ms
            lat_ms.append(lat)
            per_tech.setdefault(tname or "unattributed", []).append(lat)
        else:
            misses += 1

    lat_ms = np.array(lat_ms)
    print(f"\n  Attack bursts (onsets): {len(onsets):,}")
    print(f"  Detected: {len(lat_ms):,}   Undetected: {misses:,}")
    if len(lat_ms):
        print("\n  Detection latency (ms), over detected bursts:")
        print(f"    median : {np.median(lat_ms):8.1f} ms")
        print(f"    mean   : {np.mean(lat_ms):8.1f} ms")
        print(f"    p90    : {np.percentile(lat_ms,90):8.1f} ms")
        print(f"    max    : {np.max(lat_ms):8.1f} ms")
        print("\n  Median latency by technique:")
        for t in sorted(per_tech, key=lambda k: np.median(per_tech[k])):
            v = np.array(per_tech[t])
            print(f"    {t:<22} n={len(v):>4}  median={np.median(v):8.1f} ms")


if __name__ == "__main__":
    main()
