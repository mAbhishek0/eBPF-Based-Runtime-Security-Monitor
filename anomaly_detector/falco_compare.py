#!/usr/bin/env python3
"""
Both tools observes the same live session (Falco and our collector runs together
while the 14 techniques executed inside a cgroup and the desktop was used
normally).

"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import make_windows, apply_tfidf, _undersample_normal  # noqa: E402
from per_technique_eval import load, window_dominant_technique  # noqa: E402

TECHNIQUES = ["suid_exploit", "rapid_chain", "stealthy", "suid_discovery",
              "sensitive_file_access", "chmod_chown", "ptrace", "namespace",
              "capset_prctl", "sudo_misconfig", "cron_persistence",
              "ld_preload", "passwd_write", "webshell"]


def falco_side(falco_path, timeline_path):
    tl = []
    with open(timeline_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                tl.append((p[0], float(p[1]), float(p[2])))

    def which(ts):
        for tech, s, e in tl:
            if s <= ts <= e:
                return tech
        return None

    by_tech = defaultdict(int)
    benign = 0
    rules = defaultdict(int)
    with open(falco_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            t = a.get("output_fields", {}).get("evt.time")
            if not t or "rule" not in a:
                continue
            rules[a["rule"]] += 1
            tech = which(int(t) / 1e9)
            if tech:
                by_tech[tech] += 1
            else:
                benign += 1
    return by_tech, benign, dict(rules)


def ml_side(train, test, W, S, ngram, frac, thr):
    df_tr, df_te = load(train), load(test)
    Xtr, ytr = make_windows(df_tr, W, S, per_pid=False, ngram=ngram,
                            min_attack=1, min_attack_frac=frac)
    Xte, yte = make_windows(df_te, W, S, per_pid=False, ngram=ngram,
                            min_attack=1, min_attack_frac=frac)
    dom = np.array(window_dominant_technique(df_te, W, S, len(yte)))
    Xtr, Xte, _ = apply_tfidf(Xtr, Xte)
    Xb, yb = _undersample_normal(Xtr, ytr, ratio=2.0)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(Xb, yb)
    pred = (rf.predict_proba(Xte)[:, 1] >= thr).astype(int)
    atk = yte == 1
    rec = {}
    for t in TECHNIQUES:
        m = atk & (dom == t)
        n = int(m.sum())
        rec[t] = (n, pred[m].mean() if n else None)
    # window-level false-positive rate (benign windows flagged)
    ben = yte == 0
    fpr = pred[ben].mean() if ben.sum() else 0.0
    return rec, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--falco", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--min-attack-frac", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    fbt, fbenign, frules = falco_side(args.falco, args.timeline)
    rec, fpr = ml_side(args.train, args.test, args.window, args.stride,
                       args.ngram, args.min_attack_frac, args.threshold)

    print("=" * 60)
    print("  Falco (default ruleset) vs learned model — same session")
    print("=" * 60)
    print(f"  Falco rules that fired: {frules}")
    print(f"  Falco benign false alarms: {fbenign}\n")
    print(f"  {'technique':<22} {'Falco':>7} {'ML recall':>10}")
    print("  " + "─" * 42)
    falco_hits = ml_hits = 0
    for t in TECHNIQUES:
        f = "yes" if fbt.get(t, 0) else "—"
        if fbt.get(t, 0):
            falco_hits += 1
        n, r = rec[t]
        if r is None:
            mlc = "n/a"
        else:
            mlc = f"{r:.2f} (n={n})"
            if r >= 0.5:
                ml_hits += 1
        print(f"  {t:<22} {f:>7} {mlc:>10}")
    print("  " + "─" * 42)
    print(f"  Techniques caught:  Falco {falco_hits}/14   "
          f"ML {ml_hits}/14 (recall>=0.5)")
    print(f"  False alarms:       Falco {fbenign}   "
          f"ML window-FPR {fpr:.3f}")


if __name__ == "__main__":
    main()
