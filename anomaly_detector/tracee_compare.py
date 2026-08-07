#!/usr/bin/env python3
"""
 Tracee's JSON output
interleaves raw events with signature detections; a detection is any line whose
metadata.Properties.signatureID is set (e.g. "TRC-2", "TRC-104"). We keep only
those, map each to an attack technique via the wall-clock timeline, and count
anything outside every attack window as a benign false alarm.

"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from falco_compare import ml_side, TECHNIQUES  # noqa: E402


def read_timeline(path):
    tl = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                tl.append((p[0], float(p[1]), float(p[2])))
    return tl

"""
Attribute a detection to the attack ONLY if the detecting process is in
the attack cgroup (attack_cg). This is the same rigorous rule we use to
label our own data, and it is essential here: Tracee's TRC-104
(dynamic-code-loading) heuristic fires constantly on benign desktop JIT
(browsers, gnome-shell), so pure timeline attribution over-credits it. A
detection outside the attack cgroup is a benign false alarm, regardless of
which technique window it happens to overlap.
"""

def tracee_side(path, tl, attack_cg):
    def which(ts):
        for tech, s, e in tl:
            if s <= ts <= e:
                return tech
        return "unattributed"

    by_tech = defaultdict(int)
    benign = 0
    sigs = defaultdict(int)              # total fired
    sig_attack = defaultdict(int)        # fired from the attack cgroup
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            props = a.get("metadata", {}).get("Properties", {}) or {}
            sid = props.get("signatureID")
            if not sid:                      # raw event, not a detection
                continue
            sname = props.get("signatureName", sid)
            ts = a.get("timestamp")
            if ts is None:
                continue
            key = f"{sid}: {sname}"
            sigs[key] += 1
            if a.get("cgroupId") == attack_cg:
                sig_attack[key] += 1
                by_tech[which(ts / 1e9)] += 1
            else:
                benign += 1
    return by_tech, benign, dict(sigs), dict(sig_attack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracee", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--cgroup", required=True,
                    help="Attack cgroup id file (e.g. /tmp/pe_cg_tracee.txt) "
                         "or the integer id directly")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--min-attack-frac", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cg = args.cgroup
    attack_cg = int(cg) if cg.isdigit() else int(open(cg).read().strip())
    tl = read_timeline(args.timeline)
    tbt, tbenign, tsigs, tsig_attack = tracee_side(args.tracee, tl, attack_cg)

    print("=" * 62)
    print("  Tracee (default signatures) vs learned model — same session")
    print(f"  attack cgroup = {attack_cg}  (detections attributed by cgroup)")
    print("=" * 62)
    print("  Tracee signatures fired  (total / from attack cgroup / benign):")
    for k, c in sorted(tsigs.items(), key=lambda x: -x[1]):
        atk = tsig_attack.get(k, 0)
        print(f"    {k}: {c} total  |  {atk} attack  |  {c - atk} benign")
    print(f"\n  Tracee benign false alarms: {tbenign}")

    rec, fpr = ml_side(args.train, args.test, args.window, args.stride,
                       args.ngram, args.min_attack_frac, args.threshold)

    print(f"\n  {'technique':<22} {'Tracee':>7} {'ML recall':>12}")
    print("  " + "─" * 44)
    tracee_hits = ml_hits = 0
    for t in TECHNIQUES:
        th = "yes" if tbt.get(t, 0) else "—"
        if tbt.get(t, 0):
            tracee_hits += 1
        n, r = rec[t]
        if r is None:
            mlc = "n/a"
        else:
            mlc = f"{r:.2f} (n={n})"
            if r >= 0.5:
                ml_hits += 1
        print(f"  {t:<22} {th:>7} {mlc:>12}")
    print("  " + "─" * 44)
    print(f"  Techniques caught:  Tracee {tracee_hits}/14   "
          f"ML {ml_hits}/14 (recall>=0.5)")
    print(f"  False alarms:       Tracee {tbenign}   "
          f"ML window-FPR {fpr:.3f}")
    print("\n  NOTE: map signatures to techniques semantically (below), not "
          "just by timeline\n  timing, since detections can land in an adjacent "
          "window at technique boundaries.")


if __name__ == "__main__":
    main()
