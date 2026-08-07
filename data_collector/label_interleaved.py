#!/usr/bin/env python3
"""
During interleaved collection the machine is used normally WHILE the attack
simulator runs its 14 techniques inside a dedicated cgroup (see run_attack_scoped.sh). The collector records a cgroup_id for every event, so
attack and benign activity can be separated after the fact at the event level
rather than by whole-session labeling.

This tool reads:
  * one or more raw CSVs from collector.py (must contain a cgroup_id column),
  * the attack cgroup id written by the simulator (pe_attack_cgroup.txt),
  * the per-technique wall-clock timeline (pe_technique_timeline.tsv),
and writes a labeled CSV with two derived columns:
  * label     : 1 if the event's cgroup_id matches the attack run, else 0
  * technique : for attack events, the technique whose wall-clock window
                contains the event (else "unattributed"); empty for normal.

It also prints a cross-check: known attack binaries (suid_exploit, rapid_chain,
...) should land almost entirely in the attack cgroup. If they do not, the
cgroup id is probably wrong and the labels should not be trusted.

"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd


KNOWN_ATTACK_COMMS = {
    "suid_exploit", "rapid_chain", "stealthy", "ptrace_sim",
    "namespace_sim", "capset_sim", "cycle", "webshell_sim",
}

TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


def read_attack_cgroup(path: str) -> int:
    if not os.path.exists(path):
        print(f"[ERROR] cgroup marker not found: {path}")
        print("        Run the attack via run_attack_scoped.sh first, or pass "
              "--attack-cgroup <id> explicitly.")
        sys.exit(1)
    with open(path) as f:
        return int(f.read().strip())


def read_timeline(path: str):
    """Return list of (technique, start_dt, end_dt) as naive local datetimes."""
    if not path or not os.path.exists(path):
        print(f"[warn] timeline not found ({path}); per-technique attribution "
              "will be skipped.")
        return []
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            tech, s, e = parts
            
            rows.append((tech, datetime.fromtimestamp(float(s)),
                         datetime.fromtimestamp(float(e))))
    return rows


def load_raw(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        print(f"[ERROR] no CSVs matched: {patterns}")
        sys.exit(1)
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "cgroup_id" not in df.columns:
            print(f"[ERROR] {p} has no cgroup_id column. Re-collect with the "
                  "updated collector.py.")
            sys.exit(1)
        print(f"  loaded {os.path.basename(p):45s} ({len(df):>9,} rows)")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Label an interleaved capture by cgroup + technique timeline")
    ap.add_argument("--raw", nargs="+", required=True,
                    help="Raw collector CSV(s); globs allowed")
    ap.add_argument("--cgroup", default="/tmp/pe_attack_cgroup.txt",
                    help="File containing the attack cgroup id")
    ap.add_argument("--attack-cgroup", type=int, default=None,
                    help="Attack cgroup id directly (overrides --cgroup)")
    ap.add_argument("--timeline", default="/tmp/pe_technique_timeline.tsv",
                    help="Per-technique timeline TSV")
    ap.add_argument("--out", required=True, help="Output labeled CSV path")
    args = ap.parse_args()

    attack_cg = args.attack_cgroup if args.attack_cgroup is not None \
        else read_attack_cgroup(args.cgroup)
    print(f"[*] Attack cgroup id: {attack_cg}")

    print("[*] Loading raw capture(s) ...")
    df = load_raw(args.raw)

    # ── Event-level attack/normal label from cgroup membership ──────────
    df["label"] = (df["cgroup_id"] == attack_cg).astype(int)
    n_atk = int(df["label"].sum())
    n_nrm = len(df) - n_atk
    print(f"\n  Attack events : {n_atk:,}")
    print(f"  Normal events : {n_nrm:,}")
    if n_atk == 0:
        print("[ERROR] no events matched the attack cgroup. Wrong id, or the "
              "attack ran outside the scoped cgroup. Check the marker file.")
        sys.exit(1)

    # ── Cross-check against known attack binaries ──────────────────────
    known = df[df["comm"].isin(KNOWN_ATTACK_COMMS)]
    if len(known):
        in_atk = int((known["label"] == 1).sum())
        frac = in_atk / len(known) * 100
        print(f"\n  Cross-check: {in_atk:,}/{len(known):,} events from known "
              f"attack binaries fell in the attack cgroup ({frac:.1f}%).")
        if frac < 95:
            print("  [warn] expected ~100%. The cgroup id may be wrong; do not "
                  "trust these labels without checking.")
    else:
        print("\n  [warn] no known attack-binary events seen; cannot cross-check.")

    # Per-technique attribution for attack events
    df["technique"] = ""
    timeline = read_timeline(args.timeline)
    if timeline:
        atk_mask = df["label"] == 1
        event_dt = pd.to_datetime(df.loc[atk_mask, "timestamp_human"],
                                  format=TS_FMT, errors="coerce")
        starts = pd.to_datetime([s for _, s, _ in timeline])
        ends = pd.to_datetime([e for _, _, e in timeline])
        techs = [t for t, _, _ in timeline]
        # Techniques run sequentially, so the windows are non-overlapping.
        iv = pd.IntervalIndex.from_arrays(starts, ends, closed="both")
        pos = iv.get_indexer(event_dt)
        attributed = pd.Series(["unattributed"] * len(event_dt), index=event_dt.index)
        found = pos >= 0
        attributed[found] = [techs[i] for i in pos[found]]
        df.loc[atk_mask, "technique"] = attributed.values

        print("\n  Per-technique attack-event counts:")
        vc = df.loc[atk_mask, "technique"].value_counts()
        for name, cnt in vc.items():
            print(f"    {name:<22} {cnt:>8,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[*] Labeled dataset written → {args.out}")
    print(f"    {len(df):,} rows  |  attack {n_atk:,}  normal {n_nrm:,}  "
          f"({n_atk/len(df)*100:.1f}% attack)")


if __name__ == "__main__":
    main()
