#!/usr/bin/env python3
"""
demo.py — Live Attack Detection Demo

Replays collected CSV data through the BoSC scorer and shows a
live terminal display:
  - Green stream  = normal syscall events
  - Red stream    = attack syscall events being processed
  - ALERT banner  = when the model detects an anomaly

Usage
-----
    # Full demo (normal → attack, shows contrast)
    python3 anomaly_detector/demo.py

    # Attack-only (faster, more alerts)
    python3 anomaly_detector/demo.py --mode attack

    # Slow down / speed up the stream
    python3 anomaly_detector/demo.py --delay 0.002

    # Quiet stream (only show alerts, no per-event output)
    python3 anomaly_detector/demo.py --quiet
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_realtime import DEFAULT_NOISE_COMM

# ── ANSI colour codes ──────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GREY   = "\033[90m"

BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"


def clr(text, *codes):
    return "".join(codes) + str(text) + RESET


# ── Inline minimal scorer (no import of bosc_realtime needed) ──────────────

class DemoScorer:
    def __init__(self, model_dir: str):
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        self.W          = meta["window_size"]
        self.stride     = meta.get("stride", self.W // 2)
        self.per_pid    = meta.get("per_pid", True)
        self.syscalls   = meta["syscalls"]
        self.sc_idx     = {s: i for i, s in enumerate(self.syscalls)}
        self.n_features = meta["n_features"]
        self.n_uni      = meta.get("n_features_unigram", len(self.syscalls))
        self.ngram      = meta.get("ngram", 1)
        self.use_tfidf  = meta.get("use_tfidf", False)

        raw_bigrams      = meta.get("bigrams", [])
        self.bigram_list = [tuple(b) for b in raw_bigrams]
        self.bg_idx      = {bg: i for i, bg in enumerate(self.bigram_list)}

        iso_path = meta.get("isoforest_path", meta.get("ocsvm_path"))
        self.iso = joblib.load(os.path.join(model_dir, iso_path))
        self.rf  = joblib.load(os.path.join(model_dir, meta["rf_path"]))

        self.tfidf = None
        if self.use_tfidf and meta.get("tfidf_path"):
            tp = os.path.join(model_dir, meta["tfidf_path"])
            if os.path.exists(tp):
                self.tfidf = joblib.load(tp)

        # Windowing state
        self._global_buf  = deque(maxlen=self.W)
        self._event_count = 0
        self.exclude_comm = DEFAULT_NOISE_COMM

    def push(self, syscall_name: str, comm: str = ""):
        """Push one syscall. Returns (is_anomaly, rf_prob, iso_score, top_features) or None."""
        if comm in self.exclude_comm:
            return None
        self._global_buf.append(syscall_name)
        self._event_count += 1

        if len(self._global_buf) < self.W:
            return None
        if self._event_count % self.stride != 0:
            return None

        return self._score()

    def _vectorise(self):
        vec  = np.zeros(self.n_features, dtype=np.float32)
        names = list(self._global_buf)
        for n in names:
            idx = self.sc_idx.get(n, -1)
            if idx >= 0:
                vec[idx] += 1
        if self.ngram >= 2:
            for k in range(len(names) - 1):
                bg = (names[k], names[k + 1])
                bi = self.bg_idx.get(bg, -1)
                if bi >= 0:
                    vec[self.n_uni + bi] += 1
        if self.tfidf is not None:
            vec = self.tfidf.transform(vec.reshape(1, -1)).toarray()[0].astype(np.float32)
        return vec

    def _score(self):
        vec    = self._vectorise()
        v2     = vec.reshape(1, -1)
        iso_p  = self.iso.predict(v2)[0]
        iso_sc = float(-self.iso.decision_function(v2)[0])
        rf_pr  = self.rf.predict_proba(v2)[0]
        rf_prob = float(rf_pr[1]) if len(rf_pr) > 1 else 0.0

        is_anom = (iso_p == -1) or (rf_prob > 0.5)

        # Top features
        all_names = list(self.syscalls) + [f"{a}→{b}" for a, b in self.bigram_list]
        top_idx   = np.argsort(vec)[::-1][:4]
        top_feats = [
            f"{all_names[i] if i < len(all_names) else f'feat_{i}'}={vec[i]:.2f}"
            for i in top_idx if vec[i] > 0
        ]
        return is_anom, rf_prob, iso_sc, top_feats


# ── Terminal helpers ──────────────────────────────────────────────────────

def print_header(model_dir: str):
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    print(clr("=" * 62, CYAN, BOLD))
    print(clr("  eBPF-Based Runtime Security Monitor", CYAN, BOLD))
    print(clr("  LIVE ATTACK DETECTION DEMO", WHITE, BOLD))
    print(clr("=" * 62, CYAN, BOLD))
    print(f"  {clr('Model', DIM)} : IsolationForest + RandomForest (BoSC)")
    print(f"  {clr('Window', DIM)}: W={meta['window_size']}, stride={meta.get('stride', '?')}, "
          f"N-gram={meta.get('ngram', 1)}, TF-IDF={meta.get('use_tfidf', False)}")
    print(f"  {clr('Features', DIM)}: {meta['n_features']} dims "
          f"({meta.get('n_features_unigram', '?')} unigrams + "
          f"{meta['n_features'] - meta.get('n_features_unigram', 0)} bigrams)")
    print(clr("=" * 62, CYAN, BOLD))
    print()


def print_event(idx: int, syscall: str, pid: int, comm: str,
                label: int, quiet: bool):
    if quiet:
        return
    tag   = clr("ATK", RED, BOLD) if label == 1 else clr("NRM", GREEN)
    sc    = clr(f"{syscall:<15}", RED if label == 1 else GREY)
    comm_ = clr(f"{comm:<12}", YELLOW)
    print(f"  [{tag}] #{idx:>7}  {sc}  pid={pid:<7}  comm={comm_}", flush=True)


def print_alert(rf_prob: float, iso_score: float,
                top_feats: list, pid: int, comm: str, alert_n: int):
    w = 60
    bar_len = int(rf_prob * 40)
    bar  = clr("█" * bar_len, RED, BOLD) + clr("░" * (40 - bar_len), GREY)

    print()
    print(clr("!" * w, RED, BOLD))
    print(clr(f"{'  ANOMALY DETECTED  —  ALERT #' + str(alert_n):^{w}}", BG_RED, WHITE, BOLD))
    print(clr("!" * w, RED, BOLD))
    print(f"  {clr('Process', DIM)}  : {clr(comm, YELLOW, BOLD)} (PID {pid})")
    print(f"  {clr('RF Prob', DIM)}  : {bar}  {clr(f'{rf_prob:.0%}', RED, BOLD)}")
    print(f"  {clr('IsoScore', DIM)} : {clr(f'{iso_score:.4f}', YELLOW)}  "
          f"{'(higher = more anomalous)' if iso_score > 0 else ''}")
    print(f"  {clr('Top Feats', DIM)}: {clr(', '.join(top_feats), CYAN)}")
    print(clr("!" * w, RED, BOLD))
    print()


def print_normal_section():
    w = 62
    print()
    print(clr("=" * w, GREEN, BOLD))
    print(clr(f"{'  PHASE 1: NORMAL BEHAVIOUR  ':^{w}}", BG_GREEN, BOLD))
    print(clr("=" * w, GREEN, BOLD))
    print(f"  {clr('Streaming normal syscall events...', GREEN)}")
    print(f"  {clr('Monitoring for deviations from baseline.', DIM)}")
    print()


def print_attack_section():
    w = 62
    print()
    print(clr("=" * w, RED, BOLD))
    print(clr(f"{'  PHASE 2: ATTACK IN PROGRESS  ':^{w}}", BG_RED, WHITE, BOLD))
    print(clr("=" * w, RED, BOLD))
    print(f"  {clr('Attack simulation started!', RED, BOLD)}")
    print(f"  {clr('Privilege escalation syscalls detected in stream.', YELLOW)}")
    print(f"  {clr('Watch for anomaly alerts below...', RED)}")
    print()


def print_summary(total_events: int, normal_events: int, attack_events: int,
                  alerts: int, elapsed: float):
    w = 62
    print()
    print(clr("=" * w, CYAN, BOLD))
    print(clr(f"{'  DEMO COMPLETE ':^{w}}", CYAN, BOLD))
    print(clr("=" * w, CYAN, BOLD))
    print(f"  {clr('Total events processed', DIM)}: {total_events:,}")
    print(f"  {clr('Normal events', DIM)}          : {clr(str(normal_events), GREEN)}")
    print(f"  {clr('Attack events', DIM)}           : {clr(str(attack_events), RED)}")
    print(f"  {clr('Alerts fired', DIM)}            : {clr(str(alerts), YELLOW, BOLD)}")
    print(f"  {clr('Elapsed time', DIM)}            : {elapsed:.1f}s")
    print(clr("=" * w, CYAN, BOLD))
    print()


# ── Main demo logic ───────────────────────────────────────────────────────

def run_demo(model_dir: str, data_dir: str, mode: str,
             delay: float, quiet: bool,
             max_normal: int, max_attack: int,
             exclude_comm: set = None):

    print_header(model_dir)
    scorer = DemoScorer(model_dir)
    if exclude_comm is not None:
        scorer.exclude_comm = set(exclude_comm)

    # Load CSVs
    normal_csvs = sorted(glob.glob(os.path.join(data_dir, "syscalls_normal_*.csv")))
    attack_csvs = sorted(glob.glob(os.path.join(data_dir, "syscalls_attack_*.csv")))

    if not attack_csvs:
        print(clr("ERROR: No attack CSV found in " + data_dir, RED, BOLD))
        sys.exit(1)

    normal_df = pd.concat(
        [pd.read_csv(p) for p in normal_csvs], ignore_index=True
    ).sort_values("timestamp_ns") if normal_csvs else pd.DataFrame()

    attack_df = pd.concat(
        [pd.read_csv(p) for p in attack_csvs], ignore_index=True
    ).sort_values("timestamp_ns")

    total_events  = 0
    normal_events = 0
    attack_events = 0
    alerts        = 0
    last_alert    = 0.0
    alert_cooldown = 1.5   # seconds between consecutive alerts
    start_time    = time.time()

    # ── Phase 1: normal stream ─────────────────────────────────────────
    if mode in ("both", "normal") and len(normal_df) > 0:
        print_normal_section()
        sample = normal_df.head(max_normal)
        for _, row in sample.iterrows():
            sc   = str(row["syscall_name"])
            pid  = int(row["pid"])
            comm = str(row.get("comm", "?"))
            uid  = int(row.get("uid", 0))

            result = scorer.push(sc, comm=comm)
            total_events  += 1
            normal_events += 1

            print_event(total_events, sc, pid, comm, 0, quiet)

            if result:
                is_anom, rf_prob, iso_sc, top_feats = result
                now = time.time()
                if is_anom and (now - last_alert) > alert_cooldown:
                    last_alert = now
                    alerts += 1
                    print_alert(rf_prob, iso_sc, top_feats, pid, comm, alerts)
                    time.sleep(1.0)   # pause so alert is visible

            time.sleep(delay)

        print(clr(f"\n  Normal phase complete: {normal_events:,} events, "
                  f"{alerts} alerts (false positives).", GREEN))
        print()
        input(clr("  Press ENTER to start Phase 2: Attack Simulation...", YELLOW, BOLD))

    # ── Phase 2: attack stream ─────────────────────────────────────────
    if mode in ("both", "attack"):
        print_attack_section()
        # Reset scorer window for clean demo
        scorer._global_buf.clear()
        scorer._event_count = 0
        alerts = 0  # reset counter for phase 2

        sample = attack_df.head(max_attack)
        for _, row in sample.iterrows():
            sc   = str(row["syscall_name"])
            pid  = int(row["pid"])
            comm = str(row.get("comm", "?"))
            uid  = int(row.get("uid", 0))

            result = scorer.push(sc, comm=comm)
            total_events  += 1
            attack_events += 1

            print_event(total_events, sc, pid, comm, 1, quiet)

            if result:
                is_anom, rf_prob, iso_sc, top_feats = result
                now = time.time()
                if is_anom and (now - last_alert) > alert_cooldown:
                    last_alert = now
                    alerts += 1
                    print_alert(rf_prob, iso_sc, top_feats, pid, comm, alerts)
                    time.sleep(1.5)  # pause on alert

            time.sleep(delay)

    elapsed = time.time() - start_time
    print_summary(total_events, normal_events, attack_events, alerts, elapsed)


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live BoSC attack detection demo"
    )
    parser.add_argument("--model-dir", default="anomaly_detector/models",
                        help="Path to trained model directory")
    parser.add_argument("--data-dir",  default="data_collector/collected_data",
                        help="Path to collected CSV files")
    parser.add_argument("--mode",      default="both",
                        choices=["both", "normal", "attack"],
                        help="Which phases to show (default: both)")
    parser.add_argument("--delay",     type=float, default=0.003,
                        help="Delay (seconds) between events (default: 0.003)")
    parser.add_argument("--quiet",     action="store_true",
                        help="Hide per-event stream, show alerts only")
    parser.add_argument("--max-normal", type=int, default=5000,
                        help="Max normal events to replay (default: 5000)")
    parser.add_argument("--max-attack", type=int, default=10000,
                        help="Max attack events to replay (default: 10000)")
    parser.add_argument("--exclude-comm", nargs="*", default=list(DEFAULT_NOISE_COMM),
                        help="Process names to ignore as system noise "
                             f"(default: {sorted(DEFAULT_NOISE_COMM)}). Add "
                             "environment-specific noise (editors, browsers, ...) "
                             "seen in your own collected CSVs.")
    args = parser.parse_args()

    try:
        run_demo(
            model_dir    = args.model_dir,
            data_dir     = args.data_dir,
            mode         = args.mode,
            delay        = args.delay,
            quiet        = args.quiet,
            max_normal   = args.max_normal,
            max_attack   = args.max_attack,
            exclude_comm = set(args.exclude_comm),
        )
    except KeyboardInterrupt:
        print(clr("\n\n  Demo interrupted by user.", YELLOW))
        sys.exit(0)


if __name__ == "__main__":
    main()
