#!/usr/bin/env python3
"""
bosc_realtime.py — Real-time BoSC Scoring Engine

Supports both windowing modes:
  - Global  (per_pid=False): one shared sliding window for all events.
    Used when trained with --global-window .
  - Per-PID (per_pid=True) : separate window per process.

Automatically detects the correct mode from meta.json.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

# Ensure both system dist-packages and virtualenv site-packages are in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_site = os.path.join(_project_root, ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)
_sys_dist = "/usr/lib/python3/dist-packages"
if os.path.exists(_sys_dist) and _sys_dist not in sys.path:
    sys.path.append(_sys_dist)

import joblib
import numpy as np
import pandas as pd

ALERT_COOLDOWN_S = 2.0   # seconds between repeated alerts for same PID (per-pid mode)
GLOBAL_COOLDOWN_S = 1.0  # seconds between consecutive global alerts


DEFAULT_NOISE_COMM = {"cc1", "gcc", "cc", "as", "ld", "collect2", "make", "cmake", "ninja", "cron"}

"""
Stateful BoSC scorer — supports global and per-PID sliding window modes.

Parameters
----------
model_dir : str   — path to directory produced by bosc_train.py.
verbose   : bool  — print every scoring decision (debugging).
"""

class RealtimeScorer:
    
    def __init__(self, model_dir: str = "anomaly_detector/models",
                 verbose: bool = False,
                 exclude_comm: set = None,
                 rf_threshold: float = 0.5,
                 require_agreement: bool = False):
        self.verbose = verbose
        self.exclude_comm = DEFAULT_NOISE_COMM if exclude_comm is None else set(exclude_comm)
        self.rf_threshold = rf_threshold
        self.require_agreement = require_agreement
        self._load_models(model_dir)

        if self.per_pid:
            # Per-PID mode: separate deque per process
            self._windows: dict[int, deque] = defaultdict(
                lambda: deque(maxlen=self.window_size)
            )
            self._last_alert: dict[int, float] = {}
        else:
            # Global mode: one shared deque, stride-based scoring
            self._global_window: deque = deque(maxlen=self.window_size)
            self._event_count: int = 0      # counts events since last score
            self._last_global_alert: float = 0.0

        self._total_alerts = 0


    def _load_models(self, model_dir: str):
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in {model_dir!r}. "
                "Run bosc_train.py first."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.window_size   = meta["window_size"]
        self.stride        = meta.get("stride", self.window_size // 2)
        
        self.per_pid       = meta.get("per_pid", True)
        self.syscall_list  = meta["syscalls"]
        self.syscall_index = {s: i for i, s in enumerate(self.syscall_list)}
        self.n_features    = meta["n_features"]
        self.n_unigrams    = meta.get("n_features_unigram", len(self.syscall_list))
        self.ngram         = meta.get("ngram", 1)
        self.use_tfidf     = meta.get("use_tfidf", False)

        raw_bigrams = meta.get("bigrams", [])
        self.bigram_list  = [tuple(b) for b in raw_bigrams]
        self.bigram_index = {bg: i for i, bg in enumerate(self.bigram_list)}

        iso_path = meta.get("isoforest_path", meta.get("ocsvm_path"))
        self.anomaly_model = joblib.load(os.path.join(model_dir, iso_path))
        self._is_isoforest = "IsolationForest" in type(self.anomaly_model).__name__

        self.rf = joblib.load(os.path.join(model_dir, meta["rf_path"]))

        self.tfidf = None
        if self.use_tfidf and meta.get("tfidf_path"):
            tpath = os.path.join(model_dir, meta["tfidf_path"])
            if os.path.exists(tpath):
                self.tfidf = joblib.load(tpath)

        mode_str = "GLOBAL stream" if not self.per_pid else "per-PID"
        model_name = "IsolationForest" if self._is_isoforest else "OneClassSVM"
        print(f"\n{'='*60}")
        print(f"  [RealtimeScorer] Models loaded from {model_dir!r}")
        print(f"  Anomaly model  : {model_name}")
        print(f"  Window mode    : {mode_str}")
        print(f"  Window size    : {self.window_size} events")
        print(f"  Stride         : {self.stride} events")
        print(f"  N-gram         : {self.ngram}")
        print(f"  TF-IDF         : {self.use_tfidf}")
        print(f"  Feature dims   : {self.n_features}")
        print(f"{'='*60}\n")


    def _vectorise(self, window) -> np.ndarray:
        vec = np.zeros(self.n_features, dtype=np.float32)
        names = list(window)

        for name in names:
            idx = self.syscall_index.get(name, -1)
            if idx >= 0:
                vec[idx] += 1

        if self.ngram >= 2:
            for k in range(len(names) - 1):
                bg = (names[k], names[k + 1])
                bi = self.bigram_index.get(bg, -1)
                if bi >= 0:
                    vec[self.n_unigrams + bi] += 1

        if self.tfidf is not None:
            vec = self.tfidf.transform(vec.reshape(1, -1)).toarray()[0].astype(np.float32)

        return vec


    def _score_vector(self, vec: np.ndarray) -> tuple:
        vec_2d = vec.reshape(1, -1)

        iso_pred  = self.anomaly_model.predict(vec_2d)[0]
        iso_score = -self.anomaly_model.decision_function(vec_2d)[0]
        iso_label = "ANOMALY" if iso_pred == -1 else "normal"

        rf_proba       = self.rf.predict_proba(vec_2d)[0]
        rf_attack_prob = rf_proba[1] if len(rf_proba) > 1 else 0.0

        rf_flags_attack = rf_attack_prob > self.rf_threshold
        if self.require_agreement:
            is_anomaly = (iso_pred == -1) and rf_flags_attack
        else:
            is_anomaly = (iso_pred == -1) or rf_flags_attack

        reason = (
            f"RF_attack_prob={rf_attack_prob:.0%}  "
            f"IsoForest={iso_label}(score={iso_score:.3f})"
        )
        return is_anomaly, rf_attack_prob, iso_score, reason

    def _top_features(self, vec: np.ndarray, n: int = 5) -> list:
        all_names = list(self.syscall_list)
        if self.ngram >= 2:
            all_names += [f"{a}→{b}" for a, b in self.bigram_list]
        top_idx = np.argsort(vec)[::-1][:n]
        return [
            f"{all_names[i] if i < len(all_names) else f'feat_{i}'}={vec[i]:.2f}"
            for i in top_idx if vec[i] > 0
        ]


    def _fire_alert(self, vec, rf_prob, iso_score, pid=None, comm="", uid=0):
        self._total_alerts += 1
        top = ", ".join(self._top_features(vec))
        pid_str = f"PID={pid} " if pid is not None else ""
        comm_str = f"comm={comm!r} " if comm else ""
        print(
            f"\n  ╔══════════════════════════════════════════════════╗\n"
            f"  ║    ANOMALY DETECTED                            ║\n"
            f"  ╚══════════════════════════════════════════════════╝\n"
            f"  {pid_str}{comm_str}UID={uid}\n"
            f"  RF attack probability : {rf_prob:.0%}\n"
            f"  IsoForest anomaly score: {iso_score:.3f}\n"
            f"  Top features in window: [{top}]\n"
        )

    # Public API 


    def on_event(self, pid: int, comm: str, syscall_name: str, uid: int = 0) -> bool:
        
        if comm in self.exclude_comm:
            return False
        if self.per_pid:
            return self._on_event_per_pid(pid, comm, syscall_name, uid)
        else:
            return self._on_event_global(pid, comm, syscall_name, uid)

    def _on_event_per_pid(self, pid, comm, syscall_name, uid) -> bool:
        buf = self._windows[pid]
        buf.append(syscall_name)

        # Score when window is full, OR immediately when a high-risk privilege escalation syscall occurs
        high_risk = syscall_name in ("setuid", "setgid", "setreuid", "setresuid", "ptrace", "capset", "unshare", "execve", "chmod", "chown")
        if len(buf) < self.window_size and not (high_risk and len(buf) >= 3):
            return False

        vec = self._vectorise(buf)
        is_anomaly, rf_prob, iso_score, reason = self._score_vector(vec)

        if self.verbose:
            print(f"  [score] PID={pid} {reason}")

        if is_anomaly:
            now  = time.monotonic()
            last = self._last_alert.get(pid, 0.0)
            if now - last >= ALERT_COOLDOWN_S:
                self._last_alert[pid] = now
                self._fire_alert(vec, rf_prob, iso_score, pid=pid, comm=comm, uid=uid)
                return True
        return False

    def _on_event_global(self, pid, comm, syscall_name, uid) -> bool:
        
        self._global_window.append(syscall_name)
        self._event_count += 1

        # Only score when window is full AND we've seen stride new events
        if (len(self._global_window) < self.window_size or
                self._event_count % self.stride != 0):
            return False

        vec = self._vectorise(self._global_window)
        is_anomaly, rf_prob, iso_score, reason = self._score_vector(vec)

        if self.verbose:
            print(f"  [global score] event#{self._event_count}  {reason}")

        if is_anomaly:
            now = time.monotonic()
            if now - self._last_global_alert >= GLOBAL_COOLDOWN_S:
                self._last_global_alert = now
                self._fire_alert(vec, rf_prob, iso_score,
                                 pid=pid, comm=comm, uid=uid)
                return True
        return False

    @property
    def alert_count(self) -> int:
        return self._total_alerts


def _replay_csv(csv_path: str, scorer: RealtimeScorer):
    df = pd.read_csv(csv_path).sort_values("timestamp_ns")
    alerts = 0
    n = 0
    for row in df.itertuples(index=False):
        n += 1
        fired = scorer.on_event(
            pid=int(row.pid),
            comm=str(row.comm),
            syscall_name=str(row.syscall_name),
            uid=int(row.uid),
        )
        if fired:
            alerts += 1
    print(f"\n  Replay complete: {n:,} events → {alerts} alerts fired")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay a CSV through the RealtimeScorer"
    )
    parser.add_argument("--model-dir", default="anomaly_detector/models")
    parser.add_argument("--csv", required=True, help="CSV file to replay")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exclude-comm", nargs="*", default=None,
                        help=f"Process names to ignore (default: {sorted(DEFAULT_NOISE_COMM)})")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="RF attack-probability threshold to flag a window (default: 0.5)")
    parser.add_argument("--require-agreement", action="store_true",
                        help="Only alert when RF AND IsoForest both flag the window "
                             "(fewer false positives, may miss subtler attacks)")
    args = parser.parse_args()

    scorer = RealtimeScorer(model_dir=args.model_dir, verbose=args.verbose,
                            exclude_comm=args.exclude_comm,
                            rf_threshold=args.threshold,
                            require_agreement=args.require_agreement)
    _replay_csv(args.csv, scorer)
