#!/usr/bin/env python3
"""
bosc_train.py — Bag-of-System-Calls Feature Engineering & Model Training

* N-gram BoSC  : adds bigram counts for critical syscall pairs (--ngram 2).
* TF-IDF       : downweights frequent background syscalls (--tfidf).
* IsolationForest replaces OneClassSVM — far more robust on small datasets.
* StratifiedKFold CV for reliable metrics (--cv-folds, default 5).
* Window parameter search via --tune-window.
* RandomForest hyperparameter search via --tune-rf.

Usage
    # Basic (unigram BoSC, defaults):
    python3 anomaly_detector/bosc_train.py

    # With bigrams + TF-IDF:
    python3 anomaly_detector/bosc_train.py --ngram 2 --tfidf

    # Find best window size first, then train:
    python3 anomaly_detector/bosc_train.py --tune-window --ngram 2 --tfidf

    # Everything on:
    python3 anomaly_detector/bosc_train.py --ngram 2 --tfidf --tune-window --tune-rf
"""

import argparse
import glob
import json
import os
import sys
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)

# syscall vocabulary
KNOWN_SYSCALLS = [
    "execve", "clone", "open", "openat", "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "ptrace", "mmap",
    "chmod", "fchmod", "chown", "fchown", "mount", "execveat", "prctl", "capset", "unshare",
]

# privilege-critical syscalls
CRITICAL_SYSCALLS = {
    "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "ptrace", "capset",
    "mount", "unshare", "execve", "execveat", "chmod",
}

"""building ordered list of bigrams where ≥1 syscall is critical."""
def _build_bigram_index() -> tuple:
    bigrams = []
    for a in KNOWN_SYSCALLS:
        for b in KNOWN_SYSCALLS:
            if a in CRITICAL_SYSCALLS or b in CRITICAL_SYSCALLS:
                bigrams.append((a, b))
    idx = {bg: i for i, bg in enumerate(bigrams)}
    return idx, bigrams


BIGRAM_INDEX, BIGRAM_LIST = _build_bigram_index()
N_BIGRAMS = len(BIGRAM_LIST)


# data loading 

# The eBPF collector hooks raw_syscalls system-wide, so anything else running
# on the box during collection (the build toolchain re-invoked by the old
# simulate_attacks.sh, desktop/session daemons, etc.) gets swept in and
# inherits whatever --label was passed on the command line. Those processes
# have nothing to do with privilege escalation, so their syscalls only dilute
# and mislabel the BoSC windows. Compiler-toolchain names are filtered by
# default since they're never legitimate signal for this task; add
# environment-specific noise (editors, browsers, etc.) via --exclude-comm.
#
# NOTE: keep genuine ambient normal-workload process names
# (chrome, dpkg, python, ...) OUT of this list — that traffic is exactly the
# real baseline we collected the ambient run to teach the model.
DEFAULT_COMM_DENYLIST = {"cc1", "gcc", "cc", "as", "ld", "collect2", "make", "cmake", "ninja", "cron"}


# Only these columns are used downstream (windowing + comm noise filter);
# skipping the rest — especially the 256-char `filename` field and the
# arg1/2/3 values — roughly halves peak memory and speeds up loading, which
# matters on multi-million-row datasets that otherwise risk OOM.
_USED_COLUMNS = ["timestamp_ns", "pid", "comm", "syscall_name", "label"]


def load_csv_files(data_dir: str, exclude_comm: set = None) -> pd.DataFrame:
    """Load all syscall_*.csv files from data_dir and concatenate."""
    pattern = os.path.join(data_dir, "syscalls_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[ERROR] No CSV files found in {data_dir!r}")
        sys.exit(1)

    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=_USED_COLUMNS)
        print(f"  Loaded {os.path.basename(p):50s}  ({len(df):>7,} rows)")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows : {len(combined):,}")
    print(f"  Normal     : {(combined['label']==0).sum():,}")
    print(f"  Attack     : {(combined['label']==1).sum():,}")

    if exclude_comm:
        mask = combined["comm"].isin(exclude_comm)
        n_dropped = int(mask.sum())
        if n_dropped:
            dropped_breakdown = combined.loc[mask, "comm"].value_counts()
            print(f"\n  Filtering unrelated-process noise ({sorted(exclude_comm)}):")
            for name, cnt in dropped_breakdown.items():
                print(f"    - {name:<14} {cnt:>7,} rows dropped")
            combined = combined.loc[~mask].reset_index(drop=True)
            print(f"  Rows after filtering: {len(combined):,}")

    return combined


# feature engineering

"""
    Slide a window over sorted syscall stream → BoSC frequency vectors.

    Parameters
    ----------
    df          : DataFrame with syscall_name, pid, timestamp_ns, label.
    window_size : W — number of consecutive events per window.
    stride      : step size between windows.
    per_pid     : separate windows per PID (recommended).
    ngram       : 1 = unigrams only; 2 = unigrams + critical bigrams.

    Returns
    -------
    X : ndarray (n_windows, n_features) — raw frequency counts
    y : ndarray (n_windows,)            — 0=normal, 1=attack
    """

def make_windows(df: pd.DataFrame, window_size: int, stride: int,  per_pid: bool = True, ngram: int = 1) -> tuple:
    
    syscall_idx = {s: i for i, s in enumerate(KNOWN_SYSCALLS)}
    n_uni = len(KNOWN_SYSCALLS)
    n_features = n_uni + (N_BIGRAMS if ngram >= 2 else 0)

    X_rows, y_rows = [], []

    def _slide(events):
        # Incremental sliding window: O(1) amortized per event instead of
        # rebuilding each window's vector from scratch (O(window_size) per
        # window). Rebuilding from scratch made tuning/training effectively
        # quadratic in window_size and blew up badly on the one or two
        # outlier PIDs with 10,000+ events each. This mirrors the same
        # add-newest/remove-oldest technique bosc_realtime.py already uses
        # for live scoring.
        buf = deque()       # names currently in the window
        lbl_buf = deque()   # matching labels
        vec = np.zeros(n_features, dtype=np.float32)
        attack_count = 0

        for t, (name, lbl) in enumerate(events):
            idx = syscall_idx.get(name, -1)
            if idx >= 0:
                vec[idx] += 1
            if ngram >= 2 and buf:
                bi = BIGRAM_INDEX.get((buf[-1], name), -1)
                if bi >= 0:
                    vec[n_uni + bi] += 1
            buf.append(name)
            lbl_buf.append(lbl)
            attack_count += int(lbl == 1)

            if len(buf) > window_size:
                old_name = buf.popleft()
                old_lbl = lbl_buf.popleft()
                old_idx = syscall_idx.get(old_name, -1)
                if old_idx >= 0:
                    vec[old_idx] -= 1
                if ngram >= 2 and buf:
                    bi = BIGRAM_INDEX.get((old_name, buf[0]), -1)
                    if bi >= 0:
                        vec[n_uni + bi] -= 1
                attack_count -= int(old_lbl == 1)

            if len(buf) == window_size:
                start = t - window_size + 1
                if start % stride == 0:
                    X_rows.append(vec.copy())
                    y_rows.append(1 if attack_count > 0 else 0)

    if per_pid:
        df_s = df.sort_values(["pid", "timestamp_ns"]).reset_index(drop=True)
        for _, grp in df_s.groupby("pid"):
            if len(grp) < window_size:
                continue
            _slide(list(zip(grp["syscall_name"], grp["label"])))
    else:
        # Global mode must reflect the real chronological interleaving of
        # every process on the box — that's what RealtimeScorer sees live.
        # Sorting by pid first (as per-pid mode does) would group each
        # process's events into contiguous blocks instead, which is a
        # different distribution than the live stream and skews training.
        df_s = df.sort_values("timestamp_ns").reset_index(drop=True)
        _slide(list(zip(df_s["syscall_name"], df_s["label"])))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)

"""
    Fit TF-IDF on training windows, transform train (and optionally test).

    TF-IDF downweights common background syscalls (openat, mmap appear in
    nearly every window) and upweights rare attack indicators (setuid, ptrace).

    Returns (X_train_t, [X_test_t], transformer).
    """

def apply_tfidf(X_train: np.ndarray, X_test: np.ndarray = None):
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    X_tr = tfidf.fit_transform(X_train).toarray().astype(np.float32)
    if X_test is not None:
        X_te = tfidf.transform(X_test).toarray().astype(np.float32)
        return X_tr, X_te, tfidf
    return X_tr, tfidf


# Window parameter tuning 

def _undersample_normal(X, y, ratio: float, seed: int = 42):
    """Cap normal:attack windows at `ratio`:1 so CV stays fast on skewed data."""
    n_attack = int((y == 1).sum())
    if n_attack == 0:
        return X, y
    n_normal_target = int(n_attack * ratio)
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if len(normal_idx) > n_normal_target:
        rng = np.random.default_rng(seed)
        normal_idx = rng.choice(normal_idx, n_normal_target, replace=False)
    keep = np.concatenate([normal_idx, attack_idx])
    keep.sort()
    return X[keep], y[keep]

"""
Grid-search over (window_size, stride) pairs, scored by Attack F1 CV.

Must be searched in the SAME windowing mode (per_pid vs global) that
will actually be used for training — per-PID and global windows have
very different statistics, so a (W, stride) tuned in one mode is not
necessarily good in the other.

Normal windows are undersampled to `balance_ratio`:1 before CV — with
100k+ normal windows and a handful of attack windows, unbalanced CV is
both slow (RF on a huge skewed set per candidate) and a poor proxy for
the balanced training the final model actually gets.

Returns best_params dict and full results list.
"""

def tune_window_params(df: pd.DataFrame, ngram: int = 1, cv_folds: int = 5, per_pid: bool = True, window_candidates: list = None,
                       balance_ratio: float = 3.0) -> tuple:
    
    if window_candidates is None:
        window_candidates = [10, 20, 30, 50]
    candidates = [
        (W, max(1, int(W * frac)))
        for W in window_candidates
        for frac in [0.25, 0.5, 1.0]
    ]

    print(f"\n  {'W':>5} {'stride':>7} {'mean F1':>10} {'std':>7} {'n_windows':>10}")
    print("  " + "─" * 45)

    best_score, best_params, results = -1.0, {}, []
    for W, stride in candidates:
        X, y = make_windows(df, W, stride, per_pid=per_pid, ngram=ngram)
        if len(X) < 20 or y.sum() == 0 or (y == 0).sum() == 0:
            continue
        X, y = _undersample_normal(X, y, ratio=balance_ratio)
        rf = RandomForestClassifier(
            n_estimators=75, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y, cv=skf, scoring="f1", n_jobs=1)
        m, s = scores.mean(), scores.std()
        print(f"  {W:>5} {stride:>7} {m:>10.4f} {s:>7.4f} {len(X):>10,}")
        results.append({"window_size": W, "stride": stride,
                        "mean_f1": round(float(m), 4),
                        "std_f1":  round(float(s), 4)})
        if m > best_score:
            best_score, best_params = m, {"window_size": W, "stride": stride}

    print(f"\n  Best → W={best_params['window_size']}, "
          f"stride={best_params['stride']}, F1={best_score:.4f}")
    return best_params, results


# model training

def train_isolation_forest(X_normal: np.ndarray) -> IsolationForest:
    print("\nTraining IsolationForest on normal windows ...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expected fraction of anomalies in live stream
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_normal)
    print("    Done.")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, tune: bool = False, cv_folds: int = 5) -> tuple:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    if tune:
        print(f"\n[*] GridSearchCV for RandomForest "
              f"({cv_folds}-fold, scoring=f1) ...")
        param_grid = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [None, 10, 20],
            "min_samples_leaf": [1, 2, 5],
        }
        base = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        gs = GridSearchCV(base, param_grid, cv=skf, scoring="f1",
                          n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        params = gs.best_params_
        cv_f1 = gs.best_score_
        print(f"    Best params : {params}")
        print(f"    Best CV F1  : {cv_f1:.4f}")
        return best, params, cv_f1
    else:
        print(f"\n[*] Training RandomForest "
              f"({cv_folds}-fold StratifiedKFold CV) ...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring="f1")
        print(f"    CV F1 per fold : {[round(s, 4) for s in cv_scores]}")
        print(f"    CV F1 mean     : {cv_scores.mean():.4f} "
              f"± {cv_scores.std():.4f}")
        rf.fit(X_train, y_train)
        return rf, {"n_estimators": 200, "max_depth": None,
                    "min_samples_leaf": 1}, float(cv_scores.mean())
 

def save_models(model_dir: str, isoforest, rf, window_size: int, stride: int,
                ngram: int, use_tfidf: bool,
                per_pid: bool = True,
                tfidf_transformer=None,
                cv_f1: float = None,
                best_rf_params: dict = None,
                window_tune_results: list = None):
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(isoforest, os.path.join(model_dir, "isoforest.joblib"))
    joblib.dump(rf,        os.path.join(model_dir, "random_forest.joblib"))
    if use_tfidf and tfidf_transformer is not None:
        joblib.dump(tfidf_transformer,
                    os.path.join(model_dir, "tfidf.joblib"))

    bigrams_json = [[a, b] for a, b in BIGRAM_LIST] if ngram >= 2 else []

    meta = {
        "window_size":        window_size,
        "stride":             stride,
        "per_pid":            per_pid,
        "syscalls":           KNOWN_SYSCALLS,
        "n_features_unigram": len(KNOWN_SYSCALLS),
        "ngram":              ngram,
        "use_tfidf":          use_tfidf,
        "n_features":         len(KNOWN_SYSCALLS) + (N_BIGRAMS if ngram >= 2 else 0),
        "bigrams":            bigrams_json,
        "isoforest_path":     "isoforest.joblib",
        "rf_path":            "random_forest.joblib",
        "tfidf_path":         "tfidf.joblib" if use_tfidf else None,
        "ocsvm_path":         "isoforest.joblib",
        "cv_f1_mean":         round(cv_f1, 4) if cv_f1 is not None else None,
        "best_rf_params":     best_rf_params,
        "window_tune_results": window_tune_results,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Saved → {model_dir}/isoforest.joblib")
    print(f"  Saved → {model_dir}/random_forest.joblib")
    if use_tfidf:
        print(f"  Saved → {model_dir}/tfidf.joblib")
    print(f"  Saved → {model_dir}/meta.json")



def main():
    parser = argparse.ArgumentParser(
        description="Train BoSC anomaly detection models from eBPF-collected CSVs"
    )
    parser.add_argument("--data-dir",  default="data_collector/collected_data",
                        help="Directory containing syscalls_*.csv files")
    parser.add_argument("--model-dir", default="anomaly_detector/models",
                        help="Directory to save trained models")
    parser.add_argument("--window",    type=int, default=20,
                        help="Sliding window size in events (default: 20)")
    parser.add_argument("--stride",    type=int, default=10,
                        help="Window stride (default: 10)")
    parser.add_argument("--global-window", action="store_true",
                        help="Use global windows instead of per-PID")
    parser.add_argument("--ngram",     type=int, default=1, choices=[1, 2],
                        help="1=unigram only (default), 2=unigram+bigram")
    parser.add_argument("--tfidf",     action="store_true",
                        help="Apply TF-IDF weighting to count vectors")
    parser.add_argument("--tune-window", action="store_true",
                        help="Grid-search best (window, stride) before training")
    parser.add_argument("--window-candidates", type=int, nargs="+",
                        default=[10, 20, 30, 50, 100, 150],
                        help="Window sizes to try with --tune-window. Small "
                             "windows generate the most windows (slowest CV) "
                             "and perform worst in global mode, so narrowing "
                             "to the large-window range speeds up the search.")
    parser.add_argument("--tune-rf",   action="store_true",
                        help="GridSearchCV for RandomForest hyperparameters")
    parser.add_argument("--cv-folds",  type=int, default=5,
                        help="Number of StratifiedKFold CV folds (default: 5)")
    parser.add_argument("--balance",   action="store_true",
                        help="Undersample normal windows to fix class imbalance")
    parser.add_argument("--balance-ratio", type=float, default=2.0,
                        help="Max ratio of normal:attack windows after balancing "
                             "(default: 2.0 = 2x as many normal as attack)")
    parser.add_argument("--exclude-comm", nargs="*", default=list(DEFAULT_COMM_DENYLIST),
                        help="Process names (comm) to drop before windowing — "
                             "the collector hooks syscalls system-wide, so unrelated "
                             "processes running during collection (build tools, "
                             "editors, browsers, ...) get mislabeled as normal/attack. "
                             f"Default: {sorted(DEFAULT_COMM_DENYLIST)}. Pass "
                             "--exclude-comm with no values to disable filtering.")
    args = parser.parse_args()

    print("=" * 65)
    print("  BoSC Anomaly Detector — Training")
    print(f"  N-gram      : {args.ngram} "
          f"({'unigram' if args.ngram == 1 else 'unigram + bigram'})")
    print(f"  TF-IDF      : {args.tfidf}")
    print(f"  CV folds    : {args.cv_folds}")
    print(f"  Tune window : {args.tune_window}")
    print(f"  Tune RF     : {args.tune_rf}")
    print(f"  Balance     : {args.balance}"
          + (f" (ratio {args.balance_ratio}:1)" if args.balance else ""))
    print("=" * 65)

    # 1. Load data
    print("\n[1/6] Loading CSV files ...")
    df = load_csv_files(args.data_dir, exclude_comm=set(args.exclude_comm))

    # 2. Optionally tune window parameters
    window_tune_results = None
    window_size, stride = args.window, args.stride
    per_pid = not args.global_window

    if args.tune_window:
        print(f"\n[2/6] Tuning window parameters "
              f"(ngram={args.ngram}, per_pid={per_pid}) ...")
        best_wp, window_tune_results = tune_window_params(
            df, ngram=args.ngram, cv_folds=args.cv_folds, per_pid=per_pid,
            window_candidates=args.window_candidates,
        )
        window_size = best_wp["window_size"]
        stride      = best_wp["stride"]
        print(f"\n  Using → window_size={window_size}, stride={stride}")
    else:
        print(f"\n[2/6] Skipping window tuning "
              f"(W={window_size}, stride={stride})")

    # 3. Build windows
    print(f"\n[3/6] Building BoSC windows "
          f"(W={window_size}, stride={stride}, ngram={args.ngram}) ...")
    X, y = make_windows(df, window_size, stride,
                        per_pid=per_pid, ngram=args.ngram)
    n_uni = len(KNOWN_SYSCALLS)
    n_bi  = N_BIGRAMS if args.ngram >= 2 else 0
    print(f"  Feature dimensions : {n_uni} unigrams"
          + (f" + {n_bi} bigrams = {n_uni + n_bi} total" if n_bi else ""))
    print(f"  Total windows      : {len(X):,}")
    print(f"  Normal windows     : {(y == 0).sum():,}")
    print(f"  Attack windows     : {(y == 1).sum():,}")

    # 3b. Balance classes by undersampling normal windows
    if args.balance:
        n_attack = int((y == 1).sum())
        n_normal_target = int(n_attack * args.balance_ratio)
        n_normal_current = int((y == 0).sum())
        if n_normal_current > n_normal_target:
            print(f"\n  Balancing: undersampling normal windows "
                  f"{n_normal_current:,} → {n_normal_target:,} "
                  f"(ratio {args.balance_ratio}:1 vs attack {n_attack:,})")
            rng = np.random.default_rng(42)
            normal_idx  = np.where(y == 0)[0]
            attack_idx  = np.where(y == 1)[0]
            chosen_normal = rng.choice(normal_idx, n_normal_target, replace=False)
            keep = np.concatenate([chosen_normal, attack_idx])
            keep.sort()
            X, y = X[keep], y[keep]
            print(f"  After balancing → Normal: {(y==0).sum():,}  "
                  f"Attack: {(y==1).sum():,}  Total: {len(X):,}")
        else:
            print(f"  Balance: already balanced ({n_normal_current:,} normal "
                  f"vs {n_attack:,} attack), skipping.")

    # 4. Train/test split (stratified)
    print(f"\n[4/6] Stratified 80/20 train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_normal_train = X_train[y_train == 0]
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # 5. Apply TF-IDF if requested
    tfidf_transformer = None
    if args.tfidf:
        print("\n  Applying TF-IDF weighting ...")
        X_train, X_test, tfidf_transformer = apply_tfidf(X_train, X_test)
        X_normal_train = X_train[y_train == 0]
        print(f"  TF-IDF applied — feature range: "
              f"[{X_train.min():.3f}, {X_train.max():.3f}]")

    # Save test split for evaluator
    os.makedirs(args.model_dir, exist_ok=True)
    np.save(os.path.join(args.model_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(args.model_dir, "y_test.npy"),  y_test)
    np.save(os.path.join(args.model_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.model_dir, "y_train.npy"), y_train)
    print("  Splits saved for evaluator.")

    # 6. Train models
    print("\n[5/6] Training models ...")
    isoforest = train_isolation_forest(X_normal_train)
    rf, best_rf_params, cv_f1 = train_random_forest(
        X_train, y_train,
        tune=args.tune_rf,
        cv_folds=args.cv_folds,
    )

    # 7. Save
    print("\n[6/6] Saving models ...")
    save_models(
        model_dir=args.model_dir,
        isoforest=isoforest,
        rf=rf,
        window_size=window_size,
        stride=stride,
        per_pid=per_pid,
        ngram=args.ngram,
        use_tfidf=args.tfidf,
        tfidf_transformer=tfidf_transformer,
        cv_f1=cv_f1,
        best_rf_params=best_rf_params,
        window_tune_results=window_tune_results,
    )

    print("  Training complete!")
    print(f"  RandomForest CV F1 : {cv_f1:.4f}")
    print("  Run bosc_evaluate.py for full metrics and plots.")


if __name__ == "__main__":
    main()
