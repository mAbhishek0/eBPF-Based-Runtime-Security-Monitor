#!/usr/bin/env python3
"""
bosc_evaluate.py — Offline Evaluation of BoSC Models

Loads the saved models and test split from bosc_train.py and produces:
  1. Console report — precision, recall, F1, ROC-AUC for both models
  2. Confusion matrices (PNG)
  3. ROC curves (PNG)
  4. Precision-Recall curves (PNG)  ← new: more informative for security
  5. Feature importance bar chart for Random Forest (PNG)
  6. t-SNE 2-D embedding coloured by label (PNG)
  7. Syscall frequency comparison — Normal vs Attack (PNG)

All plots are saved to anomaly_detector/plots/.

"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})
PALETTE = {"normal": "#4CAF50", "attack": "#F44336"}


def load_artefacts(model_dir: str):
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # The Isolation Forest is a rejected baseline and is absent from the
    # shipped RF-only model (train with --with-isoforest to include it).
    isoforest = None
    iso_rel = meta.get("isoforest_path") or meta.get("ocsvm_path")
    if iso_rel and os.path.exists(os.path.join(model_dir, iso_rel)):
        isoforest = joblib.load(os.path.join(model_dir, iso_rel))
    rf = joblib.load(os.path.join(model_dir, meta["rf_path"]))

    # Optional TF-IDF transformer
    tfidf = None
    if meta.get("use_tfidf") and meta.get("tfidf_path"):
        tpath = os.path.join(model_dir, meta["tfidf_path"])
        if os.path.exists(tpath):
            tfidf = joblib.load(tpath)

    X_test  = np.load(os.path.join(model_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(model_dir, "y_test.npy"))
    X_train = np.load(os.path.join(model_dir, "X_train.npy"))
    y_train = np.load(os.path.join(model_dir, "y_train.npy"))

    return isoforest, rf, tfidf, meta, X_test, y_test, X_train, y_train



def evaluate_isoforest(model, X_test, y_test):
    raw_pred = model.predict(X_test)          # +1 / -1
    # decision_function: lower = more anomalous → negate for "attack probability"
    scores = -model.decision_function(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)
    return y_pred, scores


def evaluate_rf(rf, X_test):
    y_pred = rf.predict(X_test)
    proba  = rf.predict_proba(X_test)
    scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return y_pred, scores



def plot_confusion(y_true, y_pred, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_roc(models_data: list, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = ["#2196F3", "#FF9800", "#9C27B0"]
    for (name, y_true, scores), color in zip(models_data, colors):
        auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — BoSC Models", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_precision_recall(models_data: list, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = ["#2196F3", "#FF9800", "#9C27B0"]
    for (name, y_true, scores), color in zip(models_data, colors):
        ap = average_precision_score(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{name}  (AP={ap:.3f})")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.4,
               label=f"Random  ({baseline:.2f})")
    ax.set_xlabel("Recall (Attack)")
    ax.set_ylabel("Precision (Attack)")
    ax.set_title("Precision-Recall Curves — BoSC Models", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_feature_importance(rf, syscalls: list, bigrams: list,
                            n_top: int, out_path: str):
    importances = rf.feature_importances_
    # Build display names
    all_names = list(syscalls)
    if bigrams:
        all_names += [f"{a}→{b}" for a, b in bigrams]

    n_show = min(n_top, len(all_names))
    idx = np.argsort(importances)[::-1][:n_show]
    top_names = [all_names[i] for i in idx]
    top_imp   = importances[idx]

    med = np.median(importances)
    colors = ["#F44336" if v > med else "#90CAF9" for v in top_imp]

    fig, ax = plt.subplots(figsize=(max(10, n_show * 0.55), 5))
    bars = ax.bar(range(n_show), top_imp, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(top_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Gini Importance")
    ax.set_title("Random Forest — Top Feature Importances", fontweight="bold")

    for rank, bar in enumerate(bars[:5]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"#{rank+1}", ha="center", va="bottom", fontsize=7, color="#333")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_syscall_distribution(X_train, y_train, syscalls, bigrams,
                              n_show, out_path):
    X_normal = X_train[y_train == 0]
    X_attack = X_train[y_train == 1]

    all_names = list(syscalls)
    if bigrams:
        all_names += [f"{a}→{b}" for a, b in bigrams]

    # Show top n_show by total mean count
    totals = X_train.mean(axis=0)
    top_idx = np.argsort(totals)[::-1][:n_show]

    mean_n = X_normal.mean(axis=0)[top_idx]
    mean_a = X_attack.mean(axis=0)[top_idx] if len(X_attack) > 0 else np.zeros(n_show)
    names  = [all_names[i] for i in top_idx]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, mean_n, width, label="Normal",
           color=PALETTE["normal"], alpha=0.85)
    ax.bar(x + width/2, mean_a, width, label="Attack",
           color=PALETTE["attack"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("Mean count per window")
    ax.set_title("Syscall/Bigram Frequency — Normal vs Attack Windows",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_tsne(X, y, out_path: str, max_samples: int = 2000):
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    print(f"  Running t-SNE on {len(X)} windows ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_2d = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, color, label in [(0, PALETTE["normal"], "Normal"),
                               (1, PALETTE["attack"],  "Attack")]:
        mask = y == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=label, alpha=0.55, s=18, edgecolors="none")
    ax.set_title("t-SNE of BoSC Windows (Normal vs Attack)", fontweight="bold")
    ax.legend(markerscale=2)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")



def print_report(name: str, y_true, y_pred, scores):
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"], digits=4
    ))
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Avg Precision    : {ap:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved BoSC models on the held-out test split"
    )
    parser.add_argument("--model-dir", default="anomaly_detector/models")
    parser.add_argument("--plot-dir",  default="anomaly_detector/plots")
    parser.add_argument("--n-top",     type=int, default=30,
                        help="Number of top features to show in importance plot")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print("=" * 65)
    print("  BoSC Anomaly Detector — Offline Evaluation")
    print("=" * 65)

    # 1. Load
    print("\n[1/7] Loading models and test data ...")
    isoforest, rf, tfidf, meta, X_test, y_test, X_train, y_train = \
        load_artefacts(args.model_dir)

    syscalls = meta["syscalls"]
    bigrams  = [tuple(b) for b in meta.get("bigrams", [])]
    ngram    = meta.get("ngram", 1)

    print(f"  N-gram        : {ngram}")
    print(f"  TF-IDF        : {meta.get('use_tfidf', False)}")
    print(f"  Feature dims  : {meta['n_features']}")
    if meta.get("cv_f1_mean"):
        print(f"  Train CV F1   : {meta['cv_f1_mean']:.4f}")
    if meta.get("best_rf_params"):
        print(f"  Best RF params: {meta['best_rf_params']}")
    print(f"  Test windows  : {len(X_test):,}")
    print(f"  Normal/Attack : {(y_test==0).sum()} / {(y_test==1).sum()}")

    # 2. Score
    print("\n[2/7] Scoring test windows ...")
    rf_pred,  rf_scores  = evaluate_rf(rf, X_test)
    has_iso = isoforest is not None
    if has_iso:
        iso_pred, iso_scores = evaluate_isoforest(isoforest, X_test, y_test)
    else:
        print("  (Isolation Forest not present in model; RF-only evaluation.)")

    # 3. Console metrics
    print("\n[3/7] Computing metrics ...")
    if has_iso:
        print_report("IsolationForest (novelty detector)", y_test, iso_pred, iso_scores)
    print_report("Random Forest (supervised)",          y_test, rf_pred,  rf_scores)

    # 4. Confusion matrices
    print("\n[4/7] Plotting confusion matrices ...")
    if has_iso:
        plot_confusion(y_test, iso_pred, "IsolationForest Confusion Matrix",
                       os.path.join(args.plot_dir, "cm_isoforest.png"))
    plot_confusion(y_test, rf_pred, "RandomForest Confusion Matrix",
                   os.path.join(args.plot_dir, "cm_rf.png"))

    # 5. ROC + PR curves
    print("\n[5/7] Plotting ROC and Precision-Recall curves ...")
    curves_data = [("RandomForest", y_test, rf_scores)]
    if has_iso:
        curves_data.insert(0, ("IsolationForest", y_test, iso_scores))
    plot_roc(curves_data, os.path.join(args.plot_dir, "roc_curves.png"))
    plot_precision_recall(curves_data,
                          os.path.join(args.plot_dir, "pr_curves.png"))

    # 6. Feature importance + distribution
    print("\n[6/7] Plotting feature importance and syscall distribution ...")
    plot_feature_importance(
        rf, syscalls, bigrams,
        n_top=args.n_top,
        out_path=os.path.join(args.plot_dir, "feature_importance.png"),
    )
    plot_syscall_distribution(
        X_train, y_train, syscalls, bigrams,
        n_show=min(args.n_top, meta["n_features"]),
        out_path=os.path.join(args.plot_dir, "syscall_distribution.png"),
    )

    # 7. t-SNE
    print("\n[7/7] Generating t-SNE visualisation ...")
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    plot_tsne(X_all, y_all, os.path.join(args.plot_dir, "tsne.png"))

    print(f"  All plots saved to: {args.plot_dir}/")
    print("Run bosc_report.py to regenerate the HTML report.")


if __name__ == "__main__":
    main()
