#!/usr/bin/env python3
"""
bosc_report.py — HTML Report Generator

Reads the saved plots and model metadata to produce a self-contained,
single-page HTML report summarising the BoSC anomaly detection results.

Usage
    python3 anomaly_detector/bosc_report.py \\
        --model-dir anomaly_detector/models \\
        --plot-dir  anomaly_detector/plots  \\
        --output    anomaly_detector/report.html

Then open report.html in any browser.
"""

import argparse
import base64
import json
import os

import joblib
import numpy as np
from sklearn.metrics import ( classification_report, roc_auc_score,)
from sklearn.preprocessing import normalize


# Embed image as base64 

def _img_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def _gather_metrics(model_dir: str):
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # The Isolation Forest is a rejected baseline, absent from the shipped
    # RF-only model (train with --with-isoforest to include it).
    iso_rel = meta.get("isoforest_path") or meta.get("ocsvm_path")
    iso_model = None
    if iso_rel and os.path.exists(os.path.join(model_dir, iso_rel)):
        iso_model = joblib.load(os.path.join(model_dir, iso_rel))
    rf        = joblib.load(os.path.join(model_dir, meta["rf_path"]))

    # Optional TF-IDF
    tfidf = None
    if meta.get("use_tfidf") and meta.get("tfidf_path"):
        tpath = os.path.join(model_dir, meta["tfidf_path"])
        if os.path.exists(tpath):
            tfidf = joblib.load(tpath)

    X_test  = np.load(os.path.join(model_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(model_dir, "y_test.npy"))
    X_train = np.load(os.path.join(model_dir, "X_train.npy"))
    y_train = np.load(os.path.join(model_dir, "y_train.npy"))

    # RF
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_scores = rf_proba[:, 1] if rf_proba.shape[1] > 1 else rf_proba[:, 0]

    def _report_dict(y_true, y_pred, scores):
        rep = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Attack"],
            output_dict=True,
        )
        auc = roc_auc_score(y_true, scores)
        return rep, auc

    rf_rep,  rf_auc  = _report_dict(y_test, rf_pred,  rf_scores)

    if iso_model is not None:
        iso_raw    = iso_model.predict(X_test)
        iso_scores = -iso_model.decision_function(X_test)
        iso_pred   = np.where(iso_raw == -1, 1, 0)
        iso_rep, iso_auc = _report_dict(y_test, iso_pred, iso_scores)
        iso_label = "IsolationForest"
    else:
        iso_rep, iso_auc, iso_label = {}, float("nan"), "IsolationForest (not evaluated)"

    return {
        "meta":           meta,
        "X_train":        X_train,
        "y_train":        y_train,
        "X_test":         X_test,
        "y_test":         y_test,
        "iso_rep":        iso_rep,
        "iso_auc":        iso_auc,
        "iso_label":      iso_label,
        "rf_rep":         rf_rep,
        "rf_auc":         rf_auc,
        "rf_importances": rf.feature_importances_,
        # legacy keys for compatibility
        "ocsvm_rep":  iso_rep,
        "ocsvm_auc":  iso_auc,
    }


# Metric card HTML 

def _metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="card metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def _table_row(cells: list, header: bool = False) -> str:
    tag = "th" if header else "td"
    inner = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
    return f"<tr>{inner}</tr>"


def build_html(data: dict, plot_dir: str) -> str:
    meta      = data["meta"]
    iso_rep   = data["iso_rep"]
    iso_label = data["iso_label"]
    rf_rep    = data["rf_rep"]
    syscalls  = meta["syscalls"]
    bigrams   = meta.get("bigrams", [])
    importances = data["rf_importances"]
    ngram     = meta.get("ngram", 1)
    use_tfidf = meta.get("use_tfidf", False)
    cv_f1     = meta.get("cv_f1_mean")

    n_train_normal = int((data["y_train"] == 0).sum())
    n_train_attack = int((data["y_train"] == 1).sum())
    n_test_normal  = int((data["y_test"]  == 0).sum())
    n_test_attack  = int((data["y_test"]  == 1).sum())

    # Build feature names: unigrams + bigrams
    all_feature_names = list(syscalls)
    if ngram >= 2:
        all_feature_names += [f"{a}\u2192{b}" for a, b in bigrams]

    # Top 5 important features
    top5_idx = np.argsort(importances)[::-1][:5]
    BAR_CHAR = "\u2588"
    top5_rows = "".join(
        _table_row([
            f"#{rank+1}",
            all_feature_names[i] if i < len(all_feature_names) else f"feat_{i}",
            f"{importances[i]:.4f}",
            BAR_CHAR * int(importances[i] * 200),
        ])
        for rank, i in enumerate(top5_idx)
    )

    # Metric rows for both models
    def _model_table(rep, auc):
        rows = ""
        for cls in ["Normal", "Attack", "macro avg", "weighted avg"]:
            if cls in rep:
                r = rep[cls]
                rows += _table_row([
                    cls,
                    f"{r.get('precision', 0):.4f}",
                    f"{r.get('recall', 0):.4f}",
                    f"{r.get('f1-score', 0):.4f}",
                    f"{r.get('support', 0):.0f}",
                ])
        rows += _table_row(["ROC-AUC", "-", "-", f"{auc:.4f}", "-"])
        return rows

    iso_table = _model_table(iso_rep, data["iso_auc"])
    rf_table  = _model_table(rf_rep,  data["rf_auc"])

    # Images
    imgs = {
        k: _img_b64(os.path.join(plot_dir, f"{k}.png"))
        for k in ["syscall_distribution", "cm_isoforest", "cm_rf",
                  "roc_curves", "pr_curves", "feature_importance", "tsne"]
    }

    def _plot_section(key: str, title: str, caption: str) -> str:
        src = imgs.get(key, "")
        if not src:
            return ""
        return f"""
        <div class="card plot-card">
          <h3>{title}</h3>
          <img src="{src}" alt="{title}" />
          <p class="caption">{caption}</p>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BoSC Anomaly Detection - Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {{
    --bg:       #111111;
    --surface:  #1e1e1e;
    --surface2: #2a2a2a;
    --accent:   #ffffff;
    --accent2:  #bbbbbb;
    --green:    #cccccc;
    --red:      #eeeeee;
    --text:     #eeeeee;
    --muted:    #aaaaaa;
    --border:   rgba(255,255,255,0.12);
    --glow:     rgba(255,255,255,0.06);
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    padding: 0 0 60px;
  }}

  /* Header */
  header {{
    background: #1a1a1a;
    border-bottom: 2px solid var(--border);
    padding: 48px 60px 36px;
    position: relative;
    overflow: hidden;
  }}
  header .badge {{
    display: inline-block;
    background: #ffffff;
    color: #111111;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 14px;
  }}
  header h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
  }}
  header p {{
    color: var(--muted);
    font-size: 13px;
  }}

  /* ── Layout ── */
  main {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 0; }}
  section {{ margin-bottom: 48px; }}
  section > h2 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: .5px;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Cards ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 26px;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
    transition: box-shadow .2s;
  }}
  .card:hover {{ box-shadow: 0 8px 32px var(--glow); }}

  /* ── Metric cards ── */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 14px;
  }}
  .metric-card {{ text-align: center; padding: 20px 12px; }}
  .metric-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }}
  .metric-value {{ font-size: 2rem; font-weight: 700; color: var(--accent); font-family: 'JetBrains Mono', monospace; }}
  .metric-sub   {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

  /* ── Tables ── */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
  }}
  th {{
    background: var(--surface2);
    color: #ffffff;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .8px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    color: var(--text);
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.04); }}

  /* ── Two-column table layout ── */
  .models-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 700px) {{ .models-grid {{ grid-template-columns: 1fr; }} }}

  .model-label {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    color: var(--muted);
  }}
  .model-label span {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ffffff;
    margin-right: 6px;
    vertical-align: middle;
  }}

  /* ── Plot cards ── */
  .plots-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 20px;
  }}
  .plot-card img {{ width: 100%; border-radius: 8px; margin-top: 12px; }}
  .plot-card h3 {{ font-size: 13px; font-weight: 600; color: var(--muted); }}
  .caption {{ font-size: 11px; color: var(--muted); margin-top: 8px; font-style: italic; }}

  /* ── Pipeline diagram ── */
  .pipeline {{
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin-top: 4px;
  }}
  .pipeline-step {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 500;
    text-align: center;
    min-width: 110px;
  }}
  .pipeline-arrow {{
    color: var(--accent);
    font-size: 18px;
    padding: 0 6px;
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <div class="badge">eBPF Security Monitor</div>
  <h1>Bag-of-System-Calls Anomaly Detection Report</h1>
  <p>Window size: {meta["window_size"]} events &nbsp;|&nbsp; Stride: {meta["stride"]} &nbsp;|&nbsp; Features: {meta["n_features"]} syscalls</p>
</header>

<main>

<!-- ── Overview Metrics ── -->
<section>
  <h2>Dataset Overview</h2>
  <div class="metrics-grid">
    {_metric_card("Training Windows", f"{n_train_normal + n_train_attack:,}", f"{n_train_normal:,} normal / {n_train_attack:,} attack")}
    {_metric_card("Test Windows",     f"{n_test_normal + n_test_attack:,}",   f"{n_test_normal:,} normal / {n_test_attack:,} attack")}
    {_metric_card("Window Size",      str(meta["window_size"]), "events per window")}
    {_metric_card("Stride",           str(meta["stride"]),      "events between windows")}
    {_metric_card("N-gram",           str(ngram),               "unigram + bigram" if ngram >= 2 else "unigram only")}
    {_metric_card("Features",         str(meta["n_features"]),  "BoSC dimensions")}
    {_metric_card("RF AUC",           f"{data['rf_auc']:.3f}",  "Random Forest")}
    {_metric_card(iso_label + " AUC", f"{data['iso_auc']:.3f}", "Novelty detector")}
    {_metric_card("RF CV F1",         f"{cv_f1:.3f}" if cv_f1 else "N/A", "5-fold cross-val")}
  </div>
</section>

<!-- ── Pipeline ── -->
<section>
  <h2>BoSC Pipeline</h2>
  <div class="card">
    <div class="pipeline">
      <div class="pipeline-step">eBPF<br/>Kernel Hook</div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">Syscall<br/>Event Stream</div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">Global<br/>Window (W={meta["window_size"]})</div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">BoSC<br/>Vector {'(+bigrams)' if ngram >= 2 else ''}</div>
      <div class="pipeline-arrow">&rarr;</div>
      {'<div class="pipeline-step">TF-IDF<br/>Weighting</div><div class="pipeline-arrow">&rarr;</div>' if use_tfidf else ''}
      <div class="pipeline-step">{iso_label}</div>
      <div class="pipeline-arrow">+</div>
      <div class="pipeline-step">Random<br/>Forest</div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step" style="background:#2a2a2a;border-color:#ffffff;font-weight:700;">Alert</div>
    </div>
  </div>
</section>

<!-- ── Model Performance ── -->
<section>
  <h2>Model Performance (Test Split - 20%)</h2>
  <div class="models-grid">
    <div class="card">
      <div class="model-label">
        <span style="background:#ffffff;"></span>{iso_label}
        &nbsp;<small>(novelty / zero-day detection)</small>
      </div>
      <div class="table-wrap">
        <table>
          {_table_row(["Class", "Precision", "Recall", "F1-Score", "Support"], header=True)}
          {iso_table}
        </table>
      </div>
    </div>
    <div class="card">
      <div class="model-label">
        <span style="background:#ffffff;"></span>Random Forest
        &nbsp;<small>(supervised / known-attack detection)</small>
      </div>
      <div class="table-wrap">
        <table>
          {_table_row(["Class", "Precision", "Recall", "F1-Score", "Support"], header=True)}
          {rf_table}
        </table>
      </div>
    </div>
  </div>
</section>

<!-- ── Feature Importance ── -->
<section>
  <h2>Top-5 Discriminative Syscalls (Random Forest)</h2>
  <div class="card">
    <div class="table-wrap">
      <table>
        {_table_row(["Rank", "Syscall", "Gini Importance", "Relative Weight"], header=True)}
        {top5_rows}
      </table>
    </div>
  </div>
</section>

<!-- ── Plots ── -->
<section>
  <h2>Visualisations</h2>
  <div class="plots-grid">
    {_plot_section("syscall_distribution", "Syscall Frequency: Normal vs Attack",
      "Mean syscall count per window. Privilege-escalation syscalls (setuid, ptrace, capset) appear exclusively in attack windows.")}
    {_plot_section("roc_curves", "ROC Curves",
      "Receiver Operating Characteristic. Higher AUC = better discrimination between normal and attack windows.")}
    {_plot_section("pr_curves", "Precision-Recall Curves",
      "More informative than ROC for security: shows the tradeoff between catching attacks (recall) and false alarms (precision).")}
    {_plot_section("cm_isoforest", f"Confusion Matrix - {iso_label}",
      f"Trained on normal-only data. Rows = actual class, Columns = predicted class.")}
    {_plot_section("cm_ocsvm", f"Confusion Matrix - {iso_label} (alt)",
      "Alternative confusion matrix view.")}
    {_plot_section("cm_rf", "Confusion Matrix - Random Forest",
      "Rows: actual class. Columns: predicted class. Trained on both normal and attack data.")}
    {_plot_section("feature_importance", "Feature Importance (Random Forest)",
      "Gini importance. Red bars = above-median importance. Includes unigram and bigram BoSC features.")}
    {_plot_section("tsne", "t-SNE Embedding of All Windows",
      f"2-D t-SNE projection of {meta['window_size']}-event frequency vectors. Green = normal, Red = attack.")}
  </div>
</section>

</main>

<footer>
  <p>eBPF-Based Runtime Security Monitor &nbsp;|&nbsp; BoSC Method</p>
</footer>

</body>
</html>"""

    return html



def main():
    parser = argparse.ArgumentParser(
        description="Generate self-contained HTML report for BoSC results"
    )
    parser.add_argument("--model-dir", default="anomaly_detector/models")
    parser.add_argument("--plot-dir",  default="anomaly_detector/plots")
    parser.add_argument("--output",    default="anomaly_detector/report.html")
    args = parser.parse_args()

    print(" Gathering metrics ...")
    data = _gather_metrics(args.model_dir)

    print("Building HTML report ...")
    html = build_html(data, args.plot_dir)

    with open(args.output, "w") as f:
        f.write(html)

    print(f" Report saved → {args.output}")
    print(f"    Open in browser:  xdg-open {args.output}")


if __name__ == "__main__":
    main()
