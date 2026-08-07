#!/usr/bin/env python3

import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             roc_auc_score, precision_recall_curve, average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bosc_train import make_windows, apply_tfidf, _undersample_normal, KNOWN_SYSCALLS, N_BIGRAMS, BIGRAM_LIST
from per_technique_eval import load as pt_load, window_dominant_technique

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})

BT={"cc1","gcc","cc","as","ld","collect2","make","cmake","ninja"}
def load(p):
    d=pd.read_csv(p); return d[~d["comm"].isin(BT)].sort_values("timestamp_ns").reset_index(drop=True)

tr=load("data_collector/interleaved/labeled_session1.csv")
te=load("data_collector/interleaved/labeled_session2.csv")
W,S=200,50
Xtr,ytr=make_windows(tr,W,S,per_pid=False,ngram=2,min_attack=1,min_attack_frac=0.1)
Xte,yte=make_windows(te,W,S,per_pid=False,ngram=2,min_attack=1,min_attack_frac=0.1)
dom=np.array(window_dominant_technique(te,W,S,len(yte)))
Xtr_t,Xte_t,_=apply_tfidf(Xtr,Xte)
Xb,yb=_undersample_normal(Xtr_t,ytr,ratio=2.0)
rf=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1).fit(Xb,yb)
proba=rf.predict_proba(Xte_t)[:,1]
pred=(proba>=0.5).astype(int)

cm=confusion_matrix(yte,pred); tn,fp,fn,tp=cm.ravel()
auc=roc_auc_score(yte,proba); ap=average_precision_score(yte,proba)
print(f"Interleaved S1->S2  W={W} frac=0.1 thr=0.5")
print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  AUC={auc:.3f}  AP={ap:.3f}  test windows={len(yte)}  attack={int(yte.sum())}")

# confusion matrix
fig,ax=plt.subplots(figsize=(4,3.6))
ConfusionMatrixDisplay(cm,display_labels=["Normal","Attack"]).plot(ax=ax,colorbar=False,cmap="Blues")
ax.set_title("Random Forest (interleaved, S1$\\rightarrow$S2)")
fig.tight_layout(); fig.savefig(f"{OUT}/il_cm_rf.png",bbox_inches="tight"); plt.close(fig)

# ROC
fpr_,tpr_,_=roc_curve(yte,proba)
fig,ax=plt.subplots(figsize=(4.4,3.8))
ax.plot(fpr_,tpr_,lw=2,color="#2196F3",label=f"Random Forest (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC (interleaved cross-session)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/il_roc.png",bbox_inches="tight"); plt.close(fig)

# PR
prec,rec,_=precision_recall_curve(yte,proba)
fig,ax=plt.subplots(figsize=(4.4,3.8))
ax.plot(rec,prec,lw=2,color="#FF9800",label=f"Random Forest (AP={ap:.3f})")
base=yte.mean(); ax.axhline(base,color="k",ls="--",lw=1,alpha=.4,label=f"Baseline ({base:.2f})")
ax.set_xlabel("Recall (Attack)"); ax.set_ylabel("Precision (Attack)")
ax.set_title("Precision-Recall (interleaved)"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/il_pr.png",bbox_inches="tight"); plt.close(fig)

# feature importance (top 20)
names=list(KNOWN_SYSCALLS)+[f"{a}$\\rightarrow${b}" for a,b in BIGRAM_LIST]
imp=rf.feature_importances_; idx=np.argsort(imp)[::-1][:20]
fig,ax=plt.subplots(figsize=(7,3.6))
ax.bar(range(20),imp[idx],color="#F44336",edgecolor="white",lw=.5)
ax.set_xticks(range(20)); ax.set_xticklabels([names[i] for i in idx],rotation=55,ha="right",fontsize=7)
ax.set_ylabel("Gini importance"); ax.set_title("Top Random Forest features (interleaved)")
fig.tight_layout(); fig.savefig(f"{OUT}/il_feature_importance.png",bbox_inches="tight"); plt.close(fig)

# per-technique recall bar
atk=yte==1
rows=[]
for t in sorted(set(dom[atk])):
    m=atk&(dom==t); n=int(m.sum())
    if n>=15: rows.append((t,pred[m].mean(),n))  # skip tiny-n techniques
rows.sort(key=lambda r:r[1])
fig,ax=plt.subplots(figsize=(7,3.8))
cols=["#F44336" if r[1]<0.7 else "#4CAF50" for r in rows]
ax.barh([r[0] for r in rows],[r[1] for r in rows],color=cols,edgecolor="white")
ax.set_xlabel("Detection rate (recall)"); ax.set_xlim(0,1.02)
ax.set_title("Per-technique detection (interleaved, thr=0.5)")
for i,r in enumerate(rows): ax.text(r[1]+.01,i,f"{r[1]:.2f}",va="center",fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/il_per_technique.png",bbox_inches="tight"); plt.close(fig)

print("Saved: il_cm_rf, il_roc, il_pr, il_feature_importance, il_per_technique  -> report/figures/")
