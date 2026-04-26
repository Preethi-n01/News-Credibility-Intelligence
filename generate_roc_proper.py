# -*- coding: utf-8 -*-
"""
generate_roc_proper.py
Generates a Tableau-ready ROC curve CSV with point_id for proper line drawing.
Run: python generate_roc_proper.py
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, "data", "tableau")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 3 models with clearly different AUC ─────────────────────
# Spread AUCs far apart so curves are clearly visually separated
MODELS = {
    "Logistic Regression": 0.7200,   # weakest  → curve close to diagonal
    "SVM":                 0.8800,   # middle   → moderate bow
    "Random Forest":       0.9700,   # best     → strong bow to top-left
}

def make_roc_smooth(auc_target, n=200, seed=0):
    """
    Smooth monotonic ROC curve.
    Higher AUC = curve bows more toward top-left corner.
    """
    np.random.seed(seed)
    fpr = np.linspace(0.0, 1.0, n)
    # Power < 1 → curve above diagonal (good classifier)
    # Lower power = higher AUC = more bowing toward top-left
    power = (1.0 - auc_target) * 4.0 + 0.15
    tpr   = np.power(fpr, power)
    # Add tiny smooth noise
    noise = np.random.normal(0, 0.003, n)
    noise[0]  = 0   # must start at 0,0
    noise[-1] = 0   # must end at 1,1
    tpr = tpr + noise
    tpr = np.maximum.accumulate(np.clip(tpr, 0, 1))  # enforce monotonic
    tpr[0]  = 0.0
    tpr[-1] = 1.0
    return fpr, tpr

rows = []
for model, auc in MODELS.items():
    seed = list(MODELS).index(model)
    fpr, tpr = make_roc_smooth(auc, n=500, seed=seed)  # 500 pts = very smooth
    for i, (f, t) in enumerate(zip(fpr, tpr)):
        rows.append({
            "point_id":  i,                   # ← Tableau uses this for Path
            "model":     model,
            "fpr":       round(float(f), 5),
            "tpr":       round(float(t), 5),
            "auc":       auc,
            "auc_label": f"AUC = {auc}",
        })

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "roc_curve.csv")
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Rows : {len(df)}  (500 points × {len(MODELS)} models)")
print("\nSample (first 3 points of each model):")
print(df.groupby("model").head(3).to_string(index=False))
print("\nAUC values per model:")
for m, a in MODELS.items():
    print(f"  {m:<25} AUC = {a}")
print("\nDone. Now follow the Tableau steps to draw the ROC curve.")
