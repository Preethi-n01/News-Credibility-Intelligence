# -*- coding: utf-8 -*-
"""
generate_tableau_data.py
========================
Generates a full suite of rich CSVs inside  data/tableau/
for use in Tableau dashboards covering the News Credibility Project.

Run from the project root:
    python generate_tableau_data.py

CSVs produced
----------------------------------------------------------------------
1.  model_performance.csv           Model accuracy/precision/recall/F1/AUC
2.  model_comparison_radar.csv      Same metrics reshaped for radar charts
3.  confusion_matrix.csv            TP/FP/TN/FN per model
4.  per_class_metrics.csv           Per-class P/R/F1 per model
5.  roc_curve.csv                   ROC FPR/TPR per model
6.  pr_curve.csv                    Precision-Recall curve per model
7.  prediction_confidence.csv       Confidence histogram buckets per model
8.  news_distribution.csv           Fake vs Real record counts & text stats
9.  clickbait_distribution.csv      Clickbait vs Not-Clickbait counts
10. clickbait_predictions.csv       Full clickbait prediction table
11. sentiment_distribution.csv      Sentiment label counts across corpus
12. sentiment_by_class.csv          Sentiment breakdown inside fake vs real
13. top_terms.csv                   Top 20 TF-IDF terms per label
14. feature_importance.csv          LR coefficients as feature importance
15. word_length_distribution.csv    Text length histogram by label
16. dataset_summary.csv             High-level KPI card metrics
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(DATA_DIR, "tableau")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 62)
print("  News Credibility -- Tableau Data Generator")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading models ...")
fake_models = {
    "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "fake_logistic_regression.pkl"), "rb")),
    "Random Forest":       pickle.load(open(os.path.join(MODEL_DIR, "fake_random_forest.pkl"),       "rb")),
    "SVM":                 pickle.load(open(os.path.join(MODEL_DIR, "fake_svm.pkl"),                 "rb")),
}
fake_vec  = pickle.load(open(os.path.join(MODEL_DIR, "fake_vectorizer.pkl"),      "rb"))
click_mdl = pickle.load(open(os.path.join(MODEL_DIR, "click_model.pkl"),          "rb"))
click_vec = pickle.load(open(os.path.join(MODEL_DIR, "click_vectorizer.pkl"),     "rb"))
sent_mdl  = pickle.load(open(os.path.join(MODEL_DIR, "sentiment_model.pkl"),      "rb"))
sent_vec  = pickle.load(open(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"), "rb"))
print("   All models loaded.")

# ─────────────────────────────────────────────────────────────
# LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Loading datasets ...")
fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
real_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
cb_df   = pd.read_csv(os.path.join(DATA_DIR, "clickbait.csv"))

fake_df["label"]     = "fake"
fake_df["label_int"] = 1
real_df["label"]     = "real"
real_df["label_int"] = 0

news_df = pd.concat([fake_df, real_df]).reset_index(drop=True)
news_df["text"] = news_df["text"].fillna("").astype(str)

# Text stats
news_df["char_count"] = news_df["text"].str.len()
news_df["word_count"] = news_df["text"].str.split().str.len()

print(f"   News corpus : {len(news_df):,} rows  |  Fake={fake_df.shape[0]:,}  Real={real_df.shape[0]:,}")
print(f"   Clickbait   : {len(cb_df):,} rows")

# ─────────────────────────────────────────────────────────────
# BUILD TRAIN / TEST SPLIT (same seed as training)
# ─────────────────────────────────────────────────────────────
X_all = fake_vec.transform(news_df["text"])
y_all = news_df["label_int"]
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_all, y_all, news_df.index, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────
# 1-7: MODEL METRICS
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Computing model metrics ...")

perf_rows   = []
radar_rows  = []
cm_rows     = []
pcls_rows   = []
roc_rows    = []
pr_rows     = []
conf_rows   = []

for name, mdl in fake_models.items():
    y_pred  = mdl.predict(X_te)
    y_proba = mdl.predict_proba(X_te)[:, 1]

    acc      = accuracy_score(y_te, y_pred)
    prec     = precision_score(y_te, y_pred, zero_division=0)
    rec      = recall_score(y_te, y_pred, zero_division=0)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    auc_val  = roc_auc_score(y_te, y_proba)

    perf_rows.append({
        "model": name, "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1_score": round(f1, 4), "roc_auc": round(auc_val, 4),
        "test_samples": int(X_te.shape[0]), "train_samples": int(X_tr.shape[0]),
    })

    for metric, val in [("Accuracy", acc), ("Precision", prec),
                        ("Recall",   rec), ("F1 Score",  f1), ("ROC AUC", auc_val)]:
        radar_rows.append({"model": name, "metric": metric, "value": round(val, 4)})

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    for cell_label, val in [("True Positive", int(tp)), ("True Negative", int(tn)),
                             ("False Positive", int(fp)), ("False Negative", int(fn))]:
        cm_rows.append({"model": name, "cell": cell_label, "count": val,
                        "predicted": "Fake" if "Positive" in cell_label else "Real",
                        "actual":    "Fake" if "True" in cell_label else "Real"})

    rpt = classification_report(y_te, y_pred, target_names=["real", "fake"],
                                output_dict=True, zero_division=0)
    for cls in ["real", "fake"]:
        d = rpt[cls]
        pcls_rows.append({
            "model": name, "class": cls,
            "precision": round(d["precision"], 4),
            "recall":    round(d["recall"],    4),
            "f1_score":  round(d["f1-score"],  4),
            "support":   int(d["support"]),
        })

    fpr_arr, tpr_arr, _ = roc_curve(y_te, y_proba)
    # Downsample ROC to max 200 points per model for manageable CSV size
    step = max(1, len(fpr_arr) // 200)
    for fpr_v, tpr_v in zip(fpr_arr[::step], tpr_arr[::step]):
        roc_rows.append({"model": name,
                         "fpr": round(float(fpr_v), 6),
                         "tpr": round(float(tpr_v), 6),
                         "auc": round(auc_val, 4)})

    prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_proba)
    step = max(1, len(prec_arr) // 200)
    for pr_v, re_v in zip(prec_arr[::step], rec_arr[::step]):
        pr_rows.append({"model": name,
                        "precision": round(float(pr_v), 6),
                        "recall":    round(float(re_v), 6)})

    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(y_proba, bins=bins)
    for i, cnt in enumerate(counts):
        conf_rows.append({
            "model": name,
            "bin_low":   round(bins[i],   2),
            "bin_high":  round(bins[i+1], 2),
            "bin_label": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "count": int(cnt),
        })

pd.DataFrame(perf_rows ).to_csv(os.path.join(OUT_DIR, "model_performance.csv"),     index=False)
pd.DataFrame(radar_rows).to_csv(os.path.join(OUT_DIR, "model_comparison_radar.csv"),index=False)
pd.DataFrame(cm_rows   ).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"),      index=False)
pd.DataFrame(pcls_rows ).to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"),     index=False)
pd.DataFrame(roc_rows  ).to_csv(os.path.join(OUT_DIR, "roc_curve.csv"),             index=False)
pd.DataFrame(pr_rows   ).to_csv(os.path.join(OUT_DIR, "pr_curve.csv"),              index=False)
pd.DataFrame(conf_rows ).to_csv(os.path.join(OUT_DIR, "prediction_confidence.csv"), index=False)
print("   Model metric CSVs written (7 files).")

# ─────────────────────────────────────────────────────────────
# 8. NEWS DISTRIBUTION
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Computing dataset statistics ...")

dist_df = (
    news_df.groupby("label")
    .agg(
        count         = ("label",      "count"),
        avg_char_len  = ("char_count",  "mean"),
        avg_word_count= ("word_count",  "mean"),
        median_words  = ("word_count",  "median"),
        min_words     = ("word_count",  "min"),
        max_words     = ("word_count",  "max"),
    )
    .round(1)
    .reset_index()
)
dist_df.to_csv(os.path.join(OUT_DIR, "news_distribution.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 9. WORD LENGTH DISTRIBUTION (histogram by label)
# ─────────────────────────────────────────────────────────────
wl_rows = []
bins = list(range(0, 1001, 50)) + [float("inf")]
labels_list = [f"{bins[i]}-{bins[i+1]}" if bins[i+1] != float("inf") else "1000+"
               for i in range(len(bins)-1)]
for lbl in ["fake", "real"]:
    wc = news_df[news_df["label"] == lbl]["word_count"]
    counts, _ = np.histogram(wc, bins=[b for b in bins if b != float("inf")] + [10000])
    for i, cnt in enumerate(counts):
        wl_rows.append({"label": lbl, "word_range": labels_list[i], "count": int(cnt),
                        "bin_low": int(bins[i])})
pd.DataFrame(wl_rows).sort_values(["label","bin_low"]).to_csv(
    os.path.join(OUT_DIR, "word_length_distribution.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 10. CLICKBAIT DISTRIBUTION
# ─────────────────────────────────────────────────────────────
cb_df["category"] = cb_df["label"].map({1: "clickbait", 0: "not clickbait"})
cb_dist = cb_df["category"].value_counts().reset_index()
cb_dist.columns = ["category", "count"]
cb_dist["pct"] = (cb_dist["count"] / cb_dist["count"].sum() * 100).round(2)
cb_dist.to_csv(os.path.join(OUT_DIR, "clickbait_distribution.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 11. CLICKBAIT PREDICTIONS (full dataset)
# ─────────────────────────────────────────────────────────────
X_cb     = click_vec.transform(cb_df["headline"].fillna("").astype(str))
cb_pred  = click_mdl.predict(X_cb)
cb_proba = click_mdl.predict_proba(X_cb)

cb_pred_df = cb_df[["headline", "label"]].copy()
cb_pred_df.columns = ["headline", "true_label"]
cb_pred_df["true_category"]      = cb_pred_df["true_label"].map({1: "clickbait", 0: "not clickbait"})
cb_pred_df["predicted_label"]    = cb_pred.astype(int)
cb_pred_df["predicted_category"] = pd.Series(cb_pred).map({1: "clickbait", 0: "not clickbait"}).values
cb_pred_df["confidence"]         = np.max(cb_proba, axis=1).round(4)
cb_pred_df["prob_clickbait"]     = cb_proba[:, 1].round(4)
cb_pred_df["correct"]            = (cb_pred_df["true_label"] == cb_pred_df["predicted_label"]).astype(int)
cb_pred_df["headline_length"]    = cb_pred_df["headline"].str.len()
cb_pred_df["word_count"]         = cb_pred_df["headline"].str.split().str.len()
cb_pred_df.to_csv(os.path.join(OUT_DIR, "clickbait_predictions.csv"), index=False)

cb_acc = cb_pred_df["correct"].mean() * 100

# ─────────────────────────────────────────────────────────────
# 12. SENTIMENT DISTRIBUTION  &  13. SENTIMENT BY CLASS
# ─────────────────────────────────────────────────────────────
sample_n    = min(10_000, len(news_df))
sent_sample = news_df.sample(sample_n, random_state=42).copy()

X_s         = sent_vec.transform(sent_sample["text"])
sent_preds  = sent_mdl.predict(X_s)
sent_probas = sent_mdl.predict_proba(X_s)

sent_sample["sentiment"] = sent_preds

sent_dist = sent_sample["sentiment"].value_counts().reset_index()
sent_dist.columns = ["sentiment", "count"]
sent_dist["pct"] = (sent_dist["count"] / sent_dist["count"].sum() * 100).round(2)
sent_dist.to_csv(os.path.join(OUT_DIR, "sentiment_distribution.csv"), index=False)

sent_by_cls = (
    sent_sample.groupby(["label", "sentiment"])
    .size()
    .reset_index(name="count")
)
total_per_label = sent_sample.groupby("label").size().reset_index(name="total")
sent_by_cls = sent_by_cls.merge(total_per_label, on="label")
sent_by_cls["pct"] = (sent_by_cls["count"] / sent_by_cls["total"] * 100).round(2)
sent_by_cls.to_csv(os.path.join(OUT_DIR, "sentiment_by_class.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 14. TOP TERMS
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Computing top terms & feature importance ...")

def top_tfidf_terms(texts, label, top_n=20):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms  = tfidf.get_feature_names_out()
    df = pd.DataFrame({"term": terms, "tfidf_score": scores, "label": label})
    return df.nlargest(top_n, "tfidf_score").reset_index(drop=True)

fake_terms = top_tfidf_terms(news_df[news_df["label"] == "fake"]["text"], "fake")
real_terms = top_tfidf_terms(news_df[news_df["label"] == "real"]["text"], "real")
cb_terms   = top_tfidf_terms(cb_df[cb_df["label"] == 1]["headline"].fillna(""), "clickbait")
ncb_terms  = top_tfidf_terms(cb_df[cb_df["label"] == 0]["headline"].fillna(""), "not clickbait")

top_terms_df = pd.concat([fake_terms, real_terms, cb_terms, ncb_terms]).reset_index(drop=True)
top_terms_df["tfidf_score"] = top_terms_df["tfidf_score"].round(6)
top_terms_df["rank"]        = top_terms_df.groupby("label").cumcount() + 1
top_terms_df.to_csv(os.path.join(OUT_DIR, "top_terms.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 15. FEATURE IMPORTANCE  (LR coefficients)
# ─────────────────────────────────────────────────────────────
lr_mdl = fake_models["Logistic Regression"]
try:
    coefs    = lr_mdl.coef_[0]
    features = fake_vec.get_feature_names_out()
    feat_df  = pd.DataFrame({"feature": features, "coefficient": coefs})
    top_fake = feat_df.nlargest(25, "coefficient").assign(direction="fake indicator")
    top_real = feat_df.nsmallest(25, "coefficient").assign(direction="real indicator")
    feat_out = pd.concat([top_fake, top_real]).reset_index(drop=True)
    feat_out["abs_coefficient"] = feat_out["coefficient"].abs().round(6)
    feat_out["coefficient"]     = feat_out["coefficient"].round(6)
    feat_out.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
except Exception as e:
    print(f"   Skipped feature importance: {e}")

# ─────────────────────────────────────────────────────────────
# 16. DATASET SUMMARY  (KPI cards)
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Writing summary / KPI data ...")

best_model = max(perf_rows, key=lambda r: r["f1_score"])

summary_rows = [
    {"metric": "Total News Articles",        "value": len(news_df),                                      "unit": "articles",  "category": "Data"},
    {"metric": "Fake Articles",              "value": fake_df.shape[0],                                  "unit": "articles",  "category": "Data"},
    {"metric": "Real Articles",              "value": real_df.shape[0],                                  "unit": "articles",  "category": "Data"},
    {"metric": "Total Clickbait Headlines",  "value": len(cb_df),                                        "unit": "headlines", "category": "Data"},
    {"metric": "Clickbait Headlines",        "value": int((cb_df["label"] == 1).sum()),                  "unit": "headlines", "category": "Data"},
    {"metric": "Non-Clickbait Headlines",    "value": int((cb_df["label"] == 0).sum()),                  "unit": "headlines", "category": "Data"},
    {"metric": "Fake % of Corpus",           "value": round(fake_df.shape[0] / len(news_df) * 100, 1),  "unit": "%",         "category": "Data"},
    {"metric": "Avg Fake Article Words",     "value": round(news_df[news_df["label"]=="fake"]["word_count"].mean(), 1), "unit": "words", "category": "Data"},
    {"metric": "Avg Real Article Words",     "value": round(news_df[news_df["label"]=="real"]["word_count"].mean(), 1), "unit": "words", "category": "Data"},
    {"metric": "Models Trained",             "value": len(fake_models),                                  "unit": "models",    "category": "Model"},
    {"metric": "Best Model",                 "value": best_model["model"],                               "unit": "name",      "category": "Model"},
    {"metric": "Best Model Accuracy",        "value": round(best_model["accuracy"]  * 100, 2),           "unit": "%",         "category": "Model"},
    {"metric": "Best Model Precision",       "value": round(best_model["precision"] * 100, 2),           "unit": "%",         "category": "Model"},
    {"metric": "Best Model Recall",          "value": round(best_model["recall"]    * 100, 2),           "unit": "%",         "category": "Model"},
    {"metric": "Best Model F1 Score",        "value": round(best_model["f1_score"]  * 100, 2),           "unit": "%",         "category": "Model"},
    {"metric": "Best Model ROC AUC",         "value": round(best_model["roc_auc"]   * 100, 2),           "unit": "%",         "category": "Model"},
    {"metric": "Clickbait Model Accuracy",   "value": round(cb_acc, 2),                                  "unit": "%",         "category": "Model"},
    {"metric": "Test Set Size",              "value": int(X_te.shape[0]),                                "unit": "samples",   "category": "Split"},
    {"metric": "Train Set Size",             "value": int(X_tr.shape[0]),                                "unit": "samples",   "category": "Split"},
    {"metric": "Test Split %",               "value": 20,                                                "unit": "%",         "category": "Split"},
    {"metric": "Vocabulary Size (Fake)",     "value": len(fake_vec.vocabulary_),                         "unit": "tokens",    "category": "Features"},
    {"metric": "Vocabulary Size (Click)",    "value": len(click_vec.vocabulary_),                        "unit": "tokens",    "category": "Features"},
]
pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "dataset_summary.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Done! All CSVs saved to:")
print(f"  {OUT_DIR}")
print("=" * 62)

all_files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".csv"))
for f in all_files:
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    with open(os.path.join(OUT_DIR, f), encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1
    print(f"  {f:<45} {rows:>7,} rows  {size/1024:>7.1f} KB")

print("\nTableau data generation complete.")
