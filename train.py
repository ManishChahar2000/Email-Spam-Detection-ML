import os
import time
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_PATH = "dataset/cleaned_dataset.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42
CV_FOLDS = 5
SELECTION_METRIC = "f1"  # accuracy is misleading on imbalanced spam data

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "clean_text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'clean_text' and 'label' columns")

df = df.dropna(subset=["clean_text", "label"])

X_raw = df["clean_text"].astype(str)
y = df["label"]

print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{y.value_counts()}\n")

# ---------------------------------------------------------
# TF-IDF Vectorization
# ---------------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # unigrams + bigrams capture more spam patterns
    stop_words="english",
    sublinear_tf=True,       # log-scaled term frequency, generally stronger
    min_df=2
)

X = tfidf.fit_transform(X_raw)

# ---------------------------------------------------------
# Stratified Train/Test Split (preserves class ratio)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
}

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = []
best_model = None
best_model_name = None
best_score = -1

print("=" * 70)
print("MODEL TRAINING & EVALUATION")
print("=" * 70)

for name, model in models.items():
    start = time.time()

    # Cross-validation on training data (more reliable than a single split)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "train_time_sec": round(elapsed, 3)
    }
    results.append(metrics)

    print(f"\n{name}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1-score       : {metrics['f1']:.4f}")
    print(f"  CV F1 (mean±sd): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"  Train time (s) : {metrics['train_time_sec']}")
    print("  Confusion Matrix:")
    print(" ", confusion_matrix(y_test, pred))

    if metrics[SELECTION_METRIC] > best_score:
        best_score = metrics[SELECTION_METRIC]
        best_model = model
        best_model_name = name

# ---------------------------------------------------------
# Best model summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(f"Model    : {best_model_name}")
print(f"{SELECTION_METRIC.upper()} score: {best_score:.4f}")
print("\nFull classification report on test set:")
print(classification_report(y_test, best_model.predict(X_test)))

# ---------------------------------------------------------
# Save model, vectorizer, and metrics (useful for the paper too)
# ---------------------------------------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_model.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))

results_df = pd.DataFrame(results).sort_values(by=SELECTION_METRIC, ascending=False)
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison_results.csv"), index=False)

with open(os.path.join(MODEL_DIR, "best_model_info.json"), "w") as f:
    json.dump({
        "best_model": best_model_name,
        "selection_metric": SELECTION_METRIC,
        "score": best_score,
        "random_state": RANDOM_STATE,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": [1, 2]
    }, f, indent=2)

print(f"\nModel, vectorizer, and results saved to '{MODEL_DIR}/'")
print("model_comparison_results.csv is ready to drop straight into your results table.")