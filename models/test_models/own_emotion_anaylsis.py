# Multi-label emotion classifier trained on the GoEmotions dataset.
# Each piece of text can be assigned multiple emotion labels simultaneously.
# GoEmotions dataset: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
# https://github.com/google-research/google-research/tree/master/goemotions

from pathlib import Path

import numpy as np
import pandas as pd

# All 28 GoEmotions labels in index order (0=admiration … 27=neutral).
# The TSV files encode emotions as comma-separated integers, e.g. "2,15" means anger + gratitude.
emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_tsv(path):
    """Load an official GoEmotions TSV split into texts and a binary label matrix."""
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'labels', 'id'])
    matrix = np.zeros((len(df), len(emotion_cols)), dtype=int)
    for i, ids_str in enumerate(df['labels']):
        for idx in str(ids_str).split(','):
            matrix[i, int(idx.strip())] = 1
    return df['text'].tolist(), matrix

# Use the official GoEmotions splits so results are comparable to the paper.
# train + dev are both used for training (maximises training data for classical ML).
# test is held out for evaluation.
data_dir = Path(__file__).resolve().parent / "data" / "full_dataset"

X_train_raw, y_train_raw = load_tsv(data_dir / "train.tsv")
X_dev_raw,   y_dev_raw   = load_tsv(data_dir / "dev.tsv")
X_test,      y_test      = load_tsv(data_dir / "test.tsv")

X_train = X_train_raw + X_dev_raw
y_train = np.vstack([y_train_raw, y_dev_raw])

print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

print("------------------------------------------------")
print("Vectorizing the text data")
print("------------------------------------------------")

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF converts raw text into numeric feature vectors.
# max_features=10000 caps vocabulary size to avoid excessive memory use.
# ngram_range=(1,2) includes both single words and two-word phrases (bigrams),
# which helps capture context like "not happy" vs "happy".
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# fit_transform on train: learns the vocabulary and encodes in one step.
# transform on test: encodes using the already-learned vocabulary (no data leakage).
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("------------------------------------------------")
print("Wrapping the model")
print("------------------------------------------------")

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# MultiOutputClassifier trains one independent binary classifier per emotion label.
# Naive Bayes is fast and works well with sparse TF-IDF vectors.
# Random Forest is slower but typically more accurate; n_jobs=-1 uses all CPU cores.
nb_model = MultiOutputClassifier(MultinomialNB())
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

print("------------------------------------------------")
print("Training the models")
print("------------------------------------------------")

rf_model.fit(X_train_tfidf, y_train)
nb_model.fit(X_train_tfidf, y_train)

print("------------------------------------------------")
print("Evaluating the models")
print("------------------------------------------------")

from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)

# --- Metric explanations ---
# Accuracy      : fraction of samples where EVERY emotion label is predicted correctly.
#                 Very strict for multi-label — even one wrong label counts as a miss.
# Precision     : of all the emotion labels the model predicted as present, how many actually were?
#                 High precision = few false alarms (model is not trigger-happy).
# Recall        : of all the emotion labels that were truly present, how many did the model find?
#                 High recall = few misses (model catches most real emotions).
# F1-score      : harmonic mean of precision and recall, balancing both into one number.
#                 Useful when you care equally about false alarms and misses.
# AUC-ROC       : measures how well the model ranks emotions by confidence score.
#                 0.5 = random guessing, 1.0 = perfect separation. Robust to class imbalance
#                 because it looks at predicted probabilities, not just the final 0/1 decision.
# All multi-label metrics are averaged with 'macro', meaning each of the 28 emotions counts
# equally regardless of how often it appears in the data.

for name, model in [("Naive Bayes", nb_model), ("Random Forest", rf_model)]:
    y_pred = model.predict(X_test_tfidf)
    y_true = y_test

    # Exact-match accuracy — every label must be right for a sample to count as correct.
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-averaged precision, recall, F1 across all 28 emotion labels.
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC-ROC needs predicted probabilities, not hard 0/1 labels.
    # MultiOutputClassifier exposes one estimator per label in .estimators_;
    # we grab the positive-class probability (column index 1) from each.
    y_prob = np.column_stack(
        [est.predict_proba(X_test_tfidf)[:, 1] for est in model.estimators_]
    )
    # Skip any emotion columns where the test set has only one class (AUC is undefined there).
    valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    auc_roc = roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average='macro')

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Exact-Match Accuracy : {accuracy:.3f}")
    print(f"  Macro Precision      : {precision:.3f}")
    print(f"  Macro Recall         : {recall:.3f}")
    print(f"  Macro F1             : {f1:.3f}")
    print(f"  Macro AUC-ROC        : {auc_roc:.3f}  (over {len(valid_cols)}/28 labels)")
    print()
    print(classification_report(y_true, y_pred, target_names=emotion_cols, zero_division=0))

# ---------------------------------------------------------------------------
# Interactive prediction — type any text and both models will score it
# ---------------------------------------------------------------------------
def predict(text: str, top_n: int = 5) -> None:
    """Vectorise a single text and print the top_n emotions from each model."""
    vec = vectorizer.transform([text])

    print(f"\n  Input : \"{text}\"")
    print(f"  {'─' * 46}")

    for name, mdl in [("Naive Bayes", nb_model), ("Random Forest", rf_model)]:
        # Grab the positive-class probability for each of the 28 emotions.
        probs = np.array([
            est.predict_proba(vec)[0, 1] for est in mdl.estimators_
        ])
        # Sort descending and take top_n.
        top_idx = np.argsort(probs)[::-1][:top_n]

        print(f"\n  {name} — top {top_n} emotions:")
        for i in top_idx:
            bar = "█" * int(probs[i] * 20)
            print(f"    {emotion_cols[i]:<18} {probs[i]:.2%}  {bar}")

print("\n" + "=" * 50)
print("  Emotion Predictor  (type 'quit' to exit)")
print("=" * 50)

while True:
    try:
        user_input = input("\nEnter text: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if user_input.lower() in {"quit", "exit", "q"}:
        print("Exiting.")
        break

    if not user_input:
        print("  Please enter some text.")
        continue

    predict(user_input)
