"""
Reusable model evaluation framework for comparing emotion/sentiment models.

Supports three model types:
  - "multilabel"  : multi-label classifiers (Naive Bayes, Random Forest on GoEmotions)
  - "multiclass"  : single-label classifiers (BERT emotion pipeline)
  - "sentiment"   : three-class sentiment classifiers (VADER: positive/negative/neutral)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    model_type: str = "multilabel",
    labels: list[str] | None = None,
    inference_times: list[float] | None = None,
) -> dict:
    """Evaluate a model and return a metrics dictionary.

    Parameters
    ----------
    model_name : str
        Human-readable name shown in the printed report.
    y_true : array-like
        Ground-truth labels.  Shape (n_samples,) for multiclass/sentiment,
        (n_samples, n_labels) for multilabel.
    y_pred : array-like
        Predicted labels in the same shape as y_true.
    model_type : str
        One of "multilabel", "multiclass", or "sentiment".
    labels : list[str] | None
        Label names used in the classification report.
    inference_times : list[float] | None
        Per-sample inference durations in seconds.  When provided, latency
        and throughput are included in the report.

    Returns
    -------
    dict
        All computed metrics keyed by name, plus "model_name" and
        "model_type".
    """
    if model_type not in {"multilabel", "multiclass", "sentiment"}:
        raise ValueError(
            f"model_type must be 'multilabel', 'multiclass', or 'sentiment', got '{model_type}'"
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics: dict = {"model_name": model_name, "model_type": model_type}

    _HEADER = f"\n{'=' * 70}\n  Model : {model_name}  [{model_type}]\n{'=' * 70}"
    print(_HEADER)

    # ------------------------------------------------------------------
    # Multilabel metrics
    # ------------------------------------------------------------------
    if model_type == "multilabel":
        metrics["exact_match_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
        metrics["jaccard_score_macro"] = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["jaccard_score_weighted"] = jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Exact-match accuracy : {metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming loss         : {metrics['hamming_loss']:.4f}")
        print(f"  Jaccard (macro)      : {metrics['jaccard_score_macro']:.4f}")
        print(f"  Jaccard (weighted)   : {metrics['jaccard_score_weighted']:.4f}")
        print(f"  Precision (macro)    : {metrics['precision_macro']:.4f}")
        print(f"  Recall    (macro)    : {metrics['recall_macro']:.4f}")
        print(f"  F1        (macro)    : {metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted) : {metrics['precision_weighted']:.4f}")
        print(f"  Recall    (weighted) : {metrics['recall_weighted']:.4f}")
        print(f"  F1        (weighted) : {metrics['f1_weighted']:.4f}")

        print("\n  Per-label classification report:")
        print(
            classification_report(
                y_true, y_pred, target_names=labels, zero_division=0
            )
        )

    # ------------------------------------------------------------------
    # Multiclass metrics (BERT, etc.)
    # ------------------------------------------------------------------
    elif model_type in {"multiclass", "sentiment"}:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy             : {metrics['accuracy']:.4f}")
        print(f"  Precision (macro)    : {metrics['precision_macro']:.4f}")
        print(f"  Recall    (macro)    : {metrics['recall_macro']:.4f}")
        print(f"  F1        (macro)    : {metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted) : {metrics['precision_weighted']:.4f}")
        print(f"  Recall    (weighted) : {metrics['recall_weighted']:.4f}")
        print(f"  F1        (weighted) : {metrics['f1_weighted']:.4f}")

        print("\n  Classification report:")
        print(
            classification_report(
                y_true, y_pred, target_names=labels, zero_division=0
            )
        )

    # ------------------------------------------------------------------
    # Speed metrics (optional)
    # ------------------------------------------------------------------
    if inference_times is not None and len(inference_times) > 0:
        times = np.array(inference_times)
        metrics["mean_latency_ms"] = float(times.mean() * 1000)
        metrics["p50_latency_ms"] = float(np.percentile(times, 50) * 1000)
        metrics["p95_latency_ms"] = float(np.percentile(times, 95) * 1000)
        metrics["throughput_samples_per_sec"] = float(1.0 / times.mean())

        print(f"\n  Inference speed ({len(times)} samples):")
        print(f"    Mean latency   : {metrics['mean_latency_ms']:.2f} ms/sample")
        print(f"    P50 latency    : {metrics['p50_latency_ms']:.2f} ms/sample")
        print(f"    P95 latency    : {metrics['p95_latency_ms']:.2f} ms/sample")
        print(f"    Throughput     : {metrics['throughput_samples_per_sec']:.1f} samples/sec")

    print(f"{'=' * 70}\n")
    return metrics


# ---------------------------------------------------------------------------
# Adapter: scikit-learn multi-label models (Naive Bayes / Random Forest)
# ---------------------------------------------------------------------------

def run_sklearn_evaluation(
    named_models: list[tuple[str, Any]],
    X_test_tfidf: Any,
    y_test: Any,
    emotion_cols: list[str],
) -> list[dict]:
    """Evaluate one or more scikit-learn MultiOutputClassifier models.

    Parameters
    ----------
    named_models : list of (name, fitted_model) tuples
    X_test_tfidf : sparse matrix
        TF-IDF encoded test features (already transformed, not fitted here).
    y_test : array-like, shape (n_samples, n_labels)
        Ground-truth binary label matrix.
    emotion_cols : list[str]
        Ordered list of emotion label names matching y_test columns.

    Returns
    -------
    list[dict]
        One metrics dict per model.
    """
    results = []
    for name, model in named_models:
        inference_times: list[float] = []
        preds = []
        X_array = X_test_tfidf  # predict all at once but time it per-row for fairness
        n = X_test_tfidf.shape[0]

        for i in range(n):
            row = X_test_tfidf[i]
            t0 = time.perf_counter()
            pred = model.predict(row)
            inference_times.append(time.perf_counter() - t0)
            preds.append(pred[0])

        y_pred = np.array(preds)
        metrics = evaluate_model(
            model_name=name,
            y_true=np.array(y_test),
            y_pred=y_pred,
            model_type="multilabel",
            labels=emotion_cols,
            inference_times=inference_times,
        )
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Adapter: HuggingFace BERT emotion pipeline
# ---------------------------------------------------------------------------

def run_bert_evaluation(
    texts: list[str],
    true_labels: list[str],
) -> dict:
    """Evaluate the distilbert-base-uncased-emotion pipeline.

    The pipeline returns ranked scores for six emotions:
      sadness, joy, love, anger, fear, surprise

    The predicted label is the emotion with the highest score.

    Parameters
    ----------
    texts : list[str]
        Raw text samples to classify.
    true_labels : list[str]
        Ground-truth emotion label for each text (must be one of the six
        emotions the pipeline produces).

    Returns
    -------
    dict
        Metrics from evaluate_model().
    """
    from transformers import pipeline as hf_pipeline

    print("Loading BERT emotion pipeline…")
    emotion_pipeline = hf_pipeline(
        "sentiment-analysis",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None,
    )

    bert_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    pred_labels: list[str] = []
    inference_times: list[float] = []

    for text in texts:
        t0 = time.perf_counter()
        result = emotion_pipeline(text)
        inference_times.append(time.perf_counter() - t0)
        # result is a list of lists when top_k=None; pick highest-score emotion
        top = max(result[0], key=lambda x: x["score"])
        pred_labels.append(top["label"].lower())

    return evaluate_model(
        model_name="BERT (distilbert-base-uncased-emotion)",
        y_true=true_labels,
        y_pred=pred_labels,
        model_type="multiclass",
        labels=bert_labels,
        inference_times=inference_times,
    )


# ---------------------------------------------------------------------------
# Adapter: VADER sentiment analysis
# ---------------------------------------------------------------------------

def run_vader_evaluation(
    texts: list[str],
    true_sentiments: list[str],
) -> dict:
    """Evaluate VADER sentiment analysis.

    Compound score thresholds (matching sentiment_analysis_nltk.py):
      >= 0.05  → "Positive"
      <= -0.05 → "Negative"
      else     → "Neutral"

    Parameters
    ----------
    texts : list[str]
        Raw text samples.
    true_sentiments : list[str]
        Ground-truth labels; each must be "Positive", "Negative", or "Neutral".

    Returns
    -------
    dict
        Metrics from evaluate_model().
    """
    import nltk
    from nltk import data as nltk_data
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk_data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    analyzer = SentimentIntensityAnalyzer()
    pred_sentiments: list[str] = []
    inference_times: list[float] = []

    for text in texts:
        t0 = time.perf_counter()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        inference_times.append(time.perf_counter() - t0)
        pred_sentiments.append(label)

    return evaluate_model(
        model_name="VADER Sentiment Analysis",
        y_true=true_sentiments,
        y_pred=pred_sentiments,
        model_type="sentiment",
        labels=["Negative", "Neutral", "Positive"],
        inference_times=inference_times,
    )


# ---------------------------------------------------------------------------
# Cross-model comparison table
# ---------------------------------------------------------------------------

def compare_models(results: list[dict]) -> None:
    """Print a side-by-side comparison table for a list of evaluated models.

    Parameters
    ----------
    results : list[dict]
        Each dict is the return value of evaluate_model().
    """
    if not results:
        print("No results to compare.")
        return

    col_width = 30
    metric_keys = [
        ("F1 (macro)",          "f1_macro"),
        ("F1 (weighted)",       "f1_weighted"),
        ("Precision (macro)",   "precision_macro"),
        ("Recall (macro)",      "recall_macro"),
        ("Exact-match acc.",    "exact_match_accuracy"),
        ("Accuracy",            "accuracy"),
        ("Hamming loss",        "hamming_loss"),
        ("Jaccard (macro)",     "jaccard_score_macro"),
        ("Mean latency (ms)",   "mean_latency_ms"),
        ("Throughput (s/s)",    "throughput_samples_per_sec"),
    ]

    names = [r["model_name"] for r in results]
    header_row = f"{'Metric':<28}" + "".join(f"{n[:col_width]:<{col_width}}" for n in names)

    print("\n" + "=" * (28 + col_width * len(results)))
    print("  MODEL COMPARISON")
    print("=" * (28 + col_width * len(results)))
    print(header_row)
    print("-" * (28 + col_width * len(results)))

    for label, key in metric_keys:
        row_values = []
        any_present = False
        for r in results:
            if key in r:
                row_values.append(f"{r[key]:.4f}")
                any_present = True
            else:
                row_values.append("—")
        if any_present:
            row = f"  {label:<26}" + "".join(f"{v:<{col_width}}" for v in row_values)
            print(row)

    print("=" * (28 + col_width * len(results)) + "\n")


# ---------------------------------------------------------------------------
# __main__ — wires all three evaluations together
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier

    DATA_DIR = "C:/Users/reyno/moodjournal/models/test_models/data/full_dataset"

    # ------------------------------------------------------------------
    # Load & prepare GoEmotions data (shared across sklearn + BERT + VADER)
    # ------------------------------------------------------------------
    print("Loading GoEmotions dataset…")
    df = pd.concat([
        pd.read_csv(f"{DATA_DIR}/goemotions_1.csv"),
        pd.read_csv(f"{DATA_DIR}/goemotions_2.csv"),
        pd.read_csv(f"{DATA_DIR}/goemotions_3.csv"),
    ]).sample(n=20000, random_state=42)

    drop_cols = ["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    emotion_cols = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral",
    ]

    X = df["text"]
    y = df[emotion_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------------------------------------------
    # sklearn models — train & evaluate
    # ------------------------------------------------------------------
    print("Vectorizing…")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Naive Bayes…")
    nb_model = MultiOutputClassifier(MultinomialNB())
    nb_model.fit(X_train_tfidf, y_train)

    print("Training Random Forest…")
    rf_model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    )
    rf_model.fit(X_train_tfidf, y_train)

    # Use a small subset for per-row timing (full 4k rows is slow per-row)
    EVAL_SUBSET = 200
    X_eval = X_test_tfidf[:EVAL_SUBSET]
    y_eval = y_test.iloc[:EVAL_SUBSET]

    sklearn_results = run_sklearn_evaluation(
        named_models=[("Naive Bayes (TF-IDF)", nb_model), ("Random Forest (TF-IDF)", rf_model)],
        X_test_tfidf=X_eval,
        y_test=y_eval,
        emotion_cols=emotion_cols,
    )

    all_results = list(sklearn_results)

    # ------------------------------------------------------------------
    # VADER — derive sentiment labels from GoEmotions and evaluate
    # ------------------------------------------------------------------
    # Map 28-label multi-hot rows to a single coarse sentiment:
    # joy/love/admiration/amusement/gratitude/optimism/pride/relief → Positive
    # anger/annoyance/disappointment/disapproval/disgust/grief/remorse/sadness → Negative
    # everything else → Neutral
    POSITIVE_EMOTIONS = {"joy", "love", "admiration", "amusement", "gratitude", "optimism", "pride", "relief"}
    NEGATIVE_EMOTIONS = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "grief", "remorse", "sadness"}

    def row_to_sentiment(row: pd.Series) -> str:
        active = {col for col in emotion_cols if row[col] == 1}
        if active & POSITIVE_EMOTIONS:
            return "Positive"
        if active & NEGATIVE_EMOTIONS:
            return "Negative"
        return "Neutral"

    test_df = df.loc[X_test.index].copy()
    test_df["sentiment"] = test_df[emotion_cols].apply(row_to_sentiment, axis=1)
    vader_sample = test_df.sample(n=min(500, len(test_df)), random_state=42)

    vader_result = run_vader_evaluation(
        texts=vader_sample["text"].tolist(),
        true_sentiments=vader_sample["sentiment"].tolist(),
    )
    all_results.append(vader_result)

    # ------------------------------------------------------------------
    # BERT — evaluate on a small sample (inference is slow on CPU)
    # ------------------------------------------------------------------
    # Build a simple single-label subset: rows with exactly one active emotion
    # that falls within the 6 BERT labels.
    BERT_EMOTIONS = {"sadness", "joy", "love", "anger", "fear", "surprise"}
    bert_rows = []
    for _, row in test_df.iterrows():
        active = [col for col in BERT_EMOTIONS if row.get(col, 0) == 1]
        if len(active) == 1:
            bert_rows.append({"text": row["text"], "label": active[0]})
        if len(bert_rows) >= 100:
            break

    if bert_rows:
        bert_df = pd.DataFrame(bert_rows)
        bert_result = run_bert_evaluation(
            texts=bert_df["text"].tolist(),
            true_labels=bert_df["label"].tolist(),
        )
        all_results.append(bert_result)
    else:
        print("No suitable single-label BERT samples found; skipping BERT evaluation.")

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    compare_models(all_results)
