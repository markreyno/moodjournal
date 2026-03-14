# Multi-label emotion classifier trained on the GoEmotions dataset.
# Each piece of text can be assigned multiple emotion labels simultaneously.
# GoEmotions dataset: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
# https://github.com/google-research/google-research/tree/master/goemotions

import pandas as pd

# The dataset is split across 3 CSV files — load and merge them.
# We sample 20k rows to keep training time manageable during development;
# remove .sample() and restore n_estimators=100 when training for real results.
df1 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_1.csv")
df2 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_2.csv")
df3 = pd.read_csv("C:/Users/reyno/moodjournal/models/test_models/data/full_dataset/goemotions_3.csv")

combineddf = pd.concat([df1, df2, df3]).sample(n=20000, random_state=42)

print(combineddf.head())
print(combineddf.tail())
print(combineddf.shape)
print(combineddf.columns)

# Drop metadata columns that carry no signal for emotion prediction.
combineddf.drop(columns=["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id"], inplace=True)

print(combineddf.columns)

# All 28 GoEmotions labels. Each is a binary column (1 = labeler assigned that emotion, 0 = not).
# A single text can have multiple 1s, making this a multi-label problem.
emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# X is the raw text input; y is the full binary label matrix (one column per emotion).
X = combineddf["text"]
y = combineddf[emotion_cols]

print("------------------------------------------------")
print("Splitting the dataset into training and testing sets")
print("------------------------------------------------")

from sklearn.model_selection import train_test_split

# Hold out 20% of data for evaluation so we test on text the models have never seen.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1))

print("------------------------------------------------")
print("Training the models")
print("------------------------------------------------")

rf_model.fit(X_train_tfidf, y_train)
nb_model.fit(X_train_tfidf, y_train)

print("------------------------------------------------")
print("Evaluating the models")
print("------------------------------------------------")

from sklearn.metrics import accuracy_score, classification_report, f1_score

rf_pred = rf_model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_tfidf)

# Exact-match accuracy: a prediction only counts as correct if every emotion label matches.
# This is a strict metric — Macro F1 below gives a fairer picture for imbalanced labels.
accuracy = accuracy_score(y_test, rf_pred)

# Macro F1 averages the F1 score across all 28 emotion labels equally,
# so rare emotions (e.g. grief, pride) are weighted the same as common ones (e.g. neutral).
for name, model in [("Naive Bayes", nb_model), ("Random Forest", rf_model)]:
    y_pred = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n{name} — Macro F1: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=emotion_cols, zero_division=0))
