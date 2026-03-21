# Fine-tuned DistilBERT multi-label emotion classifier on the GoEmotions dataset.
# DistilBERT understands word context and sentence structure — unlike TF-IDF which
# treats text as a bag of words. This is the long-term replacement for own_emotion_anaylsis.py.
#
# Requirements:
#   pip install transformers datasets torch scikit-learn

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 1. Load official GoEmotions splits
# ---------------------------------------------------------------------------
# Using the paper's train/dev/test splits makes results directly comparable
# to published benchmarks. train → fine-tune, dev → val during training, test → final eval.
emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral',
]
NUM_LABELS = len(emotion_cols)

data_dir = Path(__file__).resolve().parent / "data" / "full_dataset"

def load_tsv(path):
    """Parse a GoEmotions TSV (text, comma-sep emotion IDs, comment ID) into
    a list of texts and a float32 binary label matrix."""
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'labels', 'id'])
    matrix = np.zeros((len(df), NUM_LABELS), dtype=np.float32)
    for i, ids_str in enumerate(df['labels']):
        for idx in str(ids_str).split(','):
            matrix[i, int(idx.strip())] = 1.0
    return df['text'].tolist(), matrix

X_train, y_train = load_tsv(data_dir / "train.tsv")
X_dev,   y_dev   = load_tsv(data_dir / "dev.tsv")
X_test,  y_test  = load_tsv(data_dir / "test.tsv")

# ---------------------------------------------------------------------------
# 2. Tokenise
# ---------------------------------------------------------------------------
# DistilBERT is a lighter, faster version of BERT — 40% smaller, 60% faster,
# retains 97% of BERT's performance on most NLP benchmarks.
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


train_dataset = EmotionDataset(X_train, y_train)
dev_dataset   = EmotionDataset(X_dev,   y_dev)
test_dataset  = EmotionDataset(X_test,  y_test)

# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
# problem_type="multi_label_classification" makes the model use BCEWithLogitsLoss
# (one independent sigmoid per label) instead of softmax across labels.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _apply_thresholds(probs, thresholds):
    return (probs >= thresholds).astype(int)


def _macro_metrics(labels, probs, thresholds):
    preds = _apply_thresholds(probs, thresholds)
    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    f1        = f1_score(labels, preds, average="macro", zero_division=0)

    valid = [i for i in range(labels.shape[1]) if len(np.unique(labels[:, i])) > 1]
    auc_roc = roc_auc_score(labels[:, valid], probs[:, valid], average="macro")
    return accuracy, precision, recall, f1, auc_roc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = _sigmoid(logits)
    accuracy, precision, recall, f1, auc_roc = _macro_metrics(
        labels, probs, thresholds=0.5
    )

    return {
        "exact_match_accuracy": accuracy,
        "macro_precision":      precision,
        "macro_recall":         recall,
        "macro_f1":             f1,
        "macro_auc_roc":        auc_roc,
    }


def tune_thresholds(logits, labels, candidate_thresholds=None):
    probs = _sigmoid(logits)
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.1, 0.9, 17)

    best_thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            continue
        best_f1 = -1.0
        for t in candidate_thresholds:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert_emotion_output",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,           # small fixed warmup to stabilize early training
    weight_decay=0.01,          # L2 regularisation to prevent overfitting
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # use half-precision on GPU for speed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,      # dev used for checkpoint selection during training
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------------------------------------------------------------------------
# 6. Final evaluation on held-out test set
# ---------------------------------------------------------------------------
# Evaluate on the test split — never seen during training or checkpoint selection.
dev_pred = trainer.predict(dev_dataset)
thresholds = tune_thresholds(dev_pred.predictions, dev_pred.label_ids)

print("\n" + "=" * 50)
print("  DistilBERT — Final Evaluation (test set)")
print("=" * 50)
test_pred = trainer.predict(test_dataset)
test_probs = _sigmoid(test_pred.predictions)
accuracy, precision, recall, f1, auc_roc = _macro_metrics(
    test_pred.label_ids, test_probs, thresholds=thresholds
)
results = {
    "exact_match_accuracy": accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "macro_auc_roc": auc_roc,
}
for k, v in results.items():
    print(f"  {k:<30} {v:.3f}")

# ---------------------------------------------------------------------------
# 7. Interactive prediction — type any text and the model will score it
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text: str, top_n: int = 5) -> None:
    """Tokenise a single text and print the top_n emotions with confidence scores."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    # Move inputs to the same device as the model (CPU or GPU).
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()

    # Convert raw logits → probabilities with sigmoid (one per emotion).
    probs = _sigmoid(logits)

    # Apply the per-label thresholds tuned on dev set.
    detected = [emotion_cols[i] for i, p in enumerate(probs) if p >= thresholds[i]]

    # Rank all emotions by confidence for the top-N display.
    top_idx = np.argsort(probs)[::-1][:top_n]

    print(f"\n  Input    : \"{text}\"")
    print(f"  Detected : {', '.join(detected) if detected else 'none above threshold'}")
    print(f"\n  DistilBERT — top {top_n} emotions:")
    for i in top_idx:
        marker = " ◄" if probs[i] >= thresholds[i] else ""
        bar    = "█" * int(probs[i] * 20)
        print(f"    {emotion_cols[i]:<18} {probs[i]:.2%}  {bar}{marker}")

print("\n" + "=" * 50)
print("  Emotion Predictor  (type 'quit' to exit)")
print("  ◄ = above detection threshold")
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
