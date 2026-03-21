#We will need to download libraries 
from pathlib import Path


import pandas as pd
import numpy as np
import torch

#will use sklearn to evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)


#using the train/dev/test splits from the goemotions dataset
#doing this instead of using the full dataset to make the results comparable to the paper
#all 28 GoEmotions labels in index order (0=admiration … 27=neutral).
emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral']
NUM_LABELS = len(emotion_cols)


#load the data
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

#tokenize the data
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

